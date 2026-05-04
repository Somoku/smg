//! Performance baseline benchmarks for the routing-loop subsystem.
//!
//! These benchmarks exercise the core data-structure operations that the
//! routing loop performs on every request — priority-queue push/pop,
//! version-partitioned multi-priority queue operations, and the mpsc
//! enqueue path — without requiring access to `pub(crate)` internals.
//!
//! The queue logic mirrors `routing_loop/queue.rs` exactly so that these
//! benchmarks remain a faithful proxy for the production hot path:
//!
//!   * `PriorityRequestQueue<T>` → `BinaryHeap<QueueEntry>` with a
//!     `(validation, version, final)` priority triple.
//!   * `MultiPriorityRequestQueue<T>` → `BTreeMap<i32, BinaryHeap<QueueEntry>>`.
//!   * Enqueue path → `tokio::sync::mpsc::unbounded_channel` send throughput.
//!
//! Run with:
//!   cargo bench --bench routing_loop_bench

use std::{
    collections::{BTreeMap, BinaryHeap},
    hint::black_box,
};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::runtime::Runtime;

// ---------------------------------------------------------------------------
// Minimal in-process replica of the queue data structures
// ---------------------------------------------------------------------------

/// Priority triple identical to the one computed in
/// `queue.rs → RequestPriority::priority()`.
///
/// Lower tuples are dispatched first; the heap is a max-heap so we negate
/// the values in `Ord` (same technique used in production).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueueEntry {
    /// `(validation_priority, version_priority, final_priority)`.
    /// Stored *negated* so `BinaryHeap` (max-heap) pops the lowest priority first.
    priority: (i64, i64, i64),
    /// Sequence number for FIFO tie-breaking (decreasing → earlier entries win
    /// in the max-heap because they have larger negated seq).
    seq: u64,
    /// Stand-in for the actual request payload.
    id: u64,
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower real priority = higher heap priority.
        // Negate both components so the heap pops the smallest original value.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Single-partition priority queue (mirrors `PriorityRequestQueue`).
struct SingleQueue {
    heap: BinaryHeap<QueueEntry>,
    seq: u64,
}

impl SingleQueue {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    fn push(&mut self, id: u64, validation: bool, version: i64, input_len: usize) {
        let validation_priority = i64::from(!validation);
        let version_priority = if version < 0 { i64::MAX } else { version };
        let final_priority = input_len.min(i32::MAX as usize) as i64; // ShortLength sort
        self.heap.push(QueueEntry {
            priority: (validation_priority, version_priority, final_priority),
            seq: self.seq,
            id,
        });
        self.seq = self.seq.wrapping_add(1);
    }

    fn pop(&mut self) -> Option<u64> {
        self.heap.pop().map(|e| e.id)
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

/// Version-partitioned queue (mirrors `MultiPriorityRequestQueue`).
struct MultiQueue {
    partitions: BTreeMap<i32, SingleQueue>,
    next_seq: u64,
}

impl MultiQueue {
    fn new() -> Self {
        Self {
            partitions: BTreeMap::new(),
            next_seq: 0,
        }
    }

    fn push(&mut self, id: u64, validation: bool, version: i64, input_len: usize) {
        let partition = if version < 0 {
            i32::MAX
        } else if version > i64::from(i32::MAX - 1) {
            i32::MAX - 1
        } else {
            version as i32
        };
        let q = self
            .partitions
            .entry(partition)
            .or_insert_with(SingleQueue::new);
        q.seq = self.next_seq;
        q.push(id, validation, version, input_len);
        self.next_seq = self.next_seq.wrapping_add(1);
    }

    fn pop(&mut self) -> Option<u64> {
        for q in self.partitions.values_mut() {
            if let Some(id) = q.pop() {
                return Some(id);
            }
        }
        None
    }

    fn remove_empty_partitions(&mut self) {
        self.partitions.retain(|_, q| q.len() > 0);
    }

    fn len(&self) -> usize {
        self.partitions.values().map(SingleQueue::len).sum()
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

fn make_single_queue(n: usize) -> SingleQueue {
    let mut q = SingleQueue::new();
    for i in 0..n {
        q.push(i as u64, false, (i % 4) as i64, i % 128);
    }
    q
}

fn make_multi_queue(n: usize, partitions: usize) -> MultiQueue {
    let mut q = MultiQueue::new();
    for i in 0..n {
        let version = (i % partitions) as i64;
        q.push(i as u64, false, version, i % 128);
    }
    q
}

// ---------------------------------------------------------------------------
// Benchmark 1 – Single-partition priority queue: push throughput
// ---------------------------------------------------------------------------

fn bench_single_queue_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/single/push");

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut q = SingleQueue::new();
                for i in 0..n {
                    q.push(black_box(i as u64), false, (i % 4) as i64, i % 128);
                }
                black_box(q.len());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2 – Single-partition priority queue: pop throughput
// ---------------------------------------------------------------------------

fn bench_single_queue_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/single/pop");

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || make_single_queue(n),
                |mut q| {
                    let mut count = 0u64;
                    while let Some(id) = q.pop() {
                        count += black_box(id);
                    }
                    black_box(count);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 3 – Single-partition queue: push+pop round-trip
// ---------------------------------------------------------------------------

fn bench_single_queue_push_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/single/push_pop");

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut q = SingleQueue::new();
                for i in 0..n {
                    q.push(i as u64, false, (i % 4) as i64, i % 128);
                }
                let mut total = 0u64;
                while let Some(id) = q.pop() {
                    total += black_box(id);
                }
                black_box(total);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 4 – Multi-partition queue: push throughput (2 / 8 / 32 versions)
// ---------------------------------------------------------------------------

fn bench_multi_queue_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/multi/push");

    for &n in &[1_000usize, 10_000] {
        for &partitions in &[2usize, 8, 32] {
            let param = format!("{n}_entries_{partitions}_partitions");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new("n_partitions", &param),
                &(n, partitions),
                |b, &(n, p)| {
                    b.iter(|| {
                        let mut q = MultiQueue::new();
                        for i in 0..n {
                            q.push(black_box(i as u64), false, (i % p) as i64, i % 128);
                        }
                        black_box(q.len());
                    });
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5 – Multi-partition queue: pop (partition-suppression ordering)
// ---------------------------------------------------------------------------

fn bench_multi_queue_pop(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/multi/pop");

    for &n in &[1_000usize, 10_000] {
        for &partitions in &[2usize, 8, 32] {
            let param = format!("{n}_entries_{partitions}_partitions");
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(
                BenchmarkId::new("n_partitions", &param),
                &(n, partitions),
                |b, &(n, p)| {
                    b.iter_batched(
                        || make_multi_queue(n, p),
                        |mut q| {
                            let mut total = 0u64;
                            while let Some(id) = q.pop() {
                                total += black_box(id);
                            }
                            q.remove_empty_partitions();
                            black_box(total);
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 6 – Multi-partition queue: mixed push+pop (online steady state)
// ---------------------------------------------------------------------------

fn bench_multi_queue_steady_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/multi/steady_state");

    // Simulate the loop: push a batch, pop a batch, repeat.
    for &batch in &[16usize, 64, 256] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(BenchmarkId::new("batch", batch), &batch, |b, &batch| {
            let mut q = MultiQueue::new();
            let mut id_counter = 0u64;
            b.iter(|| {
                // Push a batch with 2 version partitions.
                for i in 0..batch {
                    q.push(id_counter, i % 16 == 0, (i % 2) as i64, i % 128);
                    id_counter = id_counter.wrapping_add(1);
                }
                // Pop the same number.
                let mut total = 0u64;
                for _ in 0..batch {
                    if let Some(id) = q.pop() {
                        total += black_box(id);
                    }
                }
                q.remove_empty_partitions();
                black_box(total);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 7 – Validation priority: validate entries always pop first
// ---------------------------------------------------------------------------

fn bench_validation_priority(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_loop_queue/single/validation_priority");

    // Mix of N normal requests + 1 validation; measure time for the
    // validation entry to reach the head.
    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements((n + 1) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut q = SingleQueue::new();
                    // Push normal entries first.
                    for i in 0..n {
                        q.push(i as u64 + 1, false, 0, i % 128);
                    }
                    // Push validation entry last.
                    q.push(0, true, 0, 9999);
                    q
                },
                |mut q| {
                    // The first pop should yield the validation entry (id == 0).
                    let first = q.pop();
                    black_box(first);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 8 – mpsc enqueue throughput (the RoutingLoopRuntime::enqueue path)
// ---------------------------------------------------------------------------

fn bench_mpsc_enqueue_throughput(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("routing_loop/mpsc_enqueue");

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                rt.block_on(async {
                    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u64>();
                    for i in 0..n {
                        tx.send(black_box(i as u64)).unwrap();
                    }
                    drop(tx);
                    let mut total = 0u64;
                    while let Some(v) = rx.recv().await {
                        total += v;
                    }
                    black_box(total);
                });
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 9 – Batch drain throughput (drain_receiver pattern)
//
// The routing loop drains up to `receive_batch_size` entries from the channel
// on each tick using `try_recv`.  This benchmarks that drain pattern.
// ---------------------------------------------------------------------------

fn bench_channel_drain(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    let mut group = c.benchmark_group("routing_loop/channel_drain");

    for &n in &[64usize, 256, 1024] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u64>();
                    for i in 0..n {
                        tx.send(i as u64).unwrap();
                    }
                    (tx, rx)
                },
                |(_tx, mut rx)| {
                    rt.block_on(async {
                        let mut entries = Vec::with_capacity(n);
                        loop {
                            match rx.try_recv() {
                                Ok(v) => entries.push(v),
                                Err(_) => break,
                            }
                        }
                        black_box(entries);
                    });
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    queue_benches,
    bench_single_queue_push,
    bench_single_queue_pop,
    bench_single_queue_push_pop,
    bench_validation_priority,
    bench_multi_queue_push,
    bench_multi_queue_pop,
    bench_multi_queue_steady_state,
);

criterion_group!(
    channel_benches,
    bench_mpsc_enqueue_throughput,
    bench_channel_drain,
);

criterion_main!(queue_benches, channel_benches);
