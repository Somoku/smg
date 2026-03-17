// PR 4 §4.2: Global priority-based request queue
//!
//! Generic priority queue infrastructure for request ordering across all
//! endpoint types. Supports single-queue and multi-queue (version-partitioned)
//! modes with configurable sort indicators.
//!
//! ## Architecture
//!
//! - [`RequestPriority`]: Trait for items that can be priority-ordered.
//! - [`PriorityRequestQueue`]: Single binary-heap queue with FIFO tie-breaking.
//! - [`MultiPriorityRequestQueue`]: Version-partitioned wrapper holding multiple
//!   sub-queues indexed by a queue selector (default: `version_tag`).

use std::collections::{BTreeMap, BinaryHeap, HashMap};

use crate::config::RequestSortIndicator;

// PR 4 §4.2: Trait for items that can provide priority ordering
/// Trait for request types that can be priority-ordered in the queue.
///
/// Implementors provide a version tag for multi-queue partitioning and
/// a priority tuple `(p0, p1, p2)` where lower values = higher scheduling priority.
pub trait RequestPriority {
    /// Version tag for multi-queue partitioning.
    ///
    /// Returns `0` by default (all requests in same partition).
    /// Return `-1` for unversioned requests (mapped to `i32::MAX`).
    #[inline]
    fn get_version_tag(&self) -> i64 {
        0
    }

    /// Priority tuple for ordering. Lower values = higher scheduling priority.
    ///
    /// The three components allow multi-level sorting (e.g., validation priority,
    /// request length, request ID).
    #[inline]
    fn get_priority(&self, _indicator: RequestSortIndicator) -> (i64, i64, i64) {
        (0, 0, 0)
    }
}

// PR 4 §4.2: Internal entry wrapping a request with its computed priority and FIFO seq
struct RequestQueueEntry<T> {
    priority: (i64, i64, i64),
    seq: u64,
    request: T,
}

impl<T> Ord for RequestQueueEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap pops the "largest" item first.
        // We invert comparisons so that lower priority tuple (higher scheduling priority)
        // and lower sequence number (FIFO within same priority) are popped first.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl<T> PartialOrd for RequestQueueEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> PartialEq for RequestQueueEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl<T> Eq for RequestQueueEntry<T> {}

// PR 4 §4.2: Single priority request queue with FIFO tie-breaking
/// Single priority queue backed by a [`BinaryHeap`] with FIFO tie-breaking.
///
/// Items are dequeued in ascending priority order (lower tuple = higher priority).
/// Within the same priority, items are dequeued in insertion order (FIFO).
pub struct PriorityRequestQueue<T: RequestPriority> {
    heap: BinaryHeap<RequestQueueEntry<T>>,
    /// Monotonic counter for FIFO tie-breaking within the same priority key.
    counter: u64,
    /// Determines which sort key to compute for each request.
    sort_indicator: RequestSortIndicator,
}

impl<T: RequestPriority> PriorityRequestQueue<T> {
    pub fn new(sort_indicator: RequestSortIndicator) -> Self {
        Self {
            heap: BinaryHeap::new(),
            counter: 0,
            sort_indicator,
        }
    }

    pub fn push(&mut self, request: T) {
        let priority = request.get_priority(self.sort_indicator);
        let seq = self.counter;
        self.counter += 1;
        let entry = RequestQueueEntry {
            priority,
            seq,
            request,
        };
        self.heap.push(entry);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|entry| entry.request)
    }

    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|entry| &entry.request)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    pub fn sort_indicator(&self) -> RequestSortIndicator {
        self.sort_indicator
    }

    pub fn iter_requests(&self) -> impl Iterator<Item = &T> {
        self.heap.iter().map(|entry| &entry.request)
    }

    pub fn filter<F>(&self, predicate: F) -> Vec<&T>
    where
        F: Fn(&T) -> bool,
    {
        self.iter_requests().filter(|req| predicate(req)).collect()
    }
}

// PR 4 §4.2: Default queue selector based on version_tag
#[inline]
fn default_queue_selector<T: RequestPriority>(request: &T) -> i32 {
    if request.get_version_tag() == -1 {
        i32::MAX
    } else {
        request.get_version_tag() as i32
    }
}

// PR 4 §4.2: Multi-queue partitioned by version tag
/// Multi-queue wrapper that partitions requests into sub-queues by version tag.
///
/// When `enable_multi_priority_queue` is `true`, requests are routed to
/// sub-queues based on the queue selector function (default: `version_tag`).
/// When `false`, all requests go to a single sub-queue (key `0`).
///
/// Sub-queues are stored in a [`BTreeMap`] so iteration is in sorted order
/// (lowest version tag first).
pub struct MultiPriorityRequestQueue<T: RequestPriority> {
    /// Sorted map of queue_key → sub-queue.
    queues: BTreeMap<i32, PriorityRequestQueue<T>>,
    /// Sort indicator passed to new sub-queues.
    sort_indicator: RequestSortIndicator,
    /// Whether to partition by queue_selector or force single queue.
    enable_multi_priority_queue: bool,
    /// Queue selector function.
    /// Default is `default_queue_selector`.
    queue_selector: fn(&T) -> i32,
}

impl<T: RequestPriority> MultiPriorityRequestQueue<T> {
    pub fn new(
        sort_indicator: RequestSortIndicator,
        enable_multi_priority_queue: bool,
        queue_selector: Option<fn(&T) -> i32>,
    ) -> Self {
        Self {
            queues: BTreeMap::new(),
            sort_indicator,
            enable_multi_priority_queue,
            queue_selector: queue_selector.unwrap_or(default_queue_selector::<T>),
        }
    }

    #[inline]
    pub fn enable_multi_priority_queue(&self) -> bool {
        self.enable_multi_priority_queue
    }

    #[inline]
    pub fn sort_indicator(&self) -> RequestSortIndicator {
        self.sort_indicator
    }

    #[inline]
    fn queue_key_for_request(&self, request: &T) -> i32 {
        if self.enable_multi_priority_queue {
            (self.queue_selector)(request)
        } else {
            0
        }
    }

    pub fn push(&mut self, request: T) {
        let key = self.queue_key_for_request(&request);
        let queue = self
            .queues
            .entry(key)
            .or_insert_with(|| PriorityRequestQueue::new(self.sort_indicator));
        queue.push(request);
    }

    pub fn push_back_to_queue(&mut self, queue_key: i32, request: T) {
        let queue = self
            .queues
            .entry(queue_key)
            .or_insert_with(|| PriorityRequestQueue::new(self.sort_indicator));
        queue.push(request);
    }

    pub fn pop(&mut self) -> Option<T> {
        for queue in self.queues.values_mut() {
            if let Some(entry) = queue.pop() {
                return Some(entry);
            }
        }
        None
    }

    pub fn pop_from_queue(&mut self, queue_key: i32) -> Option<T> {
        self.queues.get_mut(&queue_key)?.pop()
    }

    pub fn peek(&self) -> Option<&T> {
        for queue in self.queues.values() {
            if let Some(request) = queue.peek() {
                return Some(request);
            }
        }
        None
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queues.values().all(|q| q.is_empty())
    }

    pub fn total_size(&self) -> usize {
        self.queues.values().map(|q| q.size()).sum()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total_size()
    }

    pub fn queue_size(&self, queue_key: i32) -> usize {
        self.queues.get(&queue_key).map_or(0, |q| q.size())
    }

    pub fn queue_keys(&self) -> Vec<i32> {
        self.queues.keys().copied().collect()
    }

    pub fn per_version_sizes(&self) -> HashMap<i32, usize> {
        self.queues.iter().map(|(&k, q)| (k, q.size())).collect()
    }

    pub fn get_queue(&self, queue_key: i32) -> Option<&PriorityRequestQueue<T>> {
        self.queues.get(&queue_key)
    }

    pub fn remove_queue(&mut self, queue_key: i32) -> bool {
        self.queues.remove(&queue_key).is_some()
    }

    pub fn remove_empty_queues(&mut self) {
        self.queues.retain(|_, q| !q.is_empty());
    }

    pub fn iter_requests(&self) -> impl Iterator<Item = &T> {
        self.queues.iter().flat_map(|(_, q)| q.iter_requests())
    }

    pub fn filter<F>(&self, predicate: F) -> Vec<&T>
    where
        F: Fn(&T) -> bool,
    {
        self.iter_requests().filter(|req| predicate(req)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestReq {
        id: i64,
        version_tag: i64,
        priority: (i64, i64, i64),
    }

    impl RequestPriority for TestReq {
        fn get_version_tag(&self) -> i64 {
            self.version_tag
        }

        fn get_priority(&self, _indicator: RequestSortIndicator) -> (i64, i64, i64) {
            self.priority
        }
    }

    // ── PriorityRequestQueue tests ──

    #[test]
    fn test_priority_queue_ordering() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (1, 10, 10),
        });
        q.push(TestReq {
            id: 2,
            version_tag: 0,
            priority: (0, 10, 10),
        });

        let first = q.pop().expect("first element should exist");
        let second = q.pop().expect("second element should exist");

        // Lower priority tuple = higher scheduling priority (popped first)
        assert_eq!(first.priority, (0, 10, 10));
        assert_eq!(second.priority, (1, 10, 10));
    }

    #[test]
    fn test_priority_queue_fifo_tiebreak() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (0, 10, 10),
        });
        q.push(TestReq {
            id: 2,
            version_tag: 0,
            priority: (0, 10, 10),
        });

        let first = q.pop().expect("first element should exist");
        let second = q.pop().expect("second element should exist");

        // FIFO: first pushed item should be popped first
        assert_eq!(first.id, 1);
        assert_eq!(second.id, 2);
    }

    #[test]
    fn test_priority_queue_push_pop() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::ShortLength);
        q.push(TestReq {
            id: 42,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        let popped = q.pop().expect("should pop");
        assert_eq!(popped.id, 42);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_priority_queue_peek() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        assert!(q.peek().is_none());
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (5, 0, 0),
        });
        q.push(TestReq {
            id: 2,
            version_tag: 0,
            priority: (1, 0, 0),
        });
        // peek should return highest priority (lowest tuple) without removing
        assert_eq!(q.peek().expect("should peek").id, 2);
        assert_eq!(q.size(), 2); // not removed
    }

    #[test]
    fn test_priority_queue_is_empty() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        assert!(q.is_empty());
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        assert!(!q.is_empty());
        q.pop();
        assert!(q.is_empty());
    }

    #[test]
    fn test_priority_queue_len() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        assert_eq!(q.size(), 0);
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        q.push(TestReq {
            id: 2,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        assert_eq!(q.size(), 2);
        q.pop();
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn test_priority_queue_filter() {
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        q.push(TestReq {
            id: 1,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        q.push(TestReq {
            id: 2,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        q.push(TestReq {
            id: 3,
            version_tag: 0,
            priority: (0, 0, 0),
        });
        let matched: Vec<&TestReq> = q.filter(|r| r.id > 1);
        assert_eq!(matched.len(), 2);
    }

    // ── MultiPriorityRequestQueue tests ──

    #[test]
    fn test_multi_queue_version_partitioning() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 3,
            version_tag: 1,
            priority: (0, 0, 0),
        });

        assert_eq!(mq.queue_keys(), vec![1, 2]);
        assert_eq!(mq.queue_size(1), 2);
        assert_eq!(mq.queue_size(2), 1);
    }

    #[test]
    fn test_multi_queue_push_pop_from_queue() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });

        let popped = mq.pop_from_queue(2).expect("should pop from queue 2");
        assert_eq!(popped.id, 2);
        assert!(mq.pop_from_queue(2).is_none());
        assert_eq!(mq.pop_from_queue(1).expect("queue 1").id, 1);
    }

    #[test]
    fn test_multi_queue_push_back_to_queue() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        // Push to queue 1 directly via push_back_to_queue
        mq.push_back_to_queue(
            1,
            TestReq {
                id: 99,
                version_tag: 999, // version_tag doesn't matter for push_back_to_queue
                priority: (0, 0, 0),
            },
        );
        assert_eq!(mq.queue_size(1), 1);
        let popped = mq.pop_from_queue(1).expect("should pop");
        assert_eq!(popped.id, 99);
    }

    #[test]
    fn test_multi_queue_queue_keys() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 3,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 3,
            version_tag: 2,
            priority: (0, 0, 0),
        });

        // BTreeMap keys are sorted
        assert_eq!(mq.queue_keys(), vec![1, 2, 3]);
    }

    #[test]
    fn test_multi_queue_per_version_sizes() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 3,
            version_tag: 2,
            priority: (0, 0, 0),
        });

        let sizes = mq.per_version_sizes();
        assert_eq!(sizes.get(&1), Some(&2));
        assert_eq!(sizes.get(&2), Some(&1));
    }

    #[test]
    fn test_multi_queue_remove_empty_queues() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });

        // Drain queue 1
        mq.pop_from_queue(1);
        assert_eq!(mq.queue_keys().len(), 2);

        mq.remove_empty_queues();
        assert_eq!(mq.queue_keys(), vec![2]);
    }

    #[test]
    fn test_multi_queue_filter() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 3,
            version_tag: 1,
            priority: (0, 0, 0),
        });

        let matched: Vec<&TestReq> = mq.filter(|r| r.version_tag == 1);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_multi_queue_len_total() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        assert_eq!(mq.len(), 0);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 3,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        assert_eq!(mq.len(), 3);
        assert_eq!(mq.total_size(), 3);
    }

    #[test]
    fn test_multi_queue_enable_multi_priority_queue() {
        // When disabled, all go to queue key 0
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, false, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 2,
            priority: (0, 0, 0),
        });

        // All in single queue with key 0
        assert_eq!(mq.queue_keys(), vec![0]);
        assert_eq!(mq.queue_size(0), 2);
    }

    #[test]
    fn test_multi_queue_pop_order_across_queues() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        // BTreeMap iterates in key order, so queue 1 is drained first
        mq.push(TestReq {
            id: 1,
            version_tag: 2,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 1,
            priority: (0, 0, 0),
        });

        // Queue 1 first (lower key), then queue 2
        let first = mq.pop().expect("first");
        assert_eq!(first.id, 2); // from queue_key=1
        let second = mq.pop().expect("second");
        assert_eq!(second.id, 1); // from queue_key=2
    }

    #[test]
    fn test_multi_queue_unversioned_requests() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        // version_tag=-1 → queue key = i32::MAX
        mq.push(TestReq {
            id: 1,
            version_tag: -1,
            priority: (0, 0, 0),
        });
        mq.push(TestReq {
            id: 2,
            version_tag: 1,
            priority: (0, 0, 0),
        });

        assert!(mq.queue_keys().contains(&i32::MAX));
        assert!(mq.queue_keys().contains(&1));
        // queue 1 pops first (lower key in BTreeMap)
        let first = mq.pop().expect("first");
        assert_eq!(first.id, 2);
    }

    #[test]
    fn test_multi_queue_is_empty() {
        let mut mq: MultiPriorityRequestQueue<TestReq> =
            MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        assert!(mq.is_empty());
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        assert!(!mq.is_empty());
        mq.pop();
        assert!(mq.is_empty());
    }

    #[test]
    fn test_multi_queue_get_queue() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        assert!(mq.get_queue(1).is_none());
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        assert!(mq.get_queue(1).is_some());
        assert_eq!(mq.get_queue(1).expect("exists").size(), 1);
    }

    #[test]
    fn test_multi_queue_remove_queue() {
        let mut mq = MultiPriorityRequestQueue::new(RequestSortIndicator::SmallId, true, None);
        mq.push(TestReq {
            id: 1,
            version_tag: 1,
            priority: (0, 0, 0),
        });
        assert!(mq.remove_queue(1));
        assert!(!mq.remove_queue(1)); // already removed
        assert!(mq.is_empty());
    }

    // ── PR 4 §4.0: Sort-indicator-aware and validate-priority tests ──

    // PR 4 §4.0: Test request type that uses the sort indicator to compute priority
    /// Richer test request that computes priority based on the sort indicator,
    /// simulating real routing queue behavior.
    #[derive(Debug, Clone)]
    struct SortAwareReq {
        id: i64,
        text: String,
        is_validate: bool,
        version_tag: i64,
    }

    impl RequestPriority for SortAwareReq {
        fn get_version_tag(&self) -> i64 {
            self.version_tag
        }

        fn get_priority(&self, indicator: RequestSortIndicator) -> (i64, i64, i64) {
            // p0: validation requests get priority 0 (higher), normal requests get 1
            let p0 = if self.is_validate { 0 } else { 1 };
            // p1: length-based or id-based depending on indicator
            let p1 = match indicator {
                RequestSortIndicator::ShortLength => self.text.len() as i64,
                RequestSortIndicator::LongLength => -(self.text.len() as i64),
                RequestSortIndicator::SmallId => self.id,
            };
            // p2: tie-break by id
            let p2 = self.id;
            (p0, p1, p2)
        }
    }

    #[test]
    fn test_request_sort_short_length() {
        // PR 4 §4.0: ShortLength indicator → shorter text first
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::ShortLength);
        q.push(SortAwareReq {
            id: 1,
            text: "long text here".to_string(),
            is_validate: false,
            version_tag: 0,
        });
        q.push(SortAwareReq {
            id: 2,
            text: "hi".to_string(),
            is_validate: false,
            version_tag: 0,
        });

        let first = q.pop().expect("first");
        let second = q.pop().expect("second");

        // Shorter text should be popped first (lower p1 = len)
        assert_eq!(first.id, 2, "shorter text should come first");
        assert_eq!(second.id, 1);
    }

    #[test]
    fn test_request_sort_long_length() {
        // PR 4 §4.0: LongLength indicator → longer text first
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::LongLength);
        q.push(SortAwareReq {
            id: 1,
            text: "hi".to_string(),
            is_validate: false,
            version_tag: 0,
        });
        q.push(SortAwareReq {
            id: 2,
            text: "long text here".to_string(),
            is_validate: false,
            version_tag: 0,
        });

        let first = q.pop().expect("first");
        let second = q.pop().expect("second");

        // Longer text should be popped first (lower p1 = -len, so longer → more negative → lower)
        assert_eq!(first.id, 2, "longer text should come first");
        assert_eq!(second.id, 1);
    }

    #[test]
    fn test_request_sort_small_id() {
        // PR 4 §4.0: SmallId indicator → lower request_id first
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::SmallId);
        q.push(SortAwareReq {
            id: 100,
            text: "same".to_string(),
            is_validate: false,
            version_tag: 0,
        });
        q.push(SortAwareReq {
            id: 5,
            text: "same".to_string(),
            is_validate: false,
            version_tag: 0,
        });
        q.push(SortAwareReq {
            id: 50,
            text: "same".to_string(),
            is_validate: false,
            version_tag: 0,
        });

        let first = q.pop().expect("first");
        let second = q.pop().expect("second");
        let third = q.pop().expect("third");

        // Lower id should be popped first
        assert_eq!(first.id, 5, "smallest id should come first");
        assert_eq!(second.id, 50);
        assert_eq!(third.id, 100);
    }

    #[test]
    fn test_validate_priority() {
        // PR 4 §4.0: is_validate=true entries have lower priority number (higher priority)
        let mut q = PriorityRequestQueue::new(RequestSortIndicator::ShortLength);
        // Normal request (short text)
        q.push(SortAwareReq {
            id: 1,
            text: "a".to_string(),
            is_validate: false,
            version_tag: 0,
        });
        // Validate request (long text — but should still come first due to p0=0)
        q.push(SortAwareReq {
            id: 2,
            text: "very long validation request text".to_string(),
            is_validate: true,
            version_tag: 0,
        });
        // Another normal request (even shorter)
        q.push(SortAwareReq {
            id: 3,
            text: String::new(),
            is_validate: false,
            version_tag: 0,
        });

        let first = q.pop().expect("first");
        // Validate request should come first regardless of text length,
        // because p0=0 < p0=1 for normal requests
        assert!(
            first.is_validate,
            "validate request should be popped first (p0=0 beats p0=1)"
        );
        assert_eq!(first.id, 2);

        // Remaining are normal requests, ordered by ShortLength
        let second = q.pop().expect("second");
        let third = q.pop().expect("third");
        assert_eq!(second.id, 3, "empty string is shortest");
        assert_eq!(third.id, 1);
    }
}
