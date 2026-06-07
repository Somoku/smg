//! Multi-tier positional indexer for cache-aware routing.
//!
//! Backends with a multi-level KV cache (e.g. vLLM GPU prefix cache + LMCache
//! CPU offload backend) publish store/remove events tagged with a tier marker
//! (`KvBlock.cache_level`, derived from vLLM's `BlockStored.medium`). The two
//! tiers have **independent block sizes** (GPU block size ≠ LMCache chunk size)
//! and independent residency, so each tier needs its own content-hash space and
//! its own [`PositionalIndexer`].
//!
//! [`TieredIndexer`] wraps one [`PositionalIndexer`] per [`Tier`] plus a learned
//! per-tier block size. This keeps the highly-optimized single-tier hot path in
//! [`event_tree`](crate::event_tree) completely untouched: tier routing happens
//! at the (cold) event-apply path and the weighted combination happens in the
//! router policy. Worker interning is per-tier (each tier assigns its own
//! internal `u32` worker id); callers always go tier → `worker_id(url)` →
//! score, so the ids never need to agree across tiers.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::event_tree::PositionalIndexer;

/// KV cache tier. The discriminant matches the `cache_level` carried in the
/// `KvBlock` proto so conversion is a plain cast.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Tier {
    /// On-GPU prefix cache (vLLM `medium == "GPU"`). Highest routing value —
    /// a hit means zero reload cost.
    GPU = 0,
    /// Off-GPU backend (LMCache CPU/disk offload). A hit still requires loading
    /// the prefix back onto the GPU, so it scores lower than [`Tier::GPU`].
    Lmcache = 1,
}

impl Tier {
    /// Number of tiers tracked.
    pub const COUNT: usize = 2;

    /// All tiers, in discriminant order.
    pub const ALL: [Tier; Self::COUNT] = [Tier::GPU, Tier::Lmcache];

    /// Map a proto `cache_level` to a tier. `Some(1)` → LMCache; everything
    /// else (including `None`, which legacy single-tier backends send) → GPU.
    #[inline]
    pub fn from_cache_level(cache_level: Option<i32>) -> Self {
        match cache_level {
            Some(1) => Tier::Lmcache,
            _ => Tier::GPU,
        }
    }

    /// Tier index for array access.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// One [`PositionalIndexer`] per [`Tier`], with a learned per-tier block size.
///
/// Cheap to construct (empty maps); stays dormant until events arrive.
pub struct TieredIndexer {
    tiers: [PositionalIndexer; Tier::COUNT],
    /// Per-tier block size learned from events (0 = not yet learned). Indexed by
    /// [`Tier::index`]. GPU and LMCache differ, so they are tracked separately.
    block_sizes: [AtomicUsize; Tier::COUNT],
}

impl TieredIndexer {
    /// Create a new tiered indexer; every tier uses the same `jump_size`.
    pub fn new(jump_size: usize) -> Self {
        Self {
            tiers: std::array::from_fn(|_| PositionalIndexer::new(jump_size)),
            block_sizes: std::array::from_fn(|_| AtomicUsize::new(0)),
        }
    }

    /// Borrow the indexer for a tier.
    #[inline]
    pub fn tier(&self, tier: Tier) -> &PositionalIndexer {
        &self.tiers[tier.index()]
    }

    /// Intern a worker URL in a tier and return its internal id.
    #[inline]
    pub fn intern_worker(&self, tier: Tier, worker: &str) -> u32 {
        self.tiers[tier.index()].intern_worker(worker)
    }

    /// Internal worker id for `worker` within `tier`, if interned.
    #[inline]
    pub fn worker_id(&self, tier: Tier, worker: &str) -> Option<u32> {
        self.tiers[tier.index()].worker_id(worker)
    }

    /// Total blocks across all tiers — used to decide whether the indexer holds
    /// any usable data before taking the event-driven routing path.
    pub fn current_size(&self) -> usize {
        self.tiers.iter().map(PositionalIndexer::current_size).sum()
    }

    /// Learned block size for a tier, or `None` if not yet learned.
    #[inline]
    pub fn block_size(&self, tier: Tier) -> Option<usize> {
        match self.block_sizes[tier.index()].load(Ordering::Relaxed) {
            0 => None,
            bs => Some(bs),
        }
    }

    /// Record the block size for a tier if not already set (first-write-wins,
    /// matching the "learn once from the first event" semantics).
    pub fn set_block_size(&self, tier: Tier, block_size: usize) {
        if block_size > 0 {
            let _ = self.block_sizes[tier.index()].compare_exchange(
                0,
                block_size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_level_maps_to_tier() {
        assert_eq!(Tier::from_cache_level(None), Tier::GPU);
        assert_eq!(Tier::from_cache_level(Some(0)), Tier::GPU);
        assert_eq!(Tier::from_cache_level(Some(1)), Tier::Lmcache);
        // Unknown levels conservatively fall back to GPU.
        assert_eq!(Tier::from_cache_level(Some(7)), Tier::GPU);
    }

    #[test]
    fn tiers_intern_independently() {
        let idx = TieredIndexer::new(4);
        let g = idx.intern_worker(Tier::GPU, "http://w1:8000");
        let l = idx.intern_worker(Tier::Lmcache, "http://w1:8000");
        // Both tiers know the worker; ids are per-tier (here both start at 0).
        assert_eq!(idx.worker_id(Tier::GPU, "http://w1:8000"), Some(g));
        assert_eq!(idx.worker_id(Tier::Lmcache, "http://w1:8000"), Some(l));
        assert_eq!(idx.worker_id(Tier::GPU, "http://absent"), None);
    }

    #[test]
    fn block_size_first_write_wins() {
        let idx = TieredIndexer::new(4);
        assert_eq!(idx.block_size(Tier::GPU), None);
        idx.set_block_size(Tier::GPU, 16);
        assert_eq!(idx.block_size(Tier::GPU), Some(16));
        // Subsequent writes are ignored (learn-once).
        idx.set_block_size(Tier::GPU, 32);
        assert_eq!(idx.block_size(Tier::GPU), Some(16));
        // Tiers are independent.
        assert_eq!(idx.block_size(Tier::Lmcache), None);
        idx.set_block_size(Tier::Lmcache, 256);
        assert_eq!(idx.block_size(Tier::Lmcache), Some(256));
    }

    #[test]
    fn current_size_sums_tiers() {
        let idx = TieredIndexer::new(4);
        assert_eq!(idx.current_size(), 0);
    }
}
