//! Phase 8: Partial Rollout Sync-Flow Integration Tests (P3-14)
//!
//! Tests the pause→filter→sync→resume coordinator protocol exercised at the
//! `RoutingLoopRuntime` API level (without the HTTP layer).
//!
//! The protocol is:
//!   1. Python coordinator calls `POST /routing_loop/pause?wait=true`
//!      → `runtime.pause()` sets `paused=true`; then polls `is_routing()` until
//!      the current dispatch batch finishes.
//!   2. Python coordinator calls `GET /routing_loop/filter?version_tag=N`
//!      → `runtime.filter_queue_by_version_tag(N)`
//!   3. Weight sync happens (Python side); coordinator updates version tags:
//!      `instance_to_version_after_sync.insert((worker, rank), new_version)`
//!   4. Python coordinator calls `POST /routing_loop/resume`
//!      → `runtime.resume()` clears `paused` flag.
//!
//! # Semantics clarification
//!
//! * `is_paused()` — whether `POST /routing_loop/pause` has been called
//!   (prevents the routing loop from dispatching new batches).
//! * `is_routing()` — whether a dispatch batch is *currently in flight*
//!   (set to `true` by `start_task()`, back to `false` when the batch finishes).
//!   Used by `pause?wait=true` to wait for in-flight work to drain.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        config::RoutingLoopConfig, routers::grpc::routing_loop::runtime::RoutingLoopRuntime,
        worker::WorkerRegistry,
    };

    fn make_runtime() -> Arc<RoutingLoopRuntime> {
        let (rt, _rx) = RoutingLoopRuntime::new(
            &RoutingLoopConfig::default(),
            Arc::new(dashmap::DashMap::new()),
            Arc::new(WorkerRegistry::new()),
        );
        rt
    }

    // ─── Pause / resume state transitions ─────────────────────────────────────

    /// `pause()` → `is_paused()` becomes true; `resume()` → false again.
    #[tokio::test]
    async fn pause_stops_routing_resume_restarts() {
        let rt = make_runtime();

        // Initially not paused
        assert!(!rt.is_paused(), "should start not paused");
        // `is_routing()` is false until a dispatch batch starts — not relevant here.

        rt.pause();
        assert!(rt.is_paused(), "should be paused after pause()");
        // `is_routing()` stays false since no task is in flight.
        assert!(!rt.is_routing(), "no dispatch batch running yet");

        rt.resume();
        assert!(!rt.is_paused(), "should be not paused after resume()");
    }

    /// Multiple consecutive `pause()` calls are idempotent.
    #[tokio::test]
    async fn pause_is_idempotent() {
        let rt = make_runtime();
        rt.pause();
        rt.pause();
        rt.pause();
        assert!(rt.is_paused());
    }

    /// Multiple consecutive `resume()` calls are idempotent.
    #[tokio::test]
    async fn resume_is_idempotent() {
        let rt = make_runtime();
        rt.pause();
        assert!(rt.is_paused());

        rt.resume();
        rt.resume();
        rt.resume();
        assert!(!rt.is_paused(), "should not be paused after resume");
    }

    /// `is_routing()` starts false when no dispatch batch is in flight.
    #[tokio::test]
    async fn is_routing_starts_false() {
        let rt = make_runtime();
        assert!(!rt.is_routing(), "no dispatch batch running initially");
    }

    /// `pause?wait=true` returns immediately when `is_routing()` is already false.
    ///
    /// This simulates the happy-path where the coordinator pauses between batches.
    #[tokio::test]
    async fn pause_wait_returns_immediately_when_not_routing() {
        let rt = make_runtime();
        rt.pause();
        // is_routing() is false → the wait loop exits immediately
        assert!(!rt.is_routing());
        assert!(rt.is_paused());
    }

    // ─── Version-tag filter (step 2 of sync protocol) ─────────────────────────

    /// Empty queue returns an empty filter result for any version_tag.
    #[tokio::test]
    async fn filter_on_empty_queue_returns_empty() {
        let rt = make_runtime();
        let result = rt.filter_queue_by_version_tag(100).await;
        assert!(
            result.is_empty(),
            "empty queue should produce empty filter result"
        );
    }

    /// Filter result is JSON-serialisable (required for the HTTP endpoint).
    #[tokio::test]
    async fn filter_returns_serialisable_metadata() {
        let rt = make_runtime();
        let result = rt.filter_queue_by_version_tag(0).await;
        let json = serde_json::to_string(&result).expect("should serialise to JSON");
        assert_eq!(json, "[]");
    }

    // ─── Version-map updates (step 3 of sync protocol) ───────────────────────

    /// After weight sync, coordinator updates version tags in `instance_to_version_after_sync`.
    #[tokio::test]
    async fn version_map_updated_after_sync() {
        let rt = make_runtime();

        // Before sync: no entries
        assert!(rt.instance_to_version_after_sync.is_empty());

        // Simulate weight sync: coordinator records new version for each worker
        rt.instance_to_version_after_sync
            .insert(("worker-0".to_string(), 0), 42);
        rt.instance_to_version_after_sync
            .insert(("worker-0".to_string(), 1), 42);

        assert_eq!(rt.instance_to_version_after_sync.len(), 2);
        assert_eq!(
            *rt.instance_to_version_after_sync
                .get(&("worker-0".to_string(), 0))
                .unwrap()
                .value(),
            42
        );
    }

    /// Version tags can be updated multiple times (each sync increments the version).
    #[tokio::test]
    async fn version_map_accumulates_syncs() {
        let rt = make_runtime();
        let key = ("worker-0".to_string(), 0);

        for version in [10i64, 20, 30] {
            rt.instance_to_version_after_sync
                .insert(key.clone(), version);
        }

        let final_version = *rt.instance_to_version_after_sync.get(&key).unwrap().value();
        assert_eq!(final_version, 30);
    }

    // ─── Full sync-flow protocol ───────────────────────────────────────────────

    /// Simulate the complete pause→filter→sync→resume flow end-to-end.
    #[tokio::test]
    async fn full_sync_protocol_pause_filter_sync_resume() {
        let rt = make_runtime();

        // Step 1: Coordinator pauses routing loop
        assert!(!rt.is_paused(), "should start not paused");
        rt.pause();
        assert!(rt.is_paused(), "paused after step 1");

        // Step 2: Filter queue for old-version requests (simulated as empty)
        let old_version_requests = rt.filter_queue_by_version_tag(5).await;
        // In production, coordinator would wait here until this is empty.
        assert!(old_version_requests.is_empty());

        // Step 3: Weight sync happens; coordinator records new versions
        let workers = [
            ("gpu-0".to_string(), 0usize),
            ("gpu-0".to_string(), 1usize),
            ("gpu-1".to_string(), 0usize),
        ];
        for (worker, rank) in &workers {
            rt.instance_to_version_after_sync
                .insert((worker.clone(), *rank), 6);
        }

        // Verify all workers are on version 6
        for (worker, rank) in &workers {
            let v = *rt
                .instance_to_version_after_sync
                .get(&(worker.clone(), *rank))
                .unwrap()
                .value();
            assert_eq!(v, 6, "worker ({worker}, {rank}) should be on version 6");
        }

        // Step 4: Coordinator resumes routing loop
        rt.resume();
        assert!(!rt.is_paused(), "resumed after step 4");

        // Post-resume: filter should still work correctly
        let new_requests = rt.filter_queue_by_version_tag(6).await;
        assert!(new_requests.is_empty(), "queue should still be empty");
    }

    /// The routing loop can perform multiple sync cycles in sequence.
    #[tokio::test]
    async fn multiple_consecutive_sync_cycles() {
        let rt = make_runtime();

        for cycle in 1..=3u64 {
            // Pause
            rt.pause();
            assert!(rt.is_paused(), "paused in cycle {cycle}");

            // Sync version
            rt.instance_to_version_after_sync
                .insert(("worker".to_string(), 0), cycle as i64);

            // Resume
            rt.resume();
            assert!(!rt.is_paused(), "resumed in cycle {cycle}");
        }

        // Final version should be from the last sync
        let final_version = *rt
            .instance_to_version_after_sync
            .get(&("worker".to_string(), 0))
            .unwrap()
            .value();
        assert_eq!(final_version, 3);
    }

    // ─── Concurrent pause / resume safety ────────────────────────────────────

    /// Concurrent pause/resume calls are race-free (no panic, no deadlock).
    #[tokio::test]
    async fn concurrent_pause_resume_is_safe() {
        let rt = make_runtime();
        let barrier = Arc::new(tokio::sync::Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let rt = rt.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    if i % 2 == 0 {
                        rt.pause();
                    } else {
                        rt.resume();
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("task should not panic");
        }

        // Runtime should be in a consistent state — either paused or not
        // The important guarantee is no panic/deadlock.
        let _state = rt.is_paused(); // should not panic
    }

    // ─── Workers stats preconditions ─────────────────────────────────────────

    /// `workers/stats` returns empty when no version map entries.
    #[tokio::test]
    async fn workers_stats_empty_when_no_version_entries() {
        let rt = make_runtime();
        // instance_to_version_after_sync is empty by default
        assert!(rt.instance_to_version_after_sync.is_empty());
        // No running prompts either
        assert!(rt.prompt_to_pinned_instance.is_empty());
    }

    /// `workers/stats` running_requests count is driven by `prompt_to_pinned_instance`.
    #[tokio::test]
    async fn workers_stats_running_requests_counts_pinned_prompts() {
        let rt = make_runtime();
        let worker_key = ("gpu-0".to_string(), 0usize);

        // Register version
        rt.instance_to_version_after_sync
            .insert(worker_key.clone(), 10);

        // Pin 3 prompts to this worker
        for prompt_id in 0i64..3 {
            rt.record_selected_instance(
                prompt_id * 100, // request_id
                Some(prompt_id),
                worker_key.clone(),
            );
        }

        // Count pinned prompts for this worker (mimics controller logic)
        let running_requests = rt
            .prompt_to_pinned_instance
            .iter()
            .filter(|e| *e.value() == worker_key)
            .count();

        assert_eq!(running_requests, 3, "should have 3 running prompts");
    }

    /// Running requests count decrements correctly when prompts finish.
    #[tokio::test]
    async fn workers_stats_running_requests_decrements_on_cleanup() {
        let rt = make_runtime();
        let worker_key = ("gpu-0".to_string(), 0usize);
        rt.instance_to_version_after_sync
            .insert(worker_key.clone(), 10);

        // Add 3 prompts
        for prompt_id in 0i64..3 {
            rt.record_selected_instance(prompt_id * 100, Some(prompt_id), worker_key.clone());
        }

        // Cleanup prompt 1
        rt.cleanup_tracking(Some(100), Some(1));

        let running_requests = rt
            .prompt_to_pinned_instance
            .iter()
            .filter(|e| *e.value() == worker_key)
            .count();

        assert_eq!(
            running_requests, 2,
            "should have 2 running prompts after cleanup"
        );
    }

    // ─── Status object consistency ────────────────────────────────────────────

    /// Status object reflects paused state correctly.
    #[tokio::test]
    async fn status_reflects_paused_state() {
        let rt = make_runtime();

        let status = rt.status().await;
        assert!(!status.paused, "status should reflect not paused");
        assert!(!status.routing, "status should reflect not routing");
        assert!(status.enabled, "status should reflect enabled");

        rt.pause();
        let paused_status = rt.status().await;
        assert!(paused_status.paused, "status should reflect paused");

        rt.resume();
        let resumed_status = rt.status().await;
        assert!(
            !resumed_status.paused,
            "status should reflect not paused after resume"
        );
    }
}
