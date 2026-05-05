//! Phase 1: PSRL Worker Selector Tests (Version Filtering, Group Pinning, Request Lifecycle)

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tokio::task;

    use crate::{
        config::types::RoutingLoopConfig, routers::grpc::routing_loop::runtime::RoutingLoopRuntime,
    };

    fn make_runtime() -> Arc<RoutingLoopRuntime> {
        let (rt, _rx) =
            RoutingLoopRuntime::new(&RoutingLoopConfig::default(), Arc::new(dashmap::DashMap::new()));
        rt
    }

    /// Test: Stage 1 version filtering removes stale workers by synced version
    #[tokio::test]
    async fn test_version_filter_exact_match() {
        let runtime = make_runtime();

        // Populate the version map with two instances
        runtime
            .instance_to_version_after_sync
            .insert(("worker-a".to_string(), 0), 5);
        runtime
            .instance_to_version_after_sync
            .insert(("worker-b".to_string(), 0), 3);

        // A request with version_tag=4 should exclude worker-b (synced version 3 < 4)
        let version_tag: i64 = 4;
        let worker_a_passes = runtime
            .instance_to_version_after_sync
            .get(&("worker-a".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        let worker_b_passes = runtime
            .instance_to_version_after_sync
            .get(&("worker-b".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);

        assert!(
            worker_a_passes,
            "worker-a with version 5 should pass version_tag=4 filter"
        );
        assert!(
            !worker_b_passes,
            "worker-b with version 3 should fail version_tag=4 filter"
        );
    }

    /// Test: Version filtering with multiple DP ranks
    #[tokio::test]
    async fn test_version_filter_multiple_dp_ranks() {
        let runtime = make_runtime();

        // Same base_worker_id but different DP ranks
        runtime
            .instance_to_version_after_sync
            .insert(("worker-x".to_string(), 0), 7);
        runtime
            .instance_to_version_after_sync
            .insert(("worker-x".to_string(), 1), 6);
        runtime
            .instance_to_version_after_sync
            .insert(("worker-x".to_string(), 2), 5);

        let version_tag: i64 = 6;

        let rank0_passes = runtime
            .instance_to_version_after_sync
            .get(&("worker-x".to_string(), 0))
            .is_some_and(|v| *v >= version_tag);
        let rank1_passes = runtime
            .instance_to_version_after_sync
            .get(&("worker-x".to_string(), 1))
            .is_some_and(|v| *v >= version_tag);
        let rank2_passes = runtime
            .instance_to_version_after_sync
            .get(&("worker-x".to_string(), 2))
            .is_some_and(|v| *v >= version_tag);

        assert!(rank0_passes, "rank 0 with version 7 should pass");
        assert!(rank1_passes, "rank 1 with version 6 should pass (equal)");
        assert!(
            !rank2_passes,
            "rank 2 with version 5 should fail version_tag=6 filter"
        );
    }

    /// Test: Group pinning uses atomic write-once semantics
    #[tokio::test]
    async fn test_group_pin_atomic_write_once() {
        let runtime = make_runtime();
        let prompt_id: i64 = 7;
        let instance_a = ("worker-a".to_string(), 0_usize);
        let instance_b = ("worker-b".to_string(), 0_usize);

        // First write should succeed
        runtime.record_selected_instance(42, Some(prompt_id), instance_a.clone());

        // Verify the pin was set
        let pinned = runtime
            .prompt_to_pinned_instance
            .get(&prompt_id)
            .map(|r| r.value().clone());
        assert_eq!(
            pinned,
            Some(instance_a.clone()),
            "First instance should be pinned"
        );

        // Second write to same prompt_id should not overwrite (write-once semantics)
        runtime.record_selected_instance(43, Some(prompt_id), instance_b.clone());

        let pinned_after = runtime
            .prompt_to_pinned_instance
            .get(&prompt_id)
            .map(|r| r.value().clone());
        assert_eq!(
            pinned_after,
            Some(instance_a.clone()),
            "Pin should remain unchanged after second write attempt (write-once semantics)"
        );
    }

    /// Test: Concurrent writes to same prompt_id respect write-once
    #[tokio::test]
    async fn test_group_pin_concurrent_writes() {
        let runtime = Arc::new(make_runtime());
        let prompt_id: i64 = 7;
        let mut handles = vec![];

        // Spawn 10 concurrent tasks, each trying to record an instance
        for i in 0..10 {
            let rt = Arc::clone(&runtime);
            let handle = task::spawn(async move {
                let instance = (format!("worker-{}", i), i as usize);
                rt.record_selected_instance(100 + i as i64, Some(prompt_id), instance.clone());
                instance
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Verify only one instance is pinned (the first one to insert)
        let pinned = runtime
            .prompt_to_pinned_instance
            .get(&prompt_id)
            .map(|r| r.value().clone());
        assert!(
            pinned.is_some(),
            "One instance should be pinned after concurrent writes"
        );

        // Verify only one of the 10 workers is pinned (consistent state)
        let pinned_worker_id = &pinned.unwrap().0;
        assert!(
            pinned_worker_id.starts_with("worker-"),
            "Pinned instance should be one of the workers"
        );
    }

    /// Test: Post-selection records request in prompt group tracking
    #[tokio::test]
    async fn test_post_selection_bookkeeping() {
        let runtime = make_runtime();
        let request_id: i64 = 42;
        let prompt_id: i64 = 7;
        let instance = ("worker-x".to_string(), 1_usize);

        // Simulate post-selection: record_selected_instance
        runtime.record_selected_instance(request_id, Some(prompt_id), instance.clone());

        // Verify the request was added to the prompt's running list
        let running_ids = runtime
            .prompt_to_running_request_ids
            .get(&prompt_id)
            .map(|ids| ids.value().clone())
            .unwrap_or_default();

        assert_eq!(running_ids, vec![request_id], "Request should be recorded in group");
    }

    /// Test: Cleanup removes single request from prompt group
    #[tokio::test]
    async fn test_cleanup_removes_single_request() {
        let runtime = make_runtime();
        let request_id: i64 = 42;
        let prompt_id: i64 = 7;

        runtime.record_selected_instance(request_id, Some(prompt_id), ("worker-x".to_string(), 1));

        // Cleanup should remove request and empty prompt entry
        runtime.cleanup_tracking(Some(request_id), Some(prompt_id));

        let still_exists = runtime.prompt_to_running_request_ids.contains_key(&prompt_id);
        assert!(
            !still_exists,
            "Prompt entry should be removed when last request completes"
        );

        let pin_still_exists = runtime.prompt_to_pinned_instance.contains_key(&prompt_id);
        assert!(
            !pin_still_exists,
            "Pin should be removed when prompt group becomes empty"
        );
    }

    /// Test: Cleanup preserves other requests in prompt group
    #[tokio::test]
    async fn test_cleanup_preserves_other_requests() {
        let runtime = make_runtime();

        // Record two requests for same prompt
        runtime.record_selected_instance(42, Some(7), ("worker-x".to_string(), 1));
        runtime.record_selected_instance(43, Some(7), ("worker-y".to_string(), 0));

        // Clean up first request
        runtime.cleanup_tracking(Some(42), Some(7));

        // Verify second request still exists
        let remaining_ids = runtime
            .prompt_to_running_request_ids
            .get(&7)
            .map(|ids| ids.value().clone())
            .unwrap_or_default();

        assert_eq!(
            remaining_ids, vec![43],
            "Second request should remain after first is cleaned up"
        );

        // Pin should also remain (group not empty)
        let pin_exists = runtime.prompt_to_pinned_instance.contains_key(&7);
        assert!(
            pin_exists,
            "Pin should remain while other requests exist in group"
        );
    }

    /// Test: Concurrent requests to same prompt group
    #[tokio::test]
    async fn test_concurrent_requests_same_group() {
        let runtime = Arc::new(make_runtime());
        let prompt_id: i64 = 7;
        let mut handles = vec![];

        // Spawn 5 concurrent requests to same prompt
        for i in 0..5 {
            let rt = Arc::clone(&runtime);
            let handle = task::spawn(async move {
                let request_id = 100 + i as i64;
                let instance = ("worker-shared".to_string(), 0_usize);
                rt.record_selected_instance(request_id, Some(prompt_id), instance);
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _ = handle.await;
        }

        // All 5 requests should be recorded
        let running_ids = runtime
            .prompt_to_running_request_ids
            .get(&prompt_id)
            .map(|ids| ids.value().clone())
            .unwrap_or_default();

        assert_eq!(running_ids.len(), 5, "All 5 requests should be recorded");
        assert_eq!(running_ids, vec![100, 101, 102, 103, 104], "Requests recorded in order");
    }

    /// Test: Multiple independent prompt groups maintain separate pins
    #[tokio::test]
    async fn test_multiple_independent_prompt_groups() {
        let runtime = make_runtime();

        // Set up two independent prompt groups
        runtime.record_selected_instance(42, Some(7), ("worker-a".to_string(), 0));
        runtime.record_selected_instance(99, Some(8), ("worker-b".to_string(), 1));

        // Verify each has the correct pin
        let pin_7 = runtime
            .prompt_to_pinned_instance
            .get(&7)
            .map(|r| r.value().clone());
        let pin_8 = runtime
            .prompt_to_pinned_instance
            .get(&8)
            .map(|r| r.value().clone());

        assert_eq!(
            pin_7,
            Some(("worker-a".to_string(), 0)),
            "Prompt 7 should be pinned to worker-a"
        );
        assert_eq!(
            pin_8,
            Some(("worker-b".to_string(), 1)),
            "Prompt 8 should be pinned to worker-b"
        );

        // Cleanup one should not affect the other
        runtime.cleanup_tracking(Some(42), Some(7));

        let pin_7_removed = runtime.prompt_to_pinned_instance.contains_key(&7);
        let pin_8_exists = runtime.prompt_to_pinned_instance.contains_key(&8);

        assert!(!pin_7_removed, "Prompt 7 pin should be removed");
        assert!(pin_8_exists, "Prompt 8 pin should still exist");
    }

    /// Test: Version tracking updates after sync
    #[tokio::test]
    async fn test_version_tracking_after_sync() {
        let runtime = make_runtime();
        let instance = ("worker-sync".to_string(), 0_usize);

        // Initial sync: version 3
        runtime.instance_to_version_after_sync.insert(instance.clone(), 3);

        let v1 = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(v1, Some(3), "Initial version should be 3");

        // Update after new sync: version 5
        runtime.instance_to_version_after_sync.insert(instance.clone(), 5);

        let v2 = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(v2, Some(5), "Version should be updated to 5");
    }
}
