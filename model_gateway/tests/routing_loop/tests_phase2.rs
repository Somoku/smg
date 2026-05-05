//! Phase 2: Version Tracking Lifecycle Tests (Version updates, sync events, lifecycle)

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        config::types::RoutingLoopConfig, routers::grpc::routing_loop::runtime::RoutingLoopRuntime,
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

    /// Test: Version entry creation on first sync
    #[tokio::test]
    async fn test_version_entry_creation_on_first_sync() {
        let runtime = make_runtime();
        let instance = ("worker-new".to_string(), 0_usize);

        // No entry initially
        assert!(!runtime
            .instance_to_version_after_sync
            .contains_key(&instance));

        // Insert first version
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), 1);

        // Entry should exist
        let version = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(version, Some(1));
    }

    /// Test: Version increments on subsequent syncs
    #[tokio::test]
    async fn test_version_increments_on_sync() {
        let runtime = make_runtime();
        let instance = ("worker-incremental".to_string(), 0_usize);

        // Simulate multiple syncs
        for version in 1..=5 {
            runtime
                .instance_to_version_after_sync
                .insert(instance.clone(), version);
            let current = runtime
                .instance_to_version_after_sync
                .get(&instance)
                .map(|r| *r.value());
            assert_eq!(current, Some(version), "Version should be {version}");
        }
    }

    /// Test: Different instances have independent version tracking
    #[tokio::test]
    async fn test_independent_version_tracking_per_instance() {
        let runtime = make_runtime();

        let inst_a = ("worker-a".to_string(), 0_usize);
        let inst_b = ("worker-b".to_string(), 0_usize);
        let inst_c = ("worker-a".to_string(), 1_usize); // Different DP rank

        runtime
            .instance_to_version_after_sync
            .insert(inst_a.clone(), 10);
        runtime
            .instance_to_version_after_sync
            .insert(inst_b.clone(), 5);
        runtime
            .instance_to_version_after_sync
            .insert(inst_c.clone(), 7);

        let v_a = runtime
            .instance_to_version_after_sync
            .get(&inst_a)
            .map(|r| *r.value());
        let v_b = runtime
            .instance_to_version_after_sync
            .get(&inst_b)
            .map(|r| *r.value());
        let v_c = runtime
            .instance_to_version_after_sync
            .get(&inst_c)
            .map(|r| *r.value());

        assert_eq!(v_a, Some(10));
        assert_eq!(v_b, Some(5));
        assert_eq!(v_c, Some(7));
    }

    /// Test: Concurrent version updates to same instance don't conflict
    #[tokio::test]
    async fn test_concurrent_version_updates_same_instance() {
        let runtime = Arc::new(make_runtime());
        let instance = ("worker-concurrent".to_string(), 0_usize);
        let mut handles = vec![];

        // Spawn 5 concurrent tasks updating the same instance
        for i in 1..=5 {
            let rt = Arc::clone(&runtime);
            let inst = instance.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    let version = 10 + i;
                    rt.instance_to_version_after_sync.insert(inst, version);
                    version
                })
            };
            handles.push(handle);
        }

        // Collect all updates
        let mut results = vec![];
        for handle in handles {
            let version = handle.await.unwrap();
            results.push(version);
        }

        // Final version should be one of the concurrent updates
        let final_version = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert!(final_version.is_some());
        assert!(results.contains(&final_version.unwrap()));
    }

    /// Test: Deletion and re-insertion of version entry
    #[tokio::test]
    async fn test_version_entry_deletion_and_reinsertion() {
        let runtime = make_runtime();
        let instance = ("worker-delete-test".to_string(), 0_usize);

        // Insert, verify, delete
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), 5);
        assert!(runtime
            .instance_to_version_after_sync
            .contains_key(&instance));

        runtime.instance_to_version_after_sync.remove(&instance);
        assert!(!runtime
            .instance_to_version_after_sync
            .contains_key(&instance));

        // Re-insert with new version
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), 10);
        let version = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(version, Some(10));
    }

    /// Test: Transitioning from old to new version respects ordering
    #[tokio::test]
    async fn test_version_transition_ordering() {
        let runtime = make_runtime();
        let instances = [
            ("worker-old".to_string(), 0_usize),
            ("worker-new".to_string(), 0_usize),
        ];

        // Set old workers to version 3, new workers to version 5
        runtime
            .instance_to_version_after_sync
            .insert(instances[0].clone(), 3);
        runtime
            .instance_to_version_after_sync
            .insert(instances[1].clone(), 5);

        // Filter: keep only workers >= version_tag 4
        let version_tag = 4;
        let passing = instances
            .iter()
            .filter(|inst| {
                runtime
                    .instance_to_version_after_sync
                    .get(inst)
                    .is_some_and(|v| *v.value() >= version_tag)
            })
            .count();

        assert_eq!(passing, 1, "Only the new worker should pass version filter");
    }

    /// Test: Version tracking with many concurrent instances
    #[tokio::test]
    async fn test_many_concurrent_instances() {
        let runtime = make_runtime();
        let mut handles = vec![];

        // Create 50 instances concurrently
        for worker_id in 0..50 {
            let rt = Arc::clone(&Arc::new(runtime.clone()));
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    let instance = (format!("worker-{worker_id}"), worker_id as usize);
                    rt.instance_to_version_after_sync
                        .insert(instance.clone(), (worker_id + 1) as i64);
                    instance
                })
            };
            handles.push(handle);
        }

        let mut instances_created = vec![];
        for handle in handles {
            let instance = handle.await.unwrap();
            instances_created.push(instance);
        }

        // Verify all were created with correct versions
        for (idx, instance) in instances_created.iter().enumerate() {
            let version = runtime
                .instance_to_version_after_sync
                .get(instance)
                .map(|r| *r.value());
            assert_eq!(version, Some((idx + 1) as i64));
        }
    }

    /// Test: Version monotonicity check (no version regression)
    #[tokio::test]
    async fn test_version_can_regress_on_reinsertion() {
        let runtime = make_runtime();
        let instance = ("worker-regress".to_string(), 0_usize);

        // Note: DashMap doesn't prevent regressions, it just overwrites.
        // This test documents that behavior.
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), 10);
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), 5);

        let version = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(
            version,
            Some(5),
            "Version can regress (no enforcement on insertion)"
        );
    }

    /// Test: Large version numbers don't overflow
    #[tokio::test]
    async fn test_large_version_numbers() {
        let runtime = make_runtime();
        let instance = ("worker-large-version".to_string(), 0_usize);

        let large_version = i64::MAX - 1;
        runtime
            .instance_to_version_after_sync
            .insert(instance.clone(), large_version);

        let version = runtime
            .instance_to_version_after_sync
            .get(&instance)
            .map(|r| *r.value());
        assert_eq!(version, Some(large_version));
    }
}
