//! Phase 6: Concurrent Session Tests (Session concurrency, locking, race conditions)

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::Barrier;

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

    /// Test: Concurrent writes to version map by multiple tasks
    #[tokio::test]
    async fn test_concurrent_version_map_writes() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(5));
        let mut handles = vec![];

        for i in 0..5 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    for j in 0..10 {
                        let version = (i * 10 + j) as i64;
                        rt.instance_to_version_after_sync
                            .insert(("worker".to_string(), i as usize), version);
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify that each worker has the final version from its task
        for i in 0..5 {
            let final_version = runtime
                .instance_to_version_after_sync
                .get(&("worker".to_string(), i as usize))
                .map(|r| *r.value());
            assert_eq!(final_version, Some((i * 10 + 9) as i64));
        }
    }

    /// Test: Concurrent record selections to prompt tracking map (write-once semantics)
    #[tokio::test]
    async fn test_concurrent_group_pin_write_once_semantics() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(5));
        let mut handles = vec![];

        // All 5 tasks race to pin the same prompt to different instances
        for i in 0..5 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    rt.record_selected_instance(
                        i as i64,
                        Some(42), // Same prompt_id
                        (format!("worker-{i}"), i as usize),
                    );
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Exactly one instance should be pinned (write-once semantics)
        let pinned = runtime
            .prompt_to_pinned_instance
            .get(&42)
            .map(|r| r.value().clone());
        assert!(pinned.is_some());

        // All 5 requests should be recorded regardless of which pinned first
        let request_ids = runtime.prompt_to_running_request_ids.get(&42).map(|r| {
            let mut ids = r.value().clone();
            ids.sort_unstable();
            ids
        });
        assert_eq!(request_ids, Some(vec![0, 1, 2, 3, 4]));
    }

    /// Test: Concurrent cleanup operations on same prompt
    #[tokio::test]
    async fn test_concurrent_cleanup_same_prompt() {
        let runtime = make_runtime();

        // Pre-populate with 5 requests in same prompt group
        for i in 0..5 {
            runtime.record_selected_instance(i as i64, Some(42), ("worker-initial".to_string(), 0));
        }

        let barrier = Arc::new(Barrier::new(5));
        let mut handles = vec![];

        // All 5 tasks race to cleanup their request
        for i in 0..5 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    rt.cleanup_tracking(Some(i as i64), Some(42));
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // After all cleanups, prompt group should be completely gone
        let has_prompt = runtime.prompt_to_running_request_ids.contains_key(&42);
        let has_pin = runtime.prompt_to_pinned_instance.contains_key(&42);

        assert!(!has_prompt, "Prompt group should be removed");
        assert!(!has_pin, "Pinned instance should be removed");
    }

    /// Test: Concurrent cleanup with staggered completion (some remain)
    #[tokio::test]
    async fn test_concurrent_cleanup_partial_completion() {
        let runtime = make_runtime();

        // Pre-populate with 3 requests in same prompt group
        for i in 0..3 {
            runtime.record_selected_instance(i as i64, Some(99), ("worker-initial".to_string(), 0));
        }

        // Cleanup first 2 requests
        runtime.cleanup_tracking(Some(0), Some(99));
        runtime.cleanup_tracking(Some(1), Some(99));

        // Verify group still exists with request 2
        let remaining = runtime
            .prompt_to_running_request_ids
            .get(&99)
            .map(|r| r.value().clone());
        assert_eq!(remaining, Some(vec![2]));

        // Pinned instance should still be there
        let still_pinned = runtime.prompt_to_pinned_instance.contains_key(&99);
        assert!(still_pinned);

        // Cleanup the last request
        runtime.cleanup_tracking(Some(2), Some(99));

        // Now everything should be gone
        let prompt_gone = !runtime.prompt_to_running_request_ids.contains_key(&99);
        let pin_gone = !runtime.prompt_to_pinned_instance.contains_key(&99);
        assert!(prompt_gone);
        assert!(pin_gone);
    }

    /// Test: Concurrent version updates to independent workers
    #[tokio::test]
    async fn test_concurrent_version_updates_independent_workers() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        // 10 tasks, each updating a different worker independently
        for worker_idx in 0..10 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    for version in 1..=5 {
                        rt.instance_to_version_after_sync
                            .insert((format!("worker-{worker_idx}"), 0), version as i64);
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Each worker should have final version 5
        for worker_idx in 0..10 {
            let version = runtime
                .instance_to_version_after_sync
                .get(&(format!("worker-{worker_idx}"), 0))
                .map(|r| *r.value());
            assert_eq!(
                version,
                Some(5),
                "Worker {worker_idx} should have version 5",
            );
        }
    }

    /// Test: Concurrent record selections to different prompts
    #[tokio::test]
    async fn test_concurrent_group_pin_different_prompts() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        // 10 tasks, each recording to a different prompt
        for prompt_idx in 0..10 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    for req in 0..3 {
                        rt.record_selected_instance(
                            (prompt_idx * 100 + req) as i64,
                            Some(prompt_idx as i64),
                            (format!("worker-{prompt_idx}"), 0),
                        );
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Each prompt should have its own 3 requests and pinned instance
        for prompt_idx in 0..10 {
            let request_ids = runtime
                .prompt_to_running_request_ids
                .get(&(prompt_idx as i64))
                .map(|r| r.value().len());
            assert_eq!(request_ids, Some(3));

            let pinned = runtime
                .prompt_to_pinned_instance
                .get(&(prompt_idx as i64))
                .map(|r| r.value().clone());
            assert_eq!(pinned, Some((format!("worker-{prompt_idx}"), 0)));
        }
    }

    /// Test: Concurrent cleanups on different prompts
    #[tokio::test]
    async fn test_concurrent_cleanup_different_prompts() {
        let runtime = make_runtime();

        // Pre-populate 5 different prompts with 3 requests each
        for prompt_idx in 0..5 {
            for req_idx in 0..3 {
                runtime.record_selected_instance(
                    (prompt_idx * 100 + req_idx) as i64,
                    Some(prompt_idx as i64),
                    (format!("worker-{prompt_idx}"), 0),
                );
            }
        }

        let barrier = Arc::new(Barrier::new(15)); // 5 prompts * 3 requests each
        let mut handles = vec![];

        // Cleanup all requests concurrently
        for prompt_idx in 0..5 {
            for req_idx in 0..3 {
                let rt = runtime.clone();
                let b = barrier.clone();
                let handle = {
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "test code: tasks are joined before test ends"
                    )]
                    tokio::spawn(async move {
                        b.wait().await;
                        rt.cleanup_tracking(
                            Some((prompt_idx * 100 + req_idx) as i64),
                            Some(prompt_idx as i64),
                        );
                    })
                };
                handles.push(handle);
            }
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // All prompt groups should be gone
        for prompt_idx in 0..5 {
            let has_prompt = runtime
                .prompt_to_running_request_ids
                .contains_key(&(prompt_idx as i64));
            let has_pin = runtime
                .prompt_to_pinned_instance
                .contains_key(&(prompt_idx as i64));
            assert!(!has_prompt);
            assert!(!has_pin);
        }
    }

    /// Test: Sequential record-then-cleanup pattern (stable)
    #[tokio::test]
    async fn test_sequential_record_then_cleanup_pattern() {
        let runtime = make_runtime();

        // Phase 1: All tasks record requests for prompts 0-4
        let barrier1 = Arc::new(Barrier::new(5));
        let mut handles = vec![];
        for prompt_idx in 0..5 {
            let rt = runtime.clone();
            let b = barrier1.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    for req_idx in 0..3 {
                        rt.record_selected_instance(
                            (prompt_idx * 100 + req_idx) as i64,
                            Some(prompt_idx as i64),
                            (format!("worker-{prompt_idx}"), 0),
                        );
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Phase 1 task should complete");
        }

        // Verify all records exist
        for prompt_idx in 0..5 {
            assert!(runtime
                .prompt_to_running_request_ids
                .contains_key(&(prompt_idx as i64)));
        }

        // Phase 2: All tasks cleanup their requests
        let barrier2 = Arc::new(Barrier::new(15)); // 5 prompts * 3 requests
        let mut handles = vec![];
        for prompt_idx in 0..5 {
            for req_idx in 0..3 {
                let rt = runtime.clone();
                let b = barrier2.clone();
                let handle = {
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "test code: tasks are joined before test ends"
                    )]
                    tokio::spawn(async move {
                        b.wait().await;
                        rt.cleanup_tracking(
                            Some((prompt_idx * 100 + req_idx) as i64),
                            Some(prompt_idx as i64),
                        );
                    })
                };
                handles.push(handle);
            }
        }

        for handle in handles {
            handle.await.expect("Phase 2 task should complete");
        }

        // All prompt groups should be gone
        for prompt_idx in 0..5 {
            assert!(!runtime
                .prompt_to_running_request_ids
                .contains_key(&(prompt_idx as i64)));
            assert!(!runtime
                .prompt_to_pinned_instance
                .contains_key(&(prompt_idx as i64)));
        }
    }

    /// Test: High concurrency with many workers and versions
    #[tokio::test]
    async fn test_high_concurrency_many_workers_many_versions() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(50));
        let mut handles = vec![];

        // 50 tasks, each updating 5 different workers with 10 versions each
        for task_idx in 0..50 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    for worker_idx in 0..5 {
                        for version in 1..=10 {
                            let actual_version = (task_idx * 50 + worker_idx * 10 + version) as i64;
                            rt.instance_to_version_after_sync.insert(
                                (format!("worker-{worker_idx}"), task_idx as usize),
                                actual_version,
                            );
                        }
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify the map contains all expected entries
        let total_entries = runtime.instance_to_version_after_sync.len();
        assert_eq!(
            total_entries,
            5 * 50,
            "Should have 250 distinct (worker, rank) entries"
        );
    }

    /// Test: Concurrent version reads while writes happening
    #[tokio::test]
    async fn test_concurrent_reads_with_writes() {
        let runtime = make_runtime();

        // Pre-populate with initial versions
        for i in 0..10 {
            runtime
                .instance_to_version_after_sync
                .insert(("worker".to_string(), i), 10);
        }

        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        // 5 writers, 5 readers
        for _i in 0..5 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    // Writer: increment versions
                    for _ in 0..5 {
                        for j in 0..10 {
                            if let Some(mut entry) = rt
                                .instance_to_version_after_sync
                                .get_mut(&("worker".to_string(), j))
                            {
                                *entry += 1;
                            }
                        }
                    }
                })
            };
            handles.push(handle);
        }

        for _i in 5..10 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    // Reader: collect all versions
                    for _ in 0..25 {
                        for j in 0..10 {
                            let _ = rt
                                .instance_to_version_after_sync
                                .get(&("worker".to_string(), j))
                                .map(|r| *r.value());
                        }
                    }
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify final state: each worker should have been incremented 25 times
        for j in 0..10 {
            let final_version = runtime
                .instance_to_version_after_sync
                .get(&("worker".to_string(), j))
                .map(|r| *r.value());
            assert_eq!(
                final_version,
                Some(10 + 25),
                "Worker {j} should have version 35",
            );
        }
    }

    /// Test: High concurrency with diverse prompt groups
    #[tokio::test]
    async fn test_high_concurrency_diverse_prompt_groups() {
        let runtime = make_runtime();
        let barrier = Arc::new(Barrier::new(100));
        let mut handles = vec![];

        // 100 tasks recording to 20 different prompts
        for task_idx in 0..100 {
            let rt = runtime.clone();
            let b = barrier.clone();
            let handle = {
                #[expect(
                    clippy::disallowed_methods,
                    reason = "test code: tasks are joined before test ends"
                )]
                tokio::spawn(async move {
                    b.wait().await;
                    let prompt_idx = task_idx % 20;
                    rt.record_selected_instance(
                        task_idx as i64,
                        Some(prompt_idx as i64),
                        (format!("worker-{prompt_idx}"), 0),
                    );
                })
            };
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("Task should complete");
        }

        // Verify each prompt has 5 requests (100 / 20 = 5)
        for prompt_idx in 0..20 {
            let count = runtime
                .prompt_to_running_request_ids
                .get(&(prompt_idx as i64))
                .map(|r| r.value().len());
            assert_eq!(count, Some(5), "Prompt {prompt_idx} should have 5 requests",);
        }
    }
}
