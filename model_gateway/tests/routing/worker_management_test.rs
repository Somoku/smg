//! Worker management integration tests
//!
//! Tests for dynamic worker add/remove operations via management API.
//! The actual worker management API uses:
//! - POST /workers - create a worker
//! - GET /workers - list workers
//! - DELETE /workers/{worker_id} - remove a worker

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use chrono::{TimeZone, Utc};
use openai_protocol::worker::HealthCheckConfig;
use serde_json::json;
use smg::worker::{BasicWorkerBuilder, Worker, WorkerType};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    fn stats_update_payload(
        worker_id: &str,
        dp_rank: Option<usize>,
        timestamp_ms: i64,
        running: usize,
        waiting: usize,
    ) -> serde_json::Value {
        let mut update = json!({
            "worker_id": worker_id,
            "timestamp": Utc.timestamp_millis_opt(timestamp_ms).unwrap().to_rfc3339(),
            "scheduler_stats": {
                "num_running_reqs": running,
                "num_waiting_reqs": waiting
            }
        });
        if let Some(dp_rank) = dp_rank {
            update["dp_rank"] = json!(dp_rank);
        }
        json!({ "updates": [update] })
    }

    async fn post_worker_stats(app: axum::Router, payload: serde_json::Value) -> serde_json::Value {
        let req = Request::builder()
            .method("POST")
            .uri("/workers/update_stats")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn weight_version_update_payload(
        worker_id: &str,
        dp_rank: Option<usize>,
        weight_version: u64,
    ) -> serde_json::Value {
        let mut update = json!({
            "worker_id": worker_id,
            "weight_version": weight_version
        });
        if let Some(dp_rank) = dp_rank {
            update["dp_rank"] = json!(dp_rank);
        }
        json!({ "updates": [update] })
    }

    async fn post_worker_weight_version(
        app: axum::Router,
        payload: serde_json::Value,
    ) -> serde_json::Value {
        let req = Request::builder()
            .method("POST")
            .uri("/workers/update_weight_version")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn ready_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// Test listing workers via API
    #[tokio::test]
    async fn test_list_workers() {
        let config = TestRouterConfig::round_robin(3900);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19900),
                TestWorkerConfig::healthy(19901),
            ],
        )
        .await;

        let app = ctx.create_app();

        // List workers via GET /workers
        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /workers should return OK"
        );

        ctx.shutdown().await;
    }

    /// Test that routing continues to work with multiple workers
    #[tokio::test]
    async fn test_routing_with_multiple_workers() {
        let config = TestRouterConfig::round_robin(3901);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19902),
                TestWorkerConfig::healthy(19903),
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        // Verify routing distributes across workers
        for i in 0..20 {
            let payload = json!({
                "text": format!("Test request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed with multiple workers"
        );

        ctx.shutdown().await;
    }

    /// Test that requests continue to work during worker operations
    #[tokio::test]
    async fn test_requests_during_worker_changes() {
        let config = TestRouterConfig::round_robin(3902);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19904)]).await;

        let app = ctx.create_app();

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request during changes {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 10,
            "All requests should succeed during normal operation"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_stats_applies_by_worker_id() {
        let config = TestRouterConfig::round_robin(3903);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let worker = ready_worker("http://worker-stats:8080");
        let worker_id = ctx
            .app_context
            .worker_registry
            .register(worker.clone())
            .unwrap();
        let body = post_worker_stats(
            app,
            stats_update_payload(worker_id.as_str(), None, 1_700_000_000_000, 5, 3),
        )
        .await;

        assert_eq!(body["total"], 1);
        assert_eq!(body["updated"], 1);
        assert_eq!(worker.engine_stats().running_queue_size(), 5);
        assert_eq!(worker.engine_stats().waiting_queue_size(), 3);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_stats_resolves_base_id_and_dp_rank() {
        let config = TestRouterConfig::round_robin(3904);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let base_id = ctx
            .app_context
            .worker_registry
            .reserve_id_for_url("http://worker:8080");
        let dp_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .dp_config(1, 2)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let dp_worker_id = ctx
            .app_context
            .worker_registry
            .register(dp_worker.clone())
            .unwrap();

        let body = post_worker_stats(
            app.clone(),
            stats_update_payload(base_id.as_str(), Some(1), 1_700_000_000_000, 7, 2),
        )
        .await;

        assert_eq!(body["updated"], 1);
        assert_eq!(body["results"][0]["worker_id"], dp_worker_id.as_str());
        assert_eq!(dp_worker.engine_stats().waiting_queue_size(), 2);

        let rejected = post_worker_stats(
            app,
            stats_update_payload(dp_worker_id.as_str(), None, 1_700_000_001_000, 1, 0),
        )
        .await;
        assert_eq!(rejected["rejected"], 1);
        assert_eq!(dp_worker.engine_stats().running_queue_size(), 7);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_stats_rejects_older_snapshot() {
        let config = TestRouterConfig::round_robin(3905);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let worker = ready_worker("http://worker-stale:8080");
        let worker_id = ctx.app_context.worker_registry.register(worker).unwrap();

        let first = post_worker_stats(
            app.clone(),
            stats_update_payload(worker_id.as_str(), None, 1_700_000_010_000, 4, 0),
        )
        .await;
        assert_eq!(first["updated"], 1);

        let stale = post_worker_stats(
            app,
            stats_update_payload(worker_id.as_str(), None, 1_700_000_000_000, 1, 0),
        )
        .await;
        assert_eq!(stale["stale_ignored"], 1);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_weight_version_updates_runtime_value() {
        let config = TestRouterConfig::round_robin(3906);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let worker = ready_worker("http://worker-weight-version:8080");
        let worker_id = ctx
            .app_context
            .worker_registry
            .register(worker.clone())
            .unwrap();

        let body = post_worker_weight_version(
            app,
            weight_version_update_payload(worker_id.as_str(), None, 42),
        )
        .await;

        assert_eq!(body["total"], 1);
        assert_eq!(body["updated"], 1);
        assert_eq!(body["rejected"], 0);
        assert_eq!(worker.dyn_weight_version(), 42);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_weight_version_resolves_base_id_and_dp_rank() {
        let config = TestRouterConfig::round_robin(3907);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let base_id = ctx
            .app_context
            .worker_registry
            .reserve_id_for_url("http://worker-weight-version-dp:8080");
        let dp_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker-weight-version-dp:8080")
                .dp_config(1, 2)
                .worker_type(WorkerType::Regular)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let dp_worker_id = ctx
            .app_context
            .worker_registry
            .register(dp_worker.clone())
            .unwrap();

        let body = post_worker_weight_version(
            app,
            weight_version_update_payload(base_id.as_str(), Some(1), 7),
        )
        .await;

        assert_eq!(body["updated"], 1);
        assert_eq!(body["results"][0]["worker_id"], dp_worker_id.as_str());
        assert_eq!(dp_worker.dyn_weight_version(), 7);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_update_worker_weight_version_rejects_missing_worker() {
        let config = TestRouterConfig::round_robin(3908);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let app = ctx.create_app();

        let body = post_worker_weight_version(
            app,
            weight_version_update_payload("00000000-0000-0000-0000-000000000001", None, 1),
        )
        .await;

        assert_eq!(body["updated"], 0);
        assert_eq!(body["rejected"], 1);

        ctx.shutdown().await;
    }
}
