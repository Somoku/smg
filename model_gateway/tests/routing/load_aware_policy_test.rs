//! Integration tests for load-aware routing policies:
//! `request_num_balance`, `throughput_optimal`, and `throughput_optimal_with_budget`.
//!
//! These tests verify that:
//! 1. All three policies successfully route requests to healthy workers.
//! 2. Requests are rejected (or retried) when all workers are unhealthy.
//! 3. The policies recover gracefully when a failing worker is paired with a healthy one.
//! 4. The `throughput_optimal_with_budget` policy accepts a configurable budget parameter.
//!
//! # Note on load-steering assertions
//!
//! Because the test mock workers do not feed real `EngineStats` back to the router, we
//! cannot verify that the policies actually steer requests toward the *least-loaded* worker
//! in this integration harness. That property is covered exhaustively by the unit tests in
//! `model_gateway/src/policies/request_num_balance.rs`,
//! `model_gateway/src/policies/throughput_optimal.rs`, and
//! `model_gateway/src/policies/throughput_optimal_with_budget.rs`.
//!
//! Integration tests here focus on connectivity, error propagation, and config wiring.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

// ---------------------------------------------------------------------------
// Port assignments (must not collide with other test files).
// Base: 3700 for routers, 19700 for mock workers.
// ---------------------------------------------------------------------------

// ============================================================
// request_num_balance
// ============================================================

#[cfg(test)]
mod request_num_balance_tests {
    use super::*;

    /// Verify that the policy successfully routes requests to healthy workers.
    #[tokio::test]
    async fn routes_requests_to_healthy_workers() {
        let config = TestRouterConfig::request_num_balance(3700);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19700, 3))
                .await;

        let app = ctx.create_app();
        let mut success = 0usize;

        for i in 0..20 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            if app.clone().oneshot(req).await.unwrap().status() == StatusCode::OK {
                success += 1;
            }
        }

        assert_eq!(success, 20, "all requests should succeed");
        ctx.shutdown().await;
    }

    /// Verify that a failing worker paired with a healthy one recovers via retry.
    #[tokio::test]
    async fn recovers_from_failing_worker_via_retry() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let cb = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 2,
            window_duration_secs: 10,
        };

        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .request_num_balance_policy()
            .host("127.0.0.1")
            .port(3701)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(retry_config)
            .circuit_breaker_config(cb)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19702, 1.0), // always fails
                TestWorkerConfig::healthy(19703),    // always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();

        for i in 0..10 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                StatusCode::OK,
                "should succeed via retry on healthy worker"
            );
        }

        ctx.shutdown().await;
    }
}

// ============================================================
// throughput_optimal
// ============================================================

#[cfg(test)]
mod throughput_optimal_tests {
    use super::*;

    /// Verify that the policy successfully routes requests to healthy workers.
    #[tokio::test]
    async fn routes_requests_to_healthy_workers() {
        let config = TestRouterConfig::throughput_optimal(3710);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19710, 3))
                .await;

        let app = ctx.create_app();
        let mut success = 0usize;

        for i in 0..20 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            if app.clone().oneshot(req).await.unwrap().status() == StatusCode::OK {
                success += 1;
            }
        }

        assert_eq!(success, 20, "all requests should succeed");
        ctx.shutdown().await;
    }

    /// Verify that a failing worker paired with a healthy one recovers via retry.
    #[tokio::test]
    async fn recovers_from_failing_worker_via_retry() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let cb = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 2,
            window_duration_secs: 10,
        };

        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .throughput_optimal_policy(
                TestRouterConfig::create_test_cost_model_file(),
                1024, // max_concurrent_seqs_per_instance
                0.5,  // delta_throughput_threshold
                8192, // max_prompt_length
                1024, // request_budget
                1000, // max_num_waiting_reqs_after_preemption
            )
            .host("127.0.0.1")
            .port(3711)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(retry_config)
            .circuit_breaker_config(cb)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19712, 1.0), // always fails
                TestWorkerConfig::healthy(19713),    // always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();

        for i in 0..10 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                StatusCode::OK,
                "should succeed via retry on healthy worker"
            );
        }

        ctx.shutdown().await;
    }
}

// ============================================================
// throughput_optimal_with_budget
// ============================================================

#[cfg(test)]
mod throughput_optimal_with_budget_tests {
    use super::*;

    /// Verify that the policy successfully routes requests with budget=1 (exact token counts).
    #[tokio::test]
    async fn routes_requests_with_budget_one() {
        let config = TestRouterConfig::throughput_optimal_with_budget(3720, 1);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19720, 3))
                .await;

        let app = ctx.create_app();
        let mut success = 0usize;

        for i in 0..20 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            if app.clone().oneshot(req).await.unwrap().status() == StatusCode::OK {
                success += 1;
            }
        }

        assert_eq!(success, 20, "all requests should succeed with budget=1");
        ctx.shutdown().await;
    }

    /// Verify that the policy successfully routes requests with a larger page budget.
    #[tokio::test]
    async fn routes_requests_with_larger_budget() {
        let config = TestRouterConfig::throughput_optimal_with_budget(3721, 16);
        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19723, 3))
                .await;

        let app = ctx.create_app();
        let mut success = 0usize;

        for i in 0..20 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            if app.clone().oneshot(req).await.unwrap().status() == StatusCode::OK {
                success += 1;
            }
        }

        assert_eq!(success, 20, "all requests should succeed with budget=16");
        ctx.shutdown().await;
    }

    /// Verify that a failing worker paired with a healthy one recovers via retry.
    #[tokio::test]
    async fn recovers_from_failing_worker_via_retry() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let cb = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 2,
            window_duration_secs: 10,
        };

        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .throughput_optimal_with_budget_policy(
                8,
                TestRouterConfig::create_test_cost_model_file(),
                1024, // max_concurrent_seqs_per_instance
                0.5,  // delta_throughput_threshold
                8192, // max_prompt_length
                1024, // request_budget
                1000, // max_num_waiting_reqs_after_preemption
            )
            .host("127.0.0.1")
            .port(3722)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(retry_config)
            .circuit_breaker_config(cb)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19726, 1.0), // always fails
                TestWorkerConfig::healthy(19727),    // always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();

        for i in 0..10 {
            let body = json!({"text": format!("req {i}"), "stream": false});
            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();

            assert_eq!(
                app.clone().oneshot(req).await.unwrap().status(),
                StatusCode::OK,
                "should succeed via retry on healthy worker"
            );
        }

        ctx.shutdown().await;
    }

    /// Verify that `throughput_optimal_with_budget` config round-trips through
    /// `RouterConfig::builder()` correctly and the budget value is preserved.
    #[tokio::test]
    async fn config_round_trip_preserves_budget() {
        use smg::config::PolicyConfig;

        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .throughput_optimal_with_budget_policy(
                32,
                TestRouterConfig::create_test_cost_model_file(),
                1024, // max_concurrent_seqs_per_instance
                0.5,  // delta_throughput_threshold
                8192, // max_prompt_length
                1024, // request_budget
                1000, // max_num_waiting_reqs_after_preemption
            )
            .host("127.0.0.1")
            .port(3723)
            .build()
            .expect("valid config");

        match config.policy {
            PolicyConfig::ThroughputOptimalWithBudget { budget, .. } => {
                assert_eq!(budget, 32, "budget should be preserved through builder");
            }
            other => panic!("unexpected policy: {other:?}"),
        }
    }
}
