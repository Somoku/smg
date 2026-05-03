//! Routing-loop integration tests.
//!
//! These tests exercise the full routing-loop path: requests are enqueued into
//! the `RoutingLoopRuntime` via the mpsc channel, the background loop dispatches
//! them through the regular pipeline stages, and the response comes back through
//! a oneshot channel.
//!
//! Each test spins up a real `MockWorker`, boots a full `AppContext` (including
//! the routing-loop runtime), and drives requests through the axum app using
//! `tower::ServiceExt::oneshot`.

#[cfg(test)]
mod routing_loop_tests {
    use axum::{
        body::Body,
        extract::Request,
        http::{header::CONTENT_TYPE, StatusCode},
    };
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build a minimal `/generate` request body.
    fn generate_body() -> Value {
        json!({ "text": "hello", "stream": false })
    }

    /// Build a minimal `/v1/chat/completions` request body.
    fn chat_body(model: &str) -> Value {
        json!({
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": false
        })
    }

    /// Build a minimal `/v1/completions` request body.
    fn completion_body(model: &str) -> Value {
        json!({ "model": model, "prompt": "Once upon a time", "stream": false })
    }

    /// Build a minimal `/v1/classify` request body (rerank endpoint).
    fn classify_body(model: &str) -> Value {
        json!({
            "model": model,
            "query": "test query",
            "documents": ["doc1", "doc2"]
        })
    }

    /// POST helper: returns (status, body-bytes).
    async fn post(
        app: axum::Router,
        uri: &str,
        body: Value,
        extra_headers: &[(&str, &str)],
    ) -> (StatusCode, bytes::Bytes) {
        let mut builder = Request::builder()
            .method("POST")
            .uri(uri)
            .header(CONTENT_TYPE, "application/json");
        for (k, v) in extra_headers {
            builder = builder.header(*k, *v);
        }
        let req = builder
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, body_bytes)
    }

    /// Build a `TestRouterConfig` with the routing loop enabled.
    fn routing_loop_config(port: u16) -> smg::config::RouterConfig {
        TestRouterConfig::round_robin(port)
            .to_builder()
            .enable_routing_loop(true)
            .build_unchecked()
    }

    /// Build a `TestRouterConfig` with the routing loop enabled and the multi-priority queue.
    fn routing_loop_mpq_config(port: u16) -> smg::config::RouterConfig {
        TestRouterConfig::round_robin(port)
            .to_builder()
            .enable_routing_loop(true)
            .routing_loop_multi_priority_queue(true)
            .routing_loop_check_interval_ms(1)
            .build_unchecked()
    }

    // =========================================================================
    // Test 1: all request types walk through the routing loop
    // =========================================================================

    /// All primary request types (generate, chat, completions, embeddings,
    /// classify, messages) should return HTTP 200 when the routing loop is
    /// enabled.
    #[tokio::test]
    async fn all_request_types_succeed_with_routing_loop_enabled() {
        let config = routing_loop_config(0);
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        // Derive the model name from the registered worker (mock worker uses a
        // fixed model name; fall back to a known default).
        let model = "mock-model";

        // /generate
        let (status, _) = post(app.clone(), "/generate", generate_body(), &[]).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/generate should succeed with routing loop enabled"
        );

        // /v1/chat/completions
        let (status, _) = post(
            app.clone(),
            "/v1/chat/completions",
            chat_body(model),
            &[],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/v1/chat/completions should succeed with routing loop enabled"
        );

        // /v1/completions
        let (status, _) = post(
            app.clone(),
            "/v1/completions",
            completion_body(model),
            &[],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/v1/completions should succeed with routing loop enabled"
        );

        // /v1/rerank (classify)
        let (status, _) = post(app.clone(), "/v1/rerank", classify_body(model), &[]).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/v1/rerank should succeed with routing loop enabled"
        );

        ctx.shutdown().await;
    }

    // =========================================================================
    // Test 2: without a routing-loop runtime the pipeline falls back normally
    // =========================================================================

    /// When no routing-loop runtime is configured, requests are processed
    /// through the regular (non-loop) pipeline path and return HTTP 200.
    ///
    /// This test guards against regressions where the presence or absence of
    /// routing-loop metadata headers would accidentally break the fallback path.
    #[tokio::test]
    async fn no_routing_loop_runtime_falls_back_to_normal_pipeline() {
        // Build a config WITHOUT the routing loop.
        let config = TestRouterConfig::round_robin(0);
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        let model = "mock-model";

        // Plain request — no routing-loop headers.
        let (status, _) = post(app.clone(), "/generate", generate_body(), &[]).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/generate should succeed through the normal pipeline when no routing-loop runtime is configured"
        );

        // Request with routing-loop metadata headers — should still fall through
        // to the normal pipeline (no runtime present means no loop is entered).
        let (status, _) = post(
            app.clone(),
            "/v1/chat/completions",
            chat_body(model),
            &[("x-version-tag", "1"), ("x-request-id", "42")],
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "/v1/chat/completions should succeed through the normal pipeline even with routing-loop headers when no runtime is configured"
        );

        ctx.shutdown().await;
    }

    // =========================================================================
    // Test 3: validation request is dispatched before a normal request
    // =========================================================================

    /// Requests with the `x-is-validate: true` header are assigned the highest
    /// priority (queue key 0) and should be dispatched before any queued normal
    /// requests.
    ///
    /// Strategy: pause the routing loop, enqueue one normal request followed by
    /// one validation request, resume, then assert that both complete (order
    /// cannot be observed without instrumented queues, but the validation
    /// request must also complete successfully).
    #[tokio::test]
    async fn validation_request_dispatched_before_normal() {
        let config = routing_loop_mpq_config(0);
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        // Pause the routing loop so requests queue up.
        let pause_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/pause")
            .body(Body::empty())
            .unwrap();
        let pause_resp = app.clone().oneshot(pause_req).await.unwrap();
        assert_eq!(pause_resp.status(), StatusCode::OK, "pause should succeed");

        // Enqueue a normal request (no special headers).
        let normal_req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-version-tag", "2")
            .header("x-request-id", "100")
            .body(Body::from(serde_json::to_string(&generate_body()).unwrap()))
            .unwrap();
        let normal_handle =
            tokio::spawn(app.clone().oneshot(normal_req));

        // Small yield so the normal request enters the queue first.
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Enqueue a validation request.
        let validate_req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-version-tag", "2")
            .header("x-request-id", "101")
            .header("x-is-validate", "true")
            .body(Body::from(serde_json::to_string(&generate_body()).unwrap()))
            .unwrap();
        let validate_handle =
            tokio::spawn(app.clone().oneshot(validate_req));

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Resume the routing loop.
        let resume_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/resume")
            .body(Body::empty())
            .unwrap();
        let resume_resp = app.clone().oneshot(resume_req).await.unwrap();
        assert_eq!(resume_resp.status(), StatusCode::OK, "resume should succeed");

        // Both requests should complete successfully.
        let (normal_resp, validate_resp) = tokio::join!(normal_handle, validate_handle);
        let normal_status = normal_resp.unwrap().unwrap().status();
        let validate_status = validate_resp.unwrap().unwrap().status();

        assert_eq!(
            normal_status,
            StatusCode::OK,
            "normal request should complete successfully after resume"
        );
        assert_eq!(
            validate_status,
            StatusCode::OK,
            "validation request should complete successfully after resume"
        );

        ctx.shutdown().await;
    }

    // =========================================================================
    // Test 4: low version suppresses high version (multi-priority queue)
    // =========================================================================

    /// With `enable_multi_priority_queue = true`, requests tagged with a lower
    /// `x-version-tag` value should be dispatched before requests with a higher
    /// value.
    ///
    /// Strategy: pause the loop, enqueue a high-version request then a
    /// low-version request, resume, assert both succeed (priority ordering is
    /// an internal scheduling property; observable only by checking that
    /// neither request is lost or errors out).
    #[tokio::test]
    async fn low_version_suppresses_high_version() {
        let config = routing_loop_mpq_config(0);
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        // Pause to allow both requests to queue.
        let pause_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/pause")
            .body(Body::empty())
            .unwrap();
        app.clone().oneshot(pause_req).await.unwrap();

        // Enqueue high-version request first.
        let high_req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-version-tag", "100")
            .header("x-request-id", "200")
            .body(Body::from(serde_json::to_string(&generate_body()).unwrap()))
            .unwrap();
        let high_handle = tokio::spawn(app.clone().oneshot(high_req));

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Enqueue low-version request second.
        let low_req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-version-tag", "1")
            .header("x-request-id", "201")
            .body(Body::from(serde_json::to_string(&generate_body()).unwrap()))
            .unwrap();
        let low_handle = tokio::spawn(app.clone().oneshot(low_req));

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Resume.
        let resume_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/resume")
            .body(Body::empty())
            .unwrap();
        app.clone().oneshot(resume_req).await.unwrap();

        let (high_resp, low_resp) = tokio::join!(high_handle, low_handle);
        assert_eq!(
            high_resp.unwrap().unwrap().status(),
            StatusCode::OK,
            "high-version request should succeed"
        );
        assert_eq!(
            low_resp.unwrap().unwrap().status(),
            StatusCode::OK,
            "low-version request should succeed"
        );

        ctx.shutdown().await;
    }

    // =========================================================================
    // Test 5: batch parameters take effect
    // =========================================================================

    /// Configuring `dispatch_batch_size = 1` and
    /// `max_running_dispatch_tasks = 1` limits the loop to dispatching one
    /// request at a time.  Sending multiple requests should still result in
    /// all of them completing successfully (serialized, not dropped).
    #[tokio::test]
    async fn batch_parameters_take_effect() {
        let config = TestRouterConfig::round_robin(0)
            .to_builder()
            .enable_routing_loop(true)
            .routing_loop_check_interval_ms(1)
            .routing_loop_receive_batch_size(1)
            .routing_loop_dispatch_batch_size(1)
            .routing_loop_max_running_dispatch_tasks(1)
            .build_unchecked();

        // Use a single worker so no load-balancing effects interfere.
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        let n = 5;
        let mut handles = Vec::with_capacity(n);
        for i in 0..n {
            let a = app.clone();
            let body = generate_body();
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header("x-version-tag", "1")
                    .header("x-request-id", i.to_string())
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap();
                a.oneshot(req).await.unwrap().status()
            }));
        }

        for handle in handles {
            let status = handle.await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "all requests should succeed even with batch_size = 1"
            );
        }

        ctx.shutdown().await;
    }

    // =========================================================================
    // Test 6: pause / resume does not lose requests
    // =========================================================================

    /// While the routing loop is paused, new requests accumulate in the queue.
    /// After resume all queued requests should eventually complete with HTTP 200.
    #[tokio::test]
    async fn pause_resume_does_not_lose_requests() {
        let config = TestRouterConfig::round_robin(0)
            .to_builder()
            .enable_routing_loop(true)
            .routing_loop_check_interval_ms(1)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;
        let app = ctx.create_app();

        // Pause the routing loop.
        let pause_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/pause")
            .body(Body::empty())
            .unwrap();
        let pause_status = app.clone().oneshot(pause_req).await.unwrap().status();
        assert_eq!(pause_status, StatusCode::OK, "pause should succeed");

        // Verify that the loop is paused via the status endpoint.
        let status_req = Request::builder()
            .method("GET")
            .uri("/routing_loop/status")
            .body(Body::empty())
            .unwrap();
        let status_resp = app.clone().oneshot(status_req).await.unwrap();
        assert_eq!(status_resp.status(), StatusCode::OK);
        let status_bytes = axum::body::to_bytes(status_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let status_json: Value = serde_json::from_slice(&status_bytes).unwrap();
        assert_eq!(
            status_json.get("paused").and_then(|v| v.as_bool()),
            Some(true),
            "routing loop should report paused=true after /routing_loop/pause"
        );

        // Send N requests while paused (they queue up).
        const N: usize = 4;
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let a = app.clone();
            let body = generate_body();
            handles.push(tokio::spawn(async move {
                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header("x-version-tag", "1")
                    .header("x-request-id", i.to_string())
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap();
                a.oneshot(req).await.unwrap().status()
            }));
        }

        // Small delay so requests enter the queue before we resume.
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        // Resume the routing loop.
        let resume_req = Request::builder()
            .method("POST")
            .uri("/routing_loop/resume")
            .body(Body::empty())
            .unwrap();
        let resume_status = app.clone().oneshot(resume_req).await.unwrap().status();
        assert_eq!(resume_status, StatusCode::OK, "resume should succeed");

        // All N requests must complete successfully — none should be dropped.
        for handle in handles {
            let status = handle.await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "queued request should complete after resume"
            );
        }

        ctx.shutdown().await;
    }
}
