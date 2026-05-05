//! Phase 3: Metadata Parsing Tests (RoutingMeta extraction and validation)

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};
    use serde_json::json;

    use crate::routers::grpc::routing_loop::metadata::parse_routing_request_meta;

    /// Test: Empty headers and body return None
    #[test]
    fn test_parse_empty_returns_none() {
        let result = parse_routing_request_meta(None, None);
        assert_eq!(result, None, "Empty metadata should return None");
    }

    /// Test: Parse request_id from header (prompt_id also required)
    #[test]
    fn test_parse_request_id_from_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("42"));
        headers.insert("x-prompt-id", HeaderValue::from_static("1"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_id, 42);
    }

    /// Test: Missing prompt_id causes None even when request_id is present
    #[test]
    fn test_missing_prompt_id_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("42"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert_eq!(result, None, "Missing prompt_id should return None");
    }

    /// Test: Missing request_id causes None even when prompt_id is present
    #[test]
    fn test_missing_request_id_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert("x-prompt-id", HeaderValue::from_static("7"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert_eq!(result, None, "Missing request_id should return None");
    }

    /// Test: Parse request_id from body when header missing (prompt_id also required)
    #[test]
    fn test_parse_request_id_from_body() {
        let body = json!({"request_id": 99, "prompt_id": 1});

        let result = parse_routing_request_meta(None, Some(&body));
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_id, 99);
    }

    /// Test: Header takes precedence over body for request_id
    #[test]
    fn test_header_precedence_over_body() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("100"));
        headers.insert("x-prompt-id", HeaderValue::from_static("1"));

        let body = json!({"request_id": 50, "prompt_id": 99});

        let result = parse_routing_request_meta(Some(&headers), Some(&body));
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_id, 100, "Header should take precedence");
    }

    /// Test: Parse prompt_id from header (request_id also required)
    #[test]
    fn test_parse_prompt_id_from_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("7"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().prompt_id, 7);
    }

    /// Test: Parse version_tag with default -1
    #[test]
    fn test_parse_version_tag_default() {
        let result = parse_routing_request_meta(None, None);
        // When no metadata at all, returns None, not default metadata
        assert_eq!(result, None);
    }

    /// Test: Parse version_tag from header (both IDs required)
    #[test]
    fn test_parse_version_tag_from_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("2"));
        headers.insert("x-version-tag", HeaderValue::from_static("5"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().version_tag, 5);
    }

    /// Test: Parse boolean is_validate flag (both IDs required)
    #[test]
    fn test_parse_is_validate_flag() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("2"));
        headers.insert("x-is-validate", HeaderValue::from_static("true"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().is_validate, true);
    }

    /// Test: Parse boolean variations (yes, 1, True, etc.)
    #[test]
    fn test_parse_boolean_variations() {
        let variations = vec!["true", "True", "TRUE", "yes", "YES", "y", "Y", "1"];

        for val in variations {
            let mut headers = HeaderMap::new();
            headers.insert("x-is-validate", HeaderValue::from_str(val).unwrap());
            // Both IDs required for result to be Some
            headers.insert("x-request-id", HeaderValue::from_static("1"));
            headers.insert("x-prompt-id", HeaderValue::from_static("2"));

            let result = parse_routing_request_meta(Some(&headers), None);
            assert!(
                result.is_some_and(|m| m.is_validate),
                "Should parse '{}' as true",
                val
            );
        }
    }

    /// Test: Parse false boolean variations
    #[test]
    fn test_parse_false_boolean_variations() {
        let variations = vec!["false", "False", "FALSE", "no", "NO", "n", "N", "0"];

        for val in variations {
            let mut headers = HeaderMap::new();
            headers.insert("x-is-validate", HeaderValue::from_str(val).unwrap());
            // Both IDs required for result to be Some
            headers.insert("x-request-id", HeaderValue::from_static("2"));
            headers.insert("x-prompt-id", HeaderValue::from_static("3"));

            let result = parse_routing_request_meta(Some(&headers), None);
            assert!(
                result.is_some_and(|m| !m.is_validate),
                "Should parse '{}' as false",
                val
            );
        }
    }

    /// Test: Parse rollout_instance_hint from headers (both IDs required)
    #[test]
    fn test_parse_rollout_instance_hint_from_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("2"));
        headers.insert("x-base-worker-id", HeaderValue::from_static("worker-a"));
        headers.insert("x-target-dp-rank", HeaderValue::from_static("2"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        let meta = result.unwrap();
        assert_eq!(meta.rollout_instance_hint, Some(("worker-a".to_string(), 2)));
    }

    /// Test: Parse rollout_instance_hint as array from body (both IDs required)
    #[test]
    fn test_parse_rollout_instance_hint_array_from_body() {
        let body = json!({
            "request_id": 1,
            "prompt_id": 2,
            "rollout_instance_id": ["worker-b", 3]
        });

        let result = parse_routing_request_meta(None, Some(&body));
        assert!(result.is_some());
        let meta = result.unwrap();
        assert_eq!(meta.rollout_instance_hint, Some(("worker-b".to_string(), 3)));
    }

    /// Test: Parse rollout_instance_hint as object from body (both IDs required)
    #[test]
    fn test_parse_rollout_instance_hint_object_from_body() {
        let body = json!({
            "request_id": 1,
            "prompt_id": 2,
            "rollout_instance_hint": {
                "base_worker_id": "worker-c",
                "target_dp_rank": 1
            }
        });

        let result = parse_routing_request_meta(None, Some(&body));
        assert!(result.is_some());
        let meta = result.unwrap();
        assert_eq!(meta.rollout_instance_hint, Some(("worker-c".to_string(), 1)));
    }

    /// Test: Fallback to replica_id alias in object (both IDs required)
    #[test]
    fn test_parse_replica_id_alias() {
        let body = json!({
            "request_id": 1,
            "prompt_id": 2,
            "rollout_instance_hint": {
                "replica_id": "worker-d",
                "data_parallel_rank": 4
            }
        });

        let result = parse_routing_request_meta(None, Some(&body));
        assert!(result.is_some());
        let meta = result.unwrap();
        assert_eq!(meta.rollout_instance_hint, Some(("worker-d".to_string(), 4)));
    }

    /// Test: Complete metadata with all fields
    #[test]
    fn test_complete_metadata_parsing() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("42"));
        headers.insert("x-prompt-id", HeaderValue::from_static("7"));
        headers.insert("x-version-tag", HeaderValue::from_static("3"));
        headers.insert("x-is-validate", HeaderValue::from_static("true"));
        headers.insert("x-is-sticky", HeaderValue::from_static("false"));
        headers.insert("x-base-worker-id", HeaderValue::from_static("worker-main"));
        headers.insert("x-target-dp-rank", HeaderValue::from_static("0"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());

        let meta = result.unwrap();
        assert_eq!(meta.request_id, 42);
        assert_eq!(meta.prompt_id, 7);
        assert_eq!(meta.version_tag, 3);
        assert_eq!(meta.is_validate, true);
        assert_eq!(meta.is_sticky, false);
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("worker-main".to_string(), 0))
        );
    }

    /// Test: Parse request_id as string in body (prompt_id also required)
    #[test]
    fn test_parse_request_id_string_in_body() {
        let body = json!({"request_id": "123", "prompt_id": 1});

        let result = parse_routing_request_meta(None, Some(&body));
        assert!(result.is_some());
        assert_eq!(result.unwrap().request_id, 123);
    }

    /// Test: Parse version_tag as string (both IDs required)
    #[test]
    fn test_parse_version_tag_string() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("2"));
        headers.insert("x-version-tag", HeaderValue::from_static("  99  "));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        assert_eq!(result.unwrap().version_tag, 99);
    }

    /// Test: Invalid request_id causes None (both IDs required; unparseable → whole result None)
    #[test]
    fn test_invalid_request_id_returns_none() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("not-a-number"));
        headers.insert("x-prompt-id", HeaderValue::from_static("7"));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert_eq!(
            result, None,
            "Unparseable request_id means the whole RoutingMeta is absent"
        );
    }

    /// Test: Whitespace handling in headers (both IDs required)
    #[test]
    fn test_whitespace_handling_in_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("1"));
        headers.insert("x-prompt-id", HeaderValue::from_static("2"));
        headers.insert(
            "x-base-worker-id",
            HeaderValue::from_static("  worker-padded  "),
        );
        headers.insert("x-target-dp-rank", HeaderValue::from_static("  1  "));

        let result = parse_routing_request_meta(Some(&headers), None);
        assert!(result.is_some());
        let meta = result.unwrap();
        assert_eq!(
            meta.rollout_instance_hint,
            Some(("worker-padded".to_string(), 1))
        );
    }
}
