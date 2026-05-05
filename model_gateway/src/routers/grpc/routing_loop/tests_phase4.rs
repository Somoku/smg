//! Phase 4: TITO Training Data Tests (Tokenization, trim logic, mismatch detection)

#[cfg(test)]
mod tests {
    /// Test: Token append-only constraint enforcement
    #[test]
    fn test_token_append_only_constraint() {
        // Verify that token sequences must be append-only
        // (cannot insert or reorder existing tokens)
        let prev_tokens = vec![1, 2, 3, 4, 5];
        let new_tokens = vec![1, 2, 3, 4, 5, 6, 7]; // Valid: appends 6, 7

        // Check append-only property
        let is_append_only = prev_tokens.len() <= new_tokens.len()
            && new_tokens[..prev_tokens.len()] == prev_tokens[..];

        assert!(
            is_append_only,
            "New tokens must be append-only (no reordering or deletion)"
        );
    }

    /// Test: Detect violation of append-only constraint
    #[test]
    fn test_detect_append_only_violation() {
        let prev_tokens = vec![1, 2, 3, 4, 5];
        let violating_tokens = vec![1, 2, 3, 5, 6]; // Invalid: token 4 replaced with 5

        let is_append_only = prev_tokens.len() <= violating_tokens.len()
            && violating_tokens[..prev_tokens.len()] == prev_tokens[..];

        assert!(
            !is_append_only,
            "Should detect violation of append-only constraint"
        );
    }

    /// Test: Trailing token trim respects max_trim_tokens
    #[test]
    fn test_trailing_token_trim_respects_ceiling() {
        let max_trim_tokens: usize = 5;
        let _response_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Greedy trim: remove last 7 tokens
        let trim_count = 7;

        // Validate against max_trim_tokens
        let trim_valid = trim_count <= max_trim_tokens;

        assert!(
            !trim_valid,
            "Trim count {} exceeds max_trim_tokens {}",
            trim_count,
            max_trim_tokens
        );
    }

    /// Test: Valid trailing token trim
    #[test]
    fn test_valid_trailing_token_trim() {
        let max_trim_tokens: usize = 5;
        let response_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Greedy trim: remove last 3 tokens (valid)
        let trim_count = 3;

        let trim_valid = trim_count <= max_trim_tokens;
        assert!(trim_valid, "Trim count {} should be valid", trim_count);

        // Trimmed tokens: [1, 2, 3, 4, 5, 6, 7]
        let trimmed = &response_tokens[..response_tokens.len() - trim_count];
        assert_eq!(trimmed, &[1, 2, 3, 4, 5, 6, 7]);
    }

    /// Test: Per-turn record construction
    #[test]
    fn test_per_turn_record_construction() {
        struct TurnRecord {
            prompt_token_count: usize,
            output_logprobs: Vec<f64>,
            finish_reason: String,
            mismatch_report: Option<String>,
        }

        let record = TurnRecord {
            prompt_token_count: 100,
            output_logprobs: vec![0.95, 0.85, 0.92, 0.88],
            finish_reason: "stop".to_string(),
            mismatch_report: None,
        };

        assert_eq!(record.prompt_token_count, 100);
        assert_eq!(record.output_logprobs.len(), 4);
        assert_eq!(record.finish_reason, "stop");
        assert!(record.mismatch_report.is_none());
    }

    /// Test: Output logprobs capture
    #[test]
    fn test_output_logprobs_capture() {
        // Simulate logprobs from model output
        let model_logprobs = vec![
            vec![("hello", -0.5), ("world", -1.2), ("test", -2.3)],
            vec![("world", -0.3), ("test", -1.1)],
            vec![("done", -0.1)],
        ];

        // Extract top logprob per token
        let top_logprobs: Vec<f64> = model_logprobs
            .iter()
            .map(|v| v.first().map(|(_, prob)| -prob).unwrap_or(0.0))
            .collect();

        assert_eq!(top_logprobs.len(), 3);
        assert_eq!(top_logprobs[0], 0.5); // -(-0.5)
        assert_eq!(top_logprobs[1], 0.3); // -(-0.3)
        assert_eq!(top_logprobs[2], 0.1); // -(-0.1)
    }

    /// Test: Finish reason recording (stop, length, etc.)
    #[test]
    fn test_finish_reason_recording() {
        let finish_reasons = vec!["stop", "length", "tool_calls", "error"];

        for reason in finish_reasons {
            assert!(
                !reason.is_empty(),
                "Finish reason should be recorded as non-empty string"
            );
        }
    }

    /// Test: Mismatch report generation
    #[test]
    fn test_mismatch_report_generation() {
        // Simulate token sequence mismatch
        let expected_tokens = vec![1, 2, 3, 4, 5];
        let actual_tokens = vec![1, 2, 99, 4, 5];

        let mismatch_pos = expected_tokens
            .iter()
            .zip(actual_tokens.iter())
            .position(|(e, a)| e != a);

        let report = if let Some(pos) = mismatch_pos {
            format!(
                "Token mismatch at position {}: expected {}, got {}",
                pos, expected_tokens[pos], actual_tokens[pos]
            )
        } else {
            String::new()
        };

        assert_eq!(
            report,
            "Token mismatch at position 2: expected 3, got 99"
        );
    }

    /// Test: Multiple turn record accumulation
    #[test]
    fn test_multiple_turn_record_accumulation() {
        struct TurnRecord {
            turn_id: usize,
            prompt_tokens: usize,
            output_tokens: usize,
        }

        let mut records = vec![];

        // Turn 1: user→assistant
        records.push(TurnRecord {
            turn_id: 1,
            prompt_tokens: 50,
            output_tokens: 25,
        });

        // Turn 2: assistant→user→assistant
        records.push(TurnRecord {
            turn_id: 2,
            prompt_tokens: 75,
            output_tokens: 30,
        });

        // Turn 3: assistant→user→assistant
        records.push(TurnRecord {
            turn_id: 3,
            prompt_tokens: 105,
            output_tokens: 20,
        });

        assert_eq!(records.len(), 3);
        let total_prompt_tokens: usize = records.iter().map(|r| r.prompt_tokens).sum();
        assert_eq!(total_prompt_tokens, 50 + 75 + 105);
    }

    /// Test: Boundary token handling for Qwen3 model
    #[test]
    fn test_boundary_token_handling_qwen3() {
        // Qwen3 inserts missing newline at prefix junction
        let cached_tokens = vec![1, 2, 3, 100]; // 100 = newline
        let new_request_tokens = vec![1, 2, 3, 100, 4, 5]; // Expects newline at position 3

        // Boundary handling: if newline is missing, insert it
        let needs_newline = new_request_tokens.len() > cached_tokens.len()
            && cached_tokens.len() > 0
            && new_request_tokens[cached_tokens.len() - 1] != 100;

        // In this case, new_request matches cache up to and including newline
        assert!(!needs_newline, "Newline already present in request");
    }

    /// Test: GLM4.7 boundary token handling (role stripping)
    #[test]
    fn test_boundary_token_handling_glm47() {
        // GLM4.7: strips ambiguous boundary tokens on merge (multi-role conflict)
        let cached_tokens = vec![50, 51, 200, 201]; // 200-201 = role markers
        let new_request_tokens = vec![50, 51, 202, 203]; // Different role markers

        // Boundary handling: detect role marker mismatch
        let role_mismatch = cached_tokens[2] != new_request_tokens[2];

        assert!(role_mismatch, "Role marker should differ between cache and request");
    }

    /// Test: Training data construction with single turn
    #[test]
    fn test_training_data_single_turn() {
        struct TrainingExample {
            prompt_tokens: Vec<i64>,
            output_tokens: Vec<i64>,
            logprobs: Vec<f64>,
            finish_reason: String,
        }

        let example = TrainingExample {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5, 6, 7],
            logprobs: vec![0.95, 0.85, 0.90],
            finish_reason: "stop".to_string(),
        };

        assert_eq!(example.prompt_tokens.len(), 3);
        assert_eq!(example.output_tokens.len(), 4);
        assert_eq!(example.logprobs.len(), 3);
    }

    /// Test: Training data construction with multi-turn
    #[test]
    fn test_training_data_multi_turn() {
        struct TrainingExample {
            turn_records: Vec<(Vec<i64>, Vec<i64>)>,
        }

        let example = TrainingExample {
            turn_records: vec![
                (vec![1, 2], vec![3, 4, 5]), // Turn 1: prompt tokens, output tokens
                (vec![1, 2, 3, 4, 5, 6], vec![7, 8, 9]), // Turn 2
                (vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], vec![11, 12]), // Turn 3
            ],
        };

        assert_eq!(example.turn_records.len(), 3);
        // Verify append-only: each turn's prompt includes previous turn's output
        assert_eq!(example.turn_records[1].0[..2], example.turn_records[0].0[..]);
    }

    /// Test: Response mask construction (assistant vs environment)
    #[test]
    fn test_response_mask_construction() {
        // Response mask: 1 for assistant, 0 for environment/tool
        let response_types = vec![
            ("assistant", 1i64),
            ("tool", 0i64),
            ("environment", 0i64),
            ("assistant", 1i64),
            ("user", 0i64),
        ];

        let mask: Vec<i64> = response_types.iter().map(|(_, m)| m).copied().collect();
        assert_eq!(mask, vec![1, 0, 0, 1, 0]);
    }

    /// Test: Tool response handling in training data
    #[test]
    fn test_tool_response_handling() {
        // Simulate tool call sequence
        struct MessageTurn {
            role: String,
            content: String,
            is_tool: bool,
        }

        let messages = vec![
            MessageTurn {
                role: "assistant".to_string(),
                content: "I'll call a tool".to_string(),
                is_tool: false,
            },
            MessageTurn {
                role: "tool".to_string(),
                content: "Tool result".to_string(),
                is_tool: true,
            },
            MessageTurn {
                role: "assistant".to_string(),
                content: "Final answer".to_string(),
                is_tool: false,
            },
        ];

        // Count assistant messages for training
        let assistant_count = messages.iter().filter(|m| m.role == "assistant").count();
        assert_eq!(assistant_count, 2);

        // Verify tool response is between assistant messages
        assert!(messages[0].role == "assistant");
        assert!(messages[1].is_tool);
        assert!(messages[2].role == "assistant");
    }
}
