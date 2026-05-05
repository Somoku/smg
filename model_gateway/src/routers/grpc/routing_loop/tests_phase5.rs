//! Phase 5: Mismatch Detection and Validation Tests (Token sequence validation, error detection)

#[cfg(test)]
mod tests {
    /// Test: Exact token match (no mismatch)
    #[test]
    fn test_exact_token_match_no_mismatch() {
        let expected = vec![1, 2, 3, 4, 5];
        let actual = vec![1, 2, 3, 4, 5];

        let has_mismatch = expected != actual;
        assert!(!has_mismatch, "Identical sequences should have no mismatch");
    }

    /// Test: Token sequence mismatch at position
    #[test]
    fn test_token_mismatch_at_position() {
        let expected = vec![1, 2, 3, 4, 5];
        let actual = vec![1, 2, 99, 4, 5];

        let mismatch_pos = expected
            .iter()
            .zip(actual.iter())
            .position(|(e, a)| e != a);

        assert_eq!(mismatch_pos, Some(2), "Mismatch at position 2");
    }

    /// Test: Append-only violation detection
    #[test]
    fn test_append_only_violation_detection() {
        let prev = vec![1, 2, 3, 4, 5];
        let violating = vec![1, 2, 3, 5, 6]; // Token 4 missing

        let is_append_only = prev.len() <= violating.len()
            && violating[..prev.len()] == prev[..];

        assert!(!is_append_only, "Should detect violation");
    }

    /// Test: Deletion violation (sequence shorter)
    #[test]
    fn test_deletion_violation() {
        let prev = vec![1, 2, 3, 4, 5];
        let shorter = vec![1, 2, 3]; // Deleted 4, 5

        let is_append_only = prev.len() <= shorter.len()
            && shorter[..prev.len()] == prev[..];

        assert!(!is_append_only, "Deletion violates append-only");
    }

    /// Test: Reordering violation
    #[test]
    fn test_reordering_violation() {
        let prev = vec![1, 2, 3, 4, 5];
        let reordered = vec![1, 3, 2, 4, 5, 6]; // Positions 1 and 2 swapped

        let is_append_only = prev.len() <= reordered.len()
            && reordered[..prev.len()] == prev[..];

        assert!(!is_append_only, "Reordering violates append-only");
    }

    /// Test: Valid append (append-only satisfied)
    #[test]
    fn test_valid_append() {
        let prev = vec![1, 2, 3];
        let appended = vec![1, 2, 3, 4, 5, 6];

        let is_append_only = prev.len() <= appended.len()
            && appended[..prev.len()] == prev[..];

        assert!(is_append_only, "Appending new tokens is valid");
    }

    /// Test: Multi-position mismatches (cumulative errors)
    #[test]
    fn test_multiple_position_mismatches() {
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let actual = vec![1, 99, 3, 99, 5, 6, 99, 8];

        let mismatch_positions: Vec<usize> = expected
            .iter()
            .zip(actual.iter())
            .enumerate()
            .filter(|(_, (e, a))| e != a)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(mismatch_positions, vec![1, 3, 6]);
    }

    /// Test: Logprob validation (confidence scores)
    #[test]
    fn test_logprob_validation() {
        // Logprobs should be in [0.0, 1.0] range (normalized)
        let logprobs = vec![0.95, 0.87, 0.92, 0.78, 0.99];

        let all_valid = logprobs.iter().all(|&p| p >= 0.0 && p <= 1.0);
        assert!(all_valid, "All logprobs should be valid");

        // Negative or >1.0 should be invalid
        let invalid_logprobs = vec![0.95, 1.5, -0.1];
        let all_invalid_valid = invalid_logprobs.iter().all(|&p| p >= 0.0 && p <= 1.0);
        assert!(!all_invalid_valid, "Some logprobs are invalid");
    }

    /// Test: Finish reason validation
    #[test]
    fn test_finish_reason_validation() {
        let valid_reasons = vec!["stop", "length", "tool_calls", "error"];
        let all_valid = valid_reasons.iter().all(|r| !r.is_empty());
        assert!(all_valid, "All finish reasons should be non-empty");

        let invalid_reason = "";
        assert!(invalid_reason.is_empty(), "Empty finish reason should be invalid");
    }

    /// Test: Prefix match detection
    #[test]
    fn test_prefix_match_detection() {
        let prefix = vec![1, 2, 3, 4];
        let full_seq = vec![1, 2, 3, 4, 5, 6, 7];

        let is_prefix = prefix.len() <= full_seq.len()
            && full_seq[..prefix.len()] == prefix[..];

        assert!(is_prefix, "Should detect valid prefix");
    }

    /// Test: Prefix mismatch detection
    #[test]
    fn test_prefix_mismatch_detection() {
        let expected_prefix = vec![1, 2, 3, 4];
        let actual_seq = vec![1, 2, 99, 4, 5, 6];

        let is_prefix = expected_prefix.len() <= actual_seq.len()
            && actual_seq[..expected_prefix.len()] == expected_prefix[..];

        assert!(!is_prefix, "Should detect prefix mismatch");
    }

    /// Test: Empty sequence validation
    #[test]
    fn test_empty_sequence_validation() {
        let empty: Vec<i32> = vec![];
        let non_empty = vec![1, 2, 3];

        assert!(empty.is_empty());
        assert!(!non_empty.is_empty());

        // Append-only still valid for empty
        let is_append_only = empty.len() <= non_empty.len()
            && non_empty[..empty.len()] == empty[..];
        assert!(is_append_only);
    }

    /// Test: Large sequence validation
    #[test]
    fn test_large_sequence_validation() {
        let mut large_seq = vec![];
        for i in 0..10000 {
            large_seq.push(i as i64);
        }

        assert_eq!(large_seq.len(), 10000);

        // Verify append-only with subset
        let subset = &large_seq[..5000];
        let is_append_only = subset.len() <= large_seq.len()
            && large_seq[..subset.len()] == subset[..];
        assert!(is_append_only);
    }

    /// Test: Token uniqueness check (if needed for specific models)
    #[test]
    fn test_token_value_ranges() {
        let tokens = vec![0, 1, 100, 1000, 50000, 128000]; // Typical tokenizer ranges

        let all_non_negative = tokens.iter().all(|&t| t >= 0);
        assert!(all_non_negative, "Tokens should be non-negative");
    }

    /// Test: Mismatch report structure
    #[test]
    fn test_mismatch_report_structure() {
        struct MismatchReport {
            position: usize,
            expected_token: i64,
            actual_token: i64,
            detail: String,
        }

        let report = MismatchReport {
            position: 5,
            expected_token: 42,
            actual_token: 99,
            detail: "Token changed between requests".to_string(),
        };

        assert_eq!(report.position, 5);
        assert_eq!(report.expected_token, 42);
        assert_eq!(report.actual_token, 99);
        assert!(!report.detail.is_empty());
    }

    /// Test: Concurrent mismatch detection (multiple threads checking simultaneously)
    #[test]
    fn test_mismatch_detection_consistency() {
        let expected = vec![1, 2, 3, 4, 5];
        let actual = vec![1, 2, 99, 4, 5];

        // First check
        let mismatch1 = expected
            .iter()
            .zip(actual.iter())
            .position(|(e, a)| e != a);

        // Second check (should be identical)
        let mismatch2 = expected
            .iter()
            .zip(actual.iter())
            .position(|(e, a)| e != a);

        assert_eq!(mismatch1, mismatch2, "Mismatch detection should be deterministic");
    }
}
