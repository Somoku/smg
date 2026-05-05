//! Building blocks for the request routing loop.
//!
//! The routing loop itself is wired in later phases.  This module keeps the
//! queue and metadata parsing isolated so the hot-path data structures can be
//! tested independently from request dispatch.

pub(crate) mod controller;
pub(crate) mod metadata;
pub(crate) mod partial_rollout;
pub(crate) mod queue;
pub(crate) mod runtime;
