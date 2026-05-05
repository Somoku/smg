//! Building blocks for the request routing loop.
//!
//! The routing loop itself is wired in later phases.  This module keeps the
//! queue and metadata parsing isolated so the hot-path data structures can be
//! tested independently from request dispatch.

pub(crate) mod controller;
pub(crate) mod metadata;
pub(crate) mod queue;
pub(crate) mod runtime;

#[cfg(test)]
mod tests_phase1;

#[cfg(test)]
mod tests_phase2;

#[cfg(test)]
mod tests_phase3;

#[cfg(test)]
mod tests_phase4;

#[cfg(test)]
mod tests_phase5;

#[cfg(test)]
mod tests_phase6;
