//! Shared code for both regular and harmony routers

pub(crate) mod response_collection;
pub(crate) mod response_formatting;
pub(crate) mod responses;
// PR 5A §5A.1a-d: Routing loop controller endpoints (pause/resume/status/filter).
pub(crate) mod routing_loop_controller;
pub(crate) mod stages;
