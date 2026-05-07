pub mod engine;
pub mod error;
pub mod model_adapter;
pub mod normalizer;
pub mod store;
pub mod validator;

pub use error::TitoError;
pub use normalizer::{hash_messages, hash_messages_with_context, RenderContext};
pub use store::{MismatchEntry, PrefixMatch, Trajectory, TitoStore, TurnRecord};

/// HTTP header name for the TITO session identifier.
pub const TITO_SESSION_HEADER: &str = "x-smg-tito-session-id";

/// HTTP header name for the TITO trajectory identifier.
///
/// Defaults to 0 when absent.
pub const TITO_TRAJECTORY_ID_HEADER: &str = "x-smg-tito-trajectory-id";
