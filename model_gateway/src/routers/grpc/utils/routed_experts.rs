//! Encoding helpers for routed-experts payloads on the response boundary.
//!
//! The gateway-side `ProtoRoutedExperts` carries a C-contiguous byte buffer
//! whose row stride is `num_layers * top_k * dtype.size()`.  Clients receive
//! the data as a base64-encoded NumPy `.npy` v1.0 file so existing trainer
//! decoders (built around `np.load`) work unchanged.

use base64::Engine as _;
use openai_protocol::npy::encode_npy;

use crate::routers::grpc::proto_wrapper::ProtoRoutedExperts;

/// Encode a routed-experts tensor as a base64-encoded NumPy `.npy` v1.0
/// file, ready to embed in a response payload.
///
/// Returns `None` if the tensor shape would yield a zero-row matrix
pub(crate) fn encode_routed_experts_for_response(re: &ProtoRoutedExperts) -> Option<String> {
    let rows = re.num_tokens();
    if rows == 0 {
        return None;
    }
    let shape = [rows as u64, u64::from(re.num_layers), u64::from(re.top_k)];
    let npy = encode_npy(&shape, re.dtype.as_npy(), &re.data);
    Some(base64::engine::general_purpose::STANDARD.encode(npy))
}
