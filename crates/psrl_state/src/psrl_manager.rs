use std::time::Duration;

use tonic::{transport::Channel, Request};

// PR 0: psrl_state crate migration (plan.md §0.4 — clippy compliance)
pub mod proto {
    #![allow(
        clippy::absolute_paths,
        clippy::doc_lazy_continuation,
        clippy::derive_partial_eq_without_eq,
        clippy::allow_attributes,
        unused_qualifications
    )]
    tonic::include_proto!("psrl.state");
}

/// gRPC client wrapper for PS Manager state operations.
#[derive(Clone)]
pub struct PSManagerStateClient {
	client: proto::ps_manager_state_client::PsManagerStateClient<Channel>,
}

impl PSManagerStateClient {
	pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
		let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
			format!("http://{addr}")
		} else if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
			endpoint.to_string()
		} else {
			format!("http://{endpoint}")
		};

		let channel = Channel::from_shared(http_endpoint)?
			.http2_keep_alive_interval(Duration::from_secs(30))
			.keep_alive_timeout(Duration::from_secs(10))
			.keep_alive_while_idle(true)
			.tcp_keepalive(Some(Duration::from_secs(60)))
			.tcp_nodelay(true)
			.connect()
			.await?;

		Ok(Self {
			client: proto::ps_manager_state_client::PsManagerStateClient::new(channel),
		})
	}

	pub async fn can_reserve_request(
		&self,
		request_ids: Vec<i64>,
		model_versions: Vec<i64>,
		without_new_reserve_entry: bool,
		is_validate: Vec<bool>,
	) -> Result<(Vec<bool>, i32), tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.can_reserve_request(Request::new(proto::CanReserveRequestReq {
				request_ids,
				model_versions,
				without_new_reserve_entry,
				is_validate,
			}))
			.await?
			.into_inner();
		Ok((resp.results, resp.n_versions))
	}

	pub async fn get_reserve_indicator(
		&self,
		request_id: i64,
		model_versions: Vec<i64>,
		is_validate: bool,
	) -> Result<Vec<f64>, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.get_reserve_indicator(Request::new(proto::GetReserveIndicatorReq {
				request_id,
				model_versions,
				is_validate,
			}))
			.await?
			.into_inner();
		Ok(resp.indicators)
	}

	pub async fn reserve_rollout_instance_requests(
		&self,
		rollout_instance_ids: Vec<(String, usize)>,
		request_ids: Vec<i64>,
		model_versions: Vec<i64>,
		guarantee_not_aborted: bool,
		is_validate: bool,
	) -> Result<(bool, Vec<i64>, Vec<i64>, String), tonic::Status> {
		let mut client = self.client.clone();
		let rollout_instance_ids = rollout_instance_ids
			.into_iter()
			.map(|(worker_id, dp_rank)| proto::RolloutInstanceId {
				worker_id,
				dp_rank: dp_rank as i64,
			})
			.collect();

		let resp = client
			.reserve_rollout_instance_requests(Request::new(
				proto::ReserveRolloutInstanceRequestsReq {
					rollout_instance_ids,
					request_ids,
					model_versions,
					guarantee_not_aborted,
					is_validate,
				},
			))
			.await?
			.into_inner();

		Ok((
			resp.success,
			resp.buffer_ids,
			resp.entry_ids,
			resp.error_message,
		))
	}

	pub async fn update_request_instance_id(
		&self,
		request_id: i64,
		new_instance_id: (String, usize),
		is_validate: bool,
	) -> Result<bool, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.update_request_instance_id(Request::new(proto::UpdateRequestInstanceIdReq {
				request_id,
				new_instance_id: Some(proto::RolloutInstanceId {
					worker_id: new_instance_id.0,
					dp_rank: new_instance_id.1 as i64,
				}),
				is_validate,
			}))
			.await?
			.into_inner();
		Ok(resp.success)
	}

	pub async fn update_request_version_tag(
		&self,
		request_id: i64,
		new_version_tag: i64,
		is_validate: bool,
	) -> Result<bool, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.update_request_version_tag(Request::new(proto::UpdateRequestVersionTagReq {
				request_id,
				new_version_tag,
				is_validate,
			}))
			.await?
			.into_inner();
		Ok(resp.success)
	}

	pub async fn check_aborted_requests(
		&self,
		request_ids: Vec<i64>,
		remove: bool,
	) -> Result<Vec<bool>, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.check_aborted_requests(Request::new(proto::CheckAbortedRequestsReq {
				request_ids,
				remove,
			}))
			.await?
			.into_inner();
		Ok(resp.is_aborted)
	}

	pub async fn check_aborted_model_versions(
		&self,
		model_versions: Vec<i64>,
	) -> Result<Vec<bool>, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.check_aborted_model_versions(Request::new(proto::CheckAbortedModelVersionsReq {
				model_versions,
			}))
			.await?
			.into_inner();
		Ok(resp.is_aborted)
	}

	pub async fn get_rollout_instance_model_version(
		&self,
		rollout_instance_id: (String, usize),
	) -> Result<i64, tonic::Status> {
		let mut client = self.client.clone();
		let resp = client
			.get_rollout_instance_model_version(Request::new(
				proto::GetRolloutInstanceModelVersionReq {
					rollout_instance_id: Some(proto::RolloutInstanceId {
						worker_id: rollout_instance_id.0,
						dp_rank: rollout_instance_id.1 as i64,
					}),
				},
			))
			.await?
			.into_inner();
		Ok(resp.model_version)
	}

	pub async fn update_request_status(
		&self,
		request_ids: Vec<i64>,
		status: String,
		model_versions: Vec<i64>,
		rollout_instance_ids: Vec<(String, usize)>,
		is_validate: bool,
	) -> Result<Vec<bool>, tonic::Status> {
		let mut client = self.client.clone();
		let rollout_instance_ids = rollout_instance_ids
			.into_iter()
			.map(|(worker_id, dp_rank)| proto::RolloutInstanceId {
				worker_id,
				dp_rank: dp_rank as i64,
			})
			.collect();

		let resp = client
			.update_request_status(Request::new(proto::UpdateRequestStatusReq {
				request_ids,
				status,
				model_versions,
				rollout_instance_ids,
				is_validate,
			}))
			.await?
			.into_inner();
		Ok(resp.succeeded)
	}

}

