# psrl-state-grpc-proto

Python gRPC stubs for the PSRL PSManager state service.

This package lives at `smg/crates/psrl_state/python/` and mirrors
`psrl_state/python/` in the PSRL workspace. The proto file is at
`smg/crates/psrl_state/proto/psrl_manager.proto`.

## Installation

```bash
pip install grpcio-tools   # build-time dependency
pip install -e smg/crates/psrl_state/python
```

## Usage

```python
from psrl_state_grpc_proto import psrl_manager_pb2, psrl_manager_pb2_grpc
import grpc

channel = grpc.insecure_channel("localhost:50051")
stub = psrl_manager_pb2_grpc.PSManagerStateStub(channel)
```
