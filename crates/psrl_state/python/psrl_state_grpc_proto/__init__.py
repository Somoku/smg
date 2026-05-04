"""PSRL state gRPC proto package."""

__version__ = "0.1.0"

try:
    from psrl_state_grpc_proto.generated import psrl_manager_pb2, psrl_manager_pb2_grpc

    __all__ = ["psrl_manager_pb2", "psrl_manager_pb2_grpc"]
except ImportError:
    __all__ = []
