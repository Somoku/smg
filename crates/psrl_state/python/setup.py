"""Build-time protobuf generation for psrl-state-grpc-proto."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


def compile_grpc_protos() -> None:
    package_dir = Path(__file__).parent
    proto_dir = package_dir / "psrl_state_grpc_proto" / "proto"
    output_dir = package_dir / "psrl_state_grpc_proto" / "generated"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "__init__.py").write_text(
        '"""Auto-generated protobuf stubs. Do not edit."""\n',
        encoding="utf-8",
    )

    proto_files = list(proto_dir.glob("*.proto"))
    if not proto_files:
        raise FileNotFoundError(f"No .proto files found in {proto_dir}")

    try:
        import grpc_tools
        from grpc_tools import protoc

        well_known = Path(grpc_tools.__file__).parent / "_proto"
        args = [
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--proto_path={well_known}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            f"--pyi_out={output_dir}",
        ] + [str(p) for p in proto_files]

        print(f"Generating protobuf stubs from {len(proto_files)} proto files...")
        result = protoc.main(args)
        if result != 0:
            raise RuntimeError(f"protoc returned non-zero exit code: {result}")
    except ImportError as exc:
        raise RuntimeError(
            "grpcio-tools not installed. Install with: pip install grpcio-tools"
        ) from exc

    mypy_header = "# mypy: ignore-errors\n"
    for py_file in output_dir.glob("*_pb2*.py"):
        content = py_file.read_text(encoding="utf-8")
        for proto_file in proto_files:
            module_name = proto_file.stem + "_pb2"
            content = content.replace(
                f"import {module_name}",
                f"from . import {module_name}",
            )
        if not content.startswith("# mypy:"):
            content = mypy_header + content
        py_file.write_text(content, encoding="utf-8")

    for pyi_file in output_dir.glob("*_pb2*.pyi"):
        content = pyi_file.read_text(encoding="utf-8")
        if not content.startswith("# mypy:"):
            pyi_file.write_text(mypy_header + content, encoding="utf-8")

    generated_count = len(list(output_dir.glob("*.py"))) + len(list(output_dir.glob("*.pyi")))
    print(f"Generated {generated_count} files (including type stubs)")


class BuildPyWithProto(build_py):
    def run(self):
        compile_grpc_protos()
        super().run()


class DevelopWithProto(develop):
    def run(self):
        compile_grpc_protos()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithProto,
        "develop": DevelopWithProto,
    }
)
