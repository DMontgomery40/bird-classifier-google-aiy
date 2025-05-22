#!/usr/bin/env python3
"""Convert the 400-species EfficientNet bird model to OpenVINO IR format."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert EfficientNet bird model to OpenVINO")
    parser.add_argument("input_model", help="Path to the downloaded model (ONNX or PyTorch)")
    parser.add_argument("--output_dir", default="openvino_model", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mo",
        "--input_model", str(args.input_model),
        "--output_dir", str(output_dir),
        "--model_name", "efficientnet_birds_400"
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
