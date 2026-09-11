#!/usr/bin/env python3
"""
Dynamic INT8 quantization utility for Edge ORT mobile models.
Converts standard FP32 ONNX models (Silero VAD, Whisper encoder/decoder, OPUS-MT)
into compact INT8 models optimized for mobile NPU (QNN/NNAPI) and CPU (XNNPACK).

Usage:
    python3 scripts/quantize_mobile_models.py --input models/silero_vad.onnx --output models/silero_vad_int8.onnx
"""

import argparse
import os
import sys

def quantize_onnx_model(input_path: str, output_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("Error: onnxruntime is required for model quantization.", file=sys.stderr)
        print("Install it with: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input model not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Quantizing: {input_path}")
    print(f"[*] Target:     {output_path}")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )

    in_size = os.path.getsize(input_path) / (1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024)
    ratio = (1.0 - (out_size / in_size)) * 100.0

    print(f"[+] Quantization Complete!")
    print(f"    Original Size:   {in_size:.2f} MB")
    print(f"    Quantized Size:  {out_size:.2f} MB")
    print(f"    Size Reduction:  {ratio:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8 for mobile deployment.")
    parser.add_argument("--input", "-i", required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--output", "-o", required=True, help="Path to output INT8 ONNX model")
    args = parser.parse_args()

    quantize_onnx_model(args.input, args.output)

if __name__ == "__main__":
    main()
