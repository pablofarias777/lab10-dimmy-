import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_mb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)


def load_quantized_model(model_id: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
    )

    return tokenizer, model


def run_step_1(model_id: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Passo 1 exige GPU CUDA para medir VRAM.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    tokenizer, model = load_quantized_model(model_id)
    elapsed = time.perf_counter() - start

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = format_mb(peak_vram_bytes)

    print("=== Passo 1: Ingestao Eficiente (QLoRA 4-bit) ===")
    print(f"Modelo: {model_id}")
    print(f"Device: {get_device()}")
    print(f"Tempo de carregamento: {elapsed:.2f}s")
    print(f"Pico de VRAM na carga: {peak_vram_mb:.2f} MB")

    # Prevent accidental optimization by Python removing references.
    _ = (tokenizer, model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio 10 - Pipeline RAG + QLoRA")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        choices=[1],
        help="Passo a executar no momento.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Modelo base auto-regressivo para o Passo 1.",
    )
    args = parser.parse_args()

    if args.step == 1:
        run_step_1(args.model_id)


if __name__ == "__main__":
    main()
