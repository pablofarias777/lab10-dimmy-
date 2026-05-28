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


def build_fake_medical_context(repeat_count: int = 900) -> str:
    base_chunk = (
        "Patient history: recurrent hypertension, type 2 diabetes, mild renal insufficiency. "
        "Medication plan includes metformin, losartan, and dietary sodium restriction. "
        "Clinical notes report progressive fatigue, nocturnal dyspnea, and lower limb edema. "
        "Differential diagnosis considers chronic heart failure, anemia of chronic disease, and infection markers. "
        "Recommended exams: complete blood count, BNP, creatinine, urinalysis, chest imaging, and ECG follow-up. "
        "Treatment guidance emphasizes medication adherence, hydration balance, and periodic risk stratification. "
    )
    return base_chunk * repeat_count


def prepare_inputs(tokenizer, context_text: str, model) -> dict:
    encoded = tokenizer(context_text, return_tensors="pt")
    first_param = next(model.parameters())
    device = first_param.device
    return {k: v.to(device) for k, v in encoded.items()}


def benchmark_generation(model, tokenizer, context_text: str, max_new_tokens: int, use_cache: bool):
    model.config.use_cache = use_cache
    model.eval()

    if not torch.cuda.is_available():
        raise RuntimeError("Passo de benchmark exige GPU CUDA para medir VRAM.")

    inputs = prepare_inputs(tokenizer, context_text, model)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    peak_vram_bytes = torch.cuda.max_memory_allocated()
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return elapsed, peak_vram_bytes, generated_text


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

    _ = (tokenizer, model)


def run_step_2(model_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rag_context = build_fake_medical_context()
    encoded = tokenizer(rag_context, return_tensors="pt")
    token_count = encoded["input_ids"].shape[1]

    print("=== Passo 2: Simulando RAG Massivo ===")
    print(f"Modelo/tokenizer: {model_id}")
    print(f"Tamanho do contexto em caracteres: {len(rag_context)}")
    print(f"Total de tokens gerados: {token_count}")
    print("Faixa esperada: 10.000 a 15.000 tokens")


def run_step_3(model_id: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Passo 3 exige GPU CUDA para medir VRAM e tempo.")

    tokenizer, model = load_quantized_model(model_id)
    rag_context = build_fake_medical_context()
    token_count = tokenizer(rag_context, return_tensors="pt")["input_ids"].shape[1]

    elapsed, peak_vram_bytes, generated_text = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        context_text=rag_context,
        max_new_tokens=100,
        use_cache=False,
    )

    print("=== Passo 3: Gargalo de Geracao (Sem KV Cache) ===")
    print(f"Modelo: {model_id}")
    print(f"Tokens no contexto de entrada: {token_count}")
    print("use_cache: False")
    print(f"Tempo total para gerar 100 tokens: {elapsed:.2f}s")
    print(f"Pico de VRAM na geracao: {format_mb(peak_vram_bytes):.2f} MB")
    print(f"Tamanho do texto de saida (chars): {len(generated_text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Laboratorio 10 - Pipeline RAG + QLoRA")
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Passo a executar no momento.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Modelo base auto-regressivo.",
    )
    args = parser.parse_args()

    if args.step == 1:
        run_step_1(args.model_id)
    elif args.step == 2:
        run_step_2(args.model_id)
    elif args.step == 3:
        run_step_3(args.model_id)


if __name__ == "__main__":
    main()
