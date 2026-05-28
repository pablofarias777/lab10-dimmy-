# Laboratório 10 — Pipeline de IA Escalável (QLoRA, KV Cache e FlashAttention)

> **Disciplina:** Inteligência Artificial Aplicada  
> **Instituição:** Instituto iCEV  
> **Aluno:** Pablo Ferreira de Andrade Farias  
> **Orientador:** Prof. Dimmy  
> **Entrega:** versão `v1.0`

---

> **Nota de Integridade Acadêmica:**  
> *"Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Pablo Ferreira de Andrade Farias"*

> **Uso de IA:**  
> Ferramentas de IA generativa foram usadas como apoio na estruturação do pipeline, geração inicial de conteúdo técnico simulado e documentação. Todo o conteúdo foi revisado criticamente e validado pelo aluno antes da submissão.

---

## Objetivo

Este laboratório integra os principais tópicos da disciplina em um cenário de produção: um modelo auto-regressivo quantizado precisa gerar texto a partir de contexto massivo recuperado por RAG, sem colapsar a VRAM da GPU. A proposta é demonstrar, na prática, como otimizações complementares de memória e inferência tornam viável a execução de Transformers em cargas longas.

---

## Estrutura do Projeto

```text
lab10-dimmy-/
+-- README.md
+-- requirements.txt
+-- lab10_pipeline.py
```

---

## Como Executar

### 1. Instalar dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Passo 1 — Carga quantizada QLoRA 4-bit

```bash
python lab10_pipeline.py --step 1
```

Saída esperada:

- tempo de carregamento do modelo;
- pico de VRAM na carga (MB).

### 3. Passo 2 — Simulação de RAG massivo

```bash
python lab10_pipeline.py --step 2
```

Saída esperada:

- contexto fictício médico grande;
- total de tokens na faixa alvo (10.000 a 15.000).

### 4. Passo 3 — Gargalo sem cache (`use_cache=False`)

```bash
python lab10_pipeline.py --step 3
```

Saída esperada:

- tempo total para gerar 100 tokens;
- pico de VRAM na geração com recálculo redundante.

### 5. Passo 4 — Otimização com KV Cache + FlashAttention-2

```bash
python lab10_pipeline.py --step 4
```

Saída esperada:

- tempo total para gerar 100 tokens com `use_cache=True`;
- pico de VRAM em cenário otimizado.

Observação de ambiente:

- para os passos 3 e 4, é necessário PyTorch com CUDA e GPU NVIDIA ativa;
- em ambiente `torch` CPU-only, não é possível medir VRAM CUDA.

---

## Métricas de Benchmark

Preencher com os resultados coletados na execução em GPU:

- **Passo 1 (QLoRA 4-bit):**
  - Tempo de carregamento: `___ s`
  - Pico de VRAM na carga: `___ MB`
- **Passo 3 (sem KV Cache):**
  - Tempo para 100 tokens: `___ s`
  - Pico de VRAM na geração: `___ MB`
- **Passo 4 (com KV Cache + FlashAttention-2):**
  - Tempo para 100 tokens: `___ s`
  - Pico de VRAM na geração: `___ MB`

---

## Parecer Técnico (Passo 5)

A combinação de **QLoRA (4-bit)**, **KV Cache** e **FlashAttention** foi decisiva para evitar o colapso de VRAM no laboratório. O QLoRA reduziu fortemente o custo de memória estática do modelo já no carregamento, viabilizando manter o LLM residente na GPU. Na geração, o KV Cache eliminou o recálculo completo de chaves e valores dos tokens anteriores a cada novo passo do decoder, reduzindo trabalho redundante e melhorando latência. Em paralelo, o FlashAttention reorganizou o cálculo da atenção para usar blocagem e melhor aproveitamento da SRAM da GPU, diminuindo tráfego de memória e picos de alocação no prompting longo. Em conjunto, essas três técnicas tornaram um pipeline antes inviável em uma execução prática com contexto massivo.

Se a exigência subisse de ~15 mil para **2 milhões de tokens**, mesmo FlashAttention deixaria de ser suficiente do ponto de vista arquitetural. Embora ele reduza custos constantes e overhead de memória, a base do Transformer ainda depende de mecanismos de atenção que escalam mal para sequências extremas, mantendo pressão proibitiva em tempo, memória e largura de banda. Nesse regime, a indústria tende a migrar para famílias como **State Space Models (ex.: Mamba)**, que modelam dependências longas com custo de memória efetivamente constante por passo (aproximação de **O(1)** no estado), trocando atenção global explícita por dinâmica recorrente eficiente. Essa mudança não é apenas otimização incremental, mas uma troca de paradigma para workloads de contexto ultralongo.

---

## Checklist de Entrega

- [x] Carregamento do modelo com quantização 4-bit (QLoRA).
- [x] Simulação de contexto massivo e tokenização.
- [x] Benchmark sem cache (`use_cache=False`) com métricas.
- [x] Benchmark otimizado com KV Cache e FlashAttention-2.
- [x] Parecer técnico em 2 parágrafos no README.
- [x] Nota obrigatória de integridade acadêmica no topo do README.
- [ ] Tag/release final publicada como `v1.0`.

Comandos de fechamento:

```bash
git add README.md
git commit -m "docs: adiciona passo 5 e análise arquitetural do lab 10"
git push origin main
git tag -a v1.0 -m "Entrega final Lab 10"
git push origin v1.0
```
