# Reprodução e Extensão da RQ3 do Ambig-SWE: Família Qwen2.5 Coder

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21211716.svg)](https://doi.org/10.5281/zenodo.21211716)

Reprodução parcial e extensão da **RQ3 (Qualidade das Perguntas)** do paper
*Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software
Engineering* (ICLR 2026), aplicada à família open-weight **Qwen2.5 Coder**
(1.5B, 7B, 14B e 32B) sob dois simuladores de usuário (GPT-4o-mini e
Gemini 3.5 Flash).

O experimento são **240 execuções**: 30 instâncias × 4 escalas × 2 simuladores.
As métricas são `avg_q` (perguntas por sessão), IG (distância de cosseno via
`text-embedding-3-small`) e LLM-as-a-Judge (GPT-4o, escala 1 a 5).

Os logs brutos das 240 execuções (trajetórias completas do OpenHands) estão
arquivados em Zenodo: [10.5281/zenodo.21211716](https://doi.org/10.5281/zenodo.21211716).

## Estrutura

```
reproducao/
├── experiments/
│   ├── extracted_qa_pairs/    Pares pergunta-resposta extraídos -> avg_q
│   ├── cosine_distance/       Embeddings e distâncias -> IG
│   ├── llm_as_judge/          Avaliações do juiz + experimento de swap -> Judge
│   ├── statistical_analysis/  Testes de hipótese e análise de sensibilidade
│   └── figures/               Scripts de geração das figuras
├── scripts/
│   └── generate_sample.py     Amostragem das 30 instâncias (seed 42)
├── baterias/                  Scripts de execução por escala (1_5b, 7b, 14b, 32b)
├── verify_all_numbers.py      Verifica os números reportados contra os dados brutos
└── requirements.txt

../validation/                 Suíte independente de validação estatística
../data/sample_30_underspecified.csv   Amostra utilizada
```

## Pré-requisitos

```bash
pip install -r requirements.txt
```

## Como reproduzir

1. **Amostragem.** `python3 scripts/generate_sample.py` seleciona 30 das 500
   issues de `data/underspecified.csv` com semente 42.
2. **Execução.** Os scripts em `baterias/<escala>/` rodam as sessões de cada
   configuração. Os modelos 7B/14B rodam localmente em MLX 4-bit, o 1.5B local
   em 16-bit, e o 32B via API (OpenRouter). As chaves de API são lidas de
   variáveis de ambiente.
3. **Métricas.** Os dados processados ficam em `experiments/`:
   `extracted_qa_pairs/` (avg_q), `cosine_distance/` (IG) e `llm_as_judge/`
   (Judge e experimento de swap).
4. **Verificação dos números.** Confere cada valor reportado contra os dados
   brutos:
   ```bash
   python3 verify_all_numbers.py
   ```
   Saída esperada: status `OK` em todos os checks (com tolerância de
   arredondamento no último dígito).
5. **Validação independente.** Suíte separada com um teste por métrica:
   ```bash
   python3 ../validation/run_all_validations.py
   ```
