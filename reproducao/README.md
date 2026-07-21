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
versionados em `FINAIS/` e também arquivados permanentemente em Zenodo:
[10.5281/zenodo.21211716](https://doi.org/10.5281/zenodo.21211716).

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
├── FINAIS/                    Logs brutos das 240 execuções (output.jsonl); espelhados no Zenodo
├── validation/                Suíte independente de validação estatística
├── verify_all_numbers.py      Verifica os números reportados contra os dados brutos
└── requirements.txt

../data/sample_30_underspecified.csv   Amostra utilizada (na raiz do repositório)
```

## Pré-requisitos (Configuração do Sistema)

Este ambiente foi projetado e testado principalmente para rodar em **Ubuntu/Debian**. Para montar a infraestrutura do zero (por exemplo, em uma Droplet recém-criada), siga os passos abaixo:

### Passo 1: Instalar as dependências do sistema
```bash
sudo apt update
sudo apt install -y build-essential python3-pip python3.12-venv netcat-openbsd curl

# Instalar Node.js 20.x (requisito do OpenHands)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Passo 2: Instalar o Docker
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl enable --now docker
```
> **Aviso sobre o Docker (Tamanho das imagens):** O download inicial das imagens do OpenHands e SWE-bench (feito automaticamente mais adiante) pode levar um bom tempo. São **mais de 40GB** de imagens Docker que precisam ser baixadas e descompactadas na primeira execução.

> **Nota para usuários de macOS (Processadores Apple Silicon):** Devido à arquitetura ARM e à necessidade de emular os contêineres Linux x86 exigidos pelo SWE-bench, este experimento **não rodará perfeitamente** em Macs com chips da família M. A emulação (Rosetta 2) pode causar extrema lentidão e travamentos na comunicação interna do agente (IPC do Jupyter). Para uma reprodução fiel e estável, recomenda-se fortemente executar em hardware x86_64 nativo.

### Passo 3: Instalar o Poetry e Fazer o Build
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
export PATH="/root/.local/bin:$PATH"

# Instala as dependências Python e faz o build do orquestrador (OpenHands)
make build
poetry install

# Cria o arquivo de configuração necessário para o OpenHands
cp config.template.toml config.toml
```


## Como reproduzir

### Passo 4: Amostragem
Gere a amostra das 30 issues originais (e o arquivo de filtro para o OpenHands):
```bash
poetry run python3 reproducao/scripts/generate_sample.py
```
*Isso seleciona 30 das 500 issues de `data/underspecified.csv` com semente 42.* Rode a partir da raiz do projeto: o script grava tanto em `data/sample_30_underspecified.csv` quanto no filtro `evaluation/benchmarks/swe_bench/config.toml` que o OpenHands exige.

**Nota importante:** esse `evaluation/benchmarks/swe_bench/config.toml` gerado aqui não é o mesmo arquivo do `config.toml` da raiz do repositório (que guarda os LLMs) — é um arquivo local, **não versionado no git**, lido pela função `filter_dataset()` em `interact_run_infer.py`. Ela sempre procura esse arquivo e, se ele existir com uma chave `selected_ids`, filtra **qualquer** dataset carregado (seja `princeton-nlp/SWE-bench_Lite` completo, seja o CSV da amostra) para as mesmas 30 instâncias — independente do que o `--dataset`/`DATASET_PATH` de cada script de bateria diga. Foi por isso que, na execução original, os scripts do 7B/32B (que ainda apontavam para `princeton-nlp/SWE-bench_Lite`) acabaram avaliando exatamente as mesmas 30 issues do 14B: o filtro sempre restringia a execução por baixo dos panos. Os scripts já foram atualizados (depois da execução) para apontar direto para `data/sample_30_underspecified.csv`, deixando isso explícito e removendo a dependência desse arquivo lateral — mas rodar o Passo 4 primeiro continua necessário de qualquer forma, para que o `interact_run_infer.py` (e o próprio `evaluation/benchmarks/swe_bench/config.toml`) funcionem.

### Passo 5: Execução
Os scripts em `baterias/<escala>/` rodam as sessões de cada configuração (recebendo automaticamente a amostra de 30 instâncias gerada no Passo 4).

Para reproduzir a infraestrutura exata do experimento, você precisará de: uma Droplet na DigitalOcean (x86_64) orquestrando o código, um Mac local rodando os modelos 1.5B/7B/14B via MLX expostos ao servidor por um túnel Ngrok (API compatível com OpenAI) para o Agente, e acesso ao OpenRouter para o Simulador (Gemini) e para o modelo 32B.

**1. Configure o Simulador (Gemini via OpenRouter)**

Como o OpenHands usa o LiteLLM para rotear os modelos, adicione o bloco abaixo no `config.toml` (na raiz do repositório) com o nome exato buscado pelo script (`gemini-flash-latest`):

```toml
# Simulador
[llm.gemini-flash-latest]
model = "openai/~google/gemini-flash-latest" # O "~" antes de "google/" é obrigatório: sem ele a OpenRouter rejeita o slug
base_url = "https://openrouter.ai/api/v1"
api_key = "SUA_CHAVE_AQUI_DO_OPENROUTER"
```

**Alternativa de Simulador: GPT-4o-mini (via OpenAI Platform)**

Os scripts `run_gpt.sh` / `run_rq3_gpt.sh` / `run_rq3.sh` (em cada `baterias/<escala>/`) usam `SIMULATORS=("gpt-4o-mini")` no lugar do Gemini. Diferente do Gemini, esse simulador **deve** ser executado direto na OpenAI Platform (não via OpenRouter). Como não existe hoje nenhum bloco `[llm.gpt-4o-mini]` no `config.toml` cujo nome bata exatamente com `gpt-4o-mini`, o LiteLLM cai no comportamento padrão e usa a API oficial da OpenAI sozinho. Basta exportar a chave antes de rodar o script:
```bash
export OPENAI_API_KEY="sua_chave_da_openai"
bash reproducao/baterias/7b/run_gpt.sh
```
**Atenção:** o `config.toml` já traz uma seção `[llm.gpt4o-mini]` (sem hífen entre "gpt4o" e "mini") com `api_key = "your-api-key"`. Esse nome **não** corresponde a `gpt-4o-mini` (o valor usado em `SIMULATORS`), então essa seção é ignorada pelo simulador — não adianta editá-la.

**2. Configure o Agente (Qwen via MLX + Ngrok, Apple Silicon)**

1. **No seu Mac**, instale o MLX e inicie o servidor do modelo desejado:
   ```bash
   pip3 install mlx-lm
   # Nota: Se trocar o tamanho do modelo (ex: para 1.5B ou 14B), lembre-se de manter
   # o sufixo de quantização no nome (como -4bit ou -bf16) exigido pelo mlx-community
   python3 -m mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --host 0.0.0.0 --port 8080
   ```
2. **No seu Mac**, exponha o servidor para a internet usando o Ngrok:
   ```bash
   ngrok http 8080
   ```
3. **No Droplet**, configure o `config.toml` apontando para o URL gerado pelo Ngrok.
   **Atenção:** O nome do bloco `[llm.*]` deve corresponder exatamente ao nome do modelo (Agent) declarado no script `.sh` (ex: `[llm.qwen_1_5b]`, `[llm.qwen_7b]`, ou `[llm.qwen_14b]`). Se você for testar todos, insira um bloco para cada.

   Exemplo para o 7B:
   ```toml
   [llm.qwen_7b]
   model = "openai/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
   base_url = "https://SEU-URL.ngrok-free.app/v1"
   api_key = "sk-1234" # API Key fictícia necessária para a biblioteca
   ```

**3. Execute o script da bateria no Droplet** (o agente apontará para o seu Mac automaticamente, enquanto o simulador baterá na API da OpenRouter):
```bash
bash reproducao/baterias/7b/run_gemini.sh
```

### Passo 6: Ambiente de análise
As etapas de métricas, verificação e validação usam um ambiente Python separado do OpenHands (testado em Python 3.10). A partir da raiz do repositório:
```bash
cd reproducao
pip install -r requirements.txt
```

### Passo 7: Métricas
Os dados processados ficam em `experiments/`: `extracted_qa_pairs/` (avg_q), `cosine_distance/` (IG) e `llm_as_judge/` (Judge e experimento de swap). Os CSVs já vêm versionados no repositório; os `output.jsonl` brutos que os originam ficam em `reproducao/FINAIS/`, versionados no próprio repositório e também arquivados no Zenodo. Para reproduzir os números do artigo bastam os CSVs já versionados (passos 8 e 9); para reexecutar a extração do zero (ex.: `extracted_qa_pairs/extract_qwen_qa_pairs.py`), a pasta `reproducao/FINAIS/` já está disponível no repositório.

### Passo 8: Verificação dos números
Confere cada valor reportado no artigo contra os dados brutos (a partir de `reproducao/`):
```bash
python3 verify_all_numbers.py
```
Saída esperada: status `OK` em todos os checks (com tolerância de arredondamento no último dígito).

### Passo 9: Validação independente
Suíte separada, com um teste por métrica (a partir de `reproducao/`):
```bash
python3 validation/run_all_validations.py
```
Saída esperada: `PASS: 35` e `FAIL: 0`.

## Versões e ambiente (reprodutibilidade)

O droplet DigitalOcean usado na execução foi destruído após o experimento. As
versões abaixo foram reconstruídas a partir de artefatos do próprio repositório:
o `poetry.lock` do OpenHands (raiz do repo), os campos `metadata`/`llm_config`
de cada `output.jsonl` em `FINAIS/` e os caches `.pyc`.

**Ambiente de execução (OpenHands), rodado em ~2026-06-23.**

| Componente | Valor |
|---|---|
| OpenHands | `openhands-ai` 0.20.0, commit `68be32fcd0fb98671f998ecee073d9b38dc4a20c` |
| Python (execução) | 3.12 (`pyproject.toml`: `python = "^3.12"`) |
| Agente | `CodeActAgent`, `max_iterations=5`, `temperature=0.0`, `top_p=1.0` |
| litellm / openai / tiktoken | 1.83.0 / 2.43.0 / 0.13.0 |
| docker / datasets / pydantic | 7.1.0 / 3.0.1 / 2.13.4 |

**Modelos do agente** (todos com `temperature=0.0`):

| Escala | Identificador | Regime |
|---|---|---|
| 1.5B | `openai/Qwen/Qwen2.5-Coder-1.5B-Instruct` | local, 16-bit |
| 7B | `openai/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | local, MLX 4-bit |
| 14B | `openai/mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | local, MLX 4-bit |
| 32B | `openrouter/qwen/qwen-2.5-coder-32b-instruct` | API (OpenRouter) |

**Simuladores, juiz e embeddings:**

| Papel | Modelo |
|---|---|
| Simulador GPT-mini | `gpt-4o-mini` |
| Simulador Gemini (1.5B/7B/14B) | `gemini-flash-latest` |
| Simulador Gemini (32B) | `openrouter/google/gemini-3.5-flash` |
| Juiz | `gpt-4o` |
| Embeddings (IG) | `text-embedding-3-small` |

> **Aviso de reprodutibilidade:** `gemini-flash-latest` é uma tag móvel. A
> execução ocorreu em junho de 2026, portanto corresponde ao Gemini Flash
> vigente nessa data. Para reexecução fiel, fixe um snapshot específico.

**Ambiente de análise** (scripts deste diretório): Python 3.10 (cache `.pyc`),
separado do ambiente 3.12 do OpenHands. As versões em `requirements.txt` vêm do
`poetry.lock` do OpenHands e servem de referência; o ambiente 3.10 pode ter
usado versões ligeiramente diferentes.

**Infraestrutura:** orquestração em droplet DigitalOcean `ubuntu-s-4vcpu-8gb-nyc1`
(4 vCPU, 8 GB RAM, 160 GB SSD; ver `FINAIS/Settings - ...DigitalOcean.pdf`).
Inferência local dos modelos Qwen em um Apple MacBook Air (Apple Silicon) via
`mlx_lm`, exposto ao orquestrador por túnel Ngrok em endpoint compatível com a
OpenAI API. Cada issue foi resolvida em um contêiner Docker do SWE-Bench.

## Solução de Problemas Comuns (Troubleshooting)

- **`ModuleNotFoundError: No module named 'litellm'` ao rodar baterias**
  - **Causa:** O script não está encontrando as dependências do OpenHands, o que significa que o ambiente virtual não foi instalado/sincronizado.
  - **Solução:** Na raiz do projeto, rode `poetry install` e tente rodar o script novamente.

- **`KeyError: 'PASS_TO_PASS'` na linha 644 do interact_run_infer.py**
  - **Causa:** O script Python do OpenHands tentou carregar a amostra bruta de 30 instâncias em vez do Dataset completo, por não encontrar o arquivo de filtro de IDs na configuração.
  - **Solução:** Certifique-se de executar o Passo 4 (`poetry run python3 reproducao/scripts/generate_sample.py`) **antes** das baterias. Esse script cria automaticamente o arquivo `evaluation/benchmarks/swe_bench/config.toml` necessário.

- **`make build` falhando no `check-nodejs` com `Error 1` ou `Error 2`**
  - **Causa:** A versão do Node.js nos repositórios padrão do Ubuntu (apt) pode ser antiga (ex: v18), e o OpenHands exige Node >= 20.x.
  - **Solução:** Revise o Passo 1 do tutorial e certifique-se de executar o download do NodeSource (`curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -`) antes do `apt install nodejs`.

- **`Repository Not Found` ao tentar baixar o Qwen na mlx-community**
  - **Causa:** Repositórios locais da mlx-community exigem que a formatação da quantização (ex: `-4bit`, `-bf16`) seja incluída na flag `--model`.
  - **Solução:** Adicione o sufixo apropriado. Exemplo correto: `--model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`.
