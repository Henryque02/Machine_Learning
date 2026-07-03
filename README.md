# Machine Learning — Jornada de Aprendizado

Projetos e exercícios da minha jornada de Machine Learning, construída a partir da trilha da **Rocketseat** e complementada com desafios práticos. O repositório está organizado na ordem em que as habilidades foram desenvolvidas: fundamentos de estatística → análise exploratória → primeiro modelo → algoritmos supervisionados com deploy.

## Projetos

| # | Projeto | O que demonstra |
|---|---------|-----------------|
| 01 | [`estatistica.ipynb`](01_estatistica/estatistica.ipynb) · [desafio: faturamento de loja](01_estatistica/desafio/desafio_faturamento_loja.ipynb) | **Estatística descritiva** com Pandas — medidas de posição e dispersão; visualização de séries de faturamento com Matplotlib |
| 02 | [`eda_churn.ipynb`](02_eda/eda_churn.ipynb) · [desafio: Netflix Top 10](02_eda/desafio/desafio_netflix_top10.ipynb) | **Análise exploratória de dados (EDA)** — junção de 3 datasets de churn, tratamento de dados, detecção de outliers via z-score e **teste qui-quadrado de independência** (scipy) |
| 03 | [`modelo_diabetes.ipynb`](03_primeiro_modelo/modelo_diabetes.ipynb) · [desafio: previsão de vendas](03_primeiro_modelo/desafio/previsao_vendas.ipynb) | **Primeiro modelo de ML** — regressão linear com scikit-learn, divisão treino/teste e métricas (MAE, RMSE, R²) |
| 04 | ⭐ [`modelo_pontuacao.ipynb`](04_regressao_linear/simples/modelo_pontuacao.ipynb) + [`api_modelo_regressao.py`](04_regressao_linear/simples/api_modelo_regressao.py) · [`modelo_colesterol.ipynb`](04_regressao_linear/multipla/modelo_colesterol.ipynb) · [desafio: sistema de irrigação](04_regressao_linear/simples/desafio/sistema_irrigacao.ipynb) | **Pipeline completo de regressão** — EDA → treino → métricas → **análise de resíduos** (Shapiro-Wilk, Kolmogorov-Smirnov, QQ-plot) → serialização com joblib → **deploy como API REST com FastAPI** |

O destaque é o **projeto 04**: além de treinar e validar o modelo, a análise de resíduos verifica os pressupostos estatísticos da regressão, e o modelo serializado (`modelo_regressao.pkl`) é servido por uma API FastAPI — cobrindo o ciclo completo, do dado bruto à predição em produção.

## Estrutura

```
├── 01_estatistica/
├── 02_eda/
├── 03_primeiro_modelo/
├── 04_regressao_linear/
│   ├── simples/          ← notebook + API FastAPI + modelo .pkl
│   └── multipla/
└── pyproject.toml
```

Cada projeto é autocontido: os notebooks leem seus dados da pasta `datasets/` local, então podem ser executados em qualquer ordem. Os desafios ficam em subpastas `desafio/` dentro do módulo correspondente.

## Como executar

Requer Python ≥ 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # instala as dependências
uv run jupyter lab      # abre os notebooks
```

### API de predição

O modelo de pontuação × horas de estudo está disponível como API REST. O `modelo_regressao.pkl` já vem versionado no repositório justamente para permitir rodar a API sem retreinar (para regenerá-lo, basta executar o `modelo_pontuacao.ipynb` até o fim):

```bash
cd 04_regressao_linear/simples
uv run uvicorn api_modelo_regressao:app --reload
```

Exemplo de requisição:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"horas_estudo": 20.5}'
# → {"pontuacao_teste": 73}
```

## O que aprendi até aqui

- Estatística descritiva como base para entender dados antes de modelar
- EDA estruturada: limpeza, junção de fontes, outliers e testes de hipótese
- O fluxo padrão de ML supervisionado: separação treino/teste, treino, avaliação por métricas
- Que validar um modelo de regressão vai além do R² — a análise de resíduos verifica os pressupostos do modelo
- Como tirar um modelo do notebook e colocá-lo atrás de uma API (joblib + FastAPI + Pydantic)

*Repositório em evolução — novos algoritmos serão adicionados conforme a jornada avança.*
