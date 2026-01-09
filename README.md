LSTM STOCK PRICE PREDICTION (VISA – VUS)

Este repositório implementa um pipeline completo de Machine Learning para séries temporais financeiras, utilizando LSTM (Long Short-Term Memory) para prever o log return de ações, com foco no ativo Visa (VUS).

O projeto cobre todo o ciclo de vida do modelo, incluindo:
- ingestão e validação de dados
- pré-processamento
- treinamento com validação manual de hiperparâmetros
- avaliação
- persistência de artefatos
- inferência via FastAPI
- monitoramento em produção

----------------------------------------------------------------

OBJETIVO

Prever o log return do preço de fechamento de uma ação com base em janelas temporais históricas, utilizando um modelo LSTM treinado com validação temporal adequada.

Observação importante:
O modelo não prevê diretamente o preço absoluto da ação, mas sim o retorno logarítmico, prática comum em finanças quantitativas por melhorar a estacionariedade da série e a estabilidade do treinamento.

----------------------------------------------------------------

ESTRUTURA DO PROJETO

project/
- data/
  - raw/                Dados brutos
  - trusted/            Scaler salvo
- models/
  - lstm_visa_vus.h5    Modelo treinado
  - grid_results.csv    Resultados do grid search manual
- src/
  - config.py           Configurações globais
  - dataset.py          Criação de sequências temporais
  - model.py            Definição do LSTM
  - preprocessing.py    Normalização / scaling
  - train.py            Treinamento com validação manual
  - main.py             Execução principal
  - tests/
    - validation.py     Validações de dados
- api/
  - app.py              FastAPI para inferência
- logs/
  - predictions.jsonl   Log de previsões
- requirements.txt
- README.md

----------------------------------------------------------------

DADOS

Fonte: arquivos CSV históricos
Frequência: diária

Colunas utilizadas:
- Open
- High
- Low
- Close
- Volume
- log_return

Cálculo do log return:

log_return = log(Close_t / Close_{t-1})

----------------------------------------------------------------

PRÉ-PROCESSAMENTO

1. Validação dos dados brutos
2. Cálculo do log return
3. Seleção das features
4. Normalização com MinMaxScaler
5. Criação de sequências temporais (lookback)

----------------------------------------------------------------

MODELO

- Arquitetura: LSTM
- Função de perda: Mean Squared Error (MSE)
- Otimizador: Adam
- Variável alvo: log_return
- Entrada: janelas temporais de dados normalizados

----------------------------------------------------------------

VALIDAÇÃO DE HIPERPARÂMETROS

Foi utilizada validação manual em grade (grid search), testando diferentes combinações de:
- tamanho da janela temporal (lookback)
- número de unidades LSTM
- taxa de dropout
- taxa de aprendizado
- batch size

A seleção do melhor modelo é realizada com base no MAE no conjunto de validação.

----------------------------------------------------------------

AVALIAÇÃO

Métricas utilizadas:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Comparação com baseline ingênuo (naive)

----------------------------------------------------------------

INFERÊNCIA COM FASTAPI

O modelo treinado é exposto via FastAPI.

Entrada da API:
- preços de fechamento (Close)

----------------------------------------------------------------

MONITORAMENTO EM PRODUÇÃO

Inclui métricas de:
- tempo de resposta
- uso de CPU e memória
