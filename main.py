# -*- coding: utf-8 -*-
# CÓDIGO ÔMEGA - API BACKEND PROFISSIONAL (v1.9 - Lógica de Sinal Original Restaurada)

# 1. Importações
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional, Any

# 2. Inicialização da Aplicação
app = FastAPI(
    title="Código Ômega API",
    description="O motor de análise e sinais para o portfólio de estratégias do Código Ômega.",
    version="1.9.0",
)

origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANOS_DE_DADOS_BACKTEST = 8
CAPITAL_INICIAL = 1000.0
TAXA_CORRETAGEM = 0.001


# 3. Modelos de Dados (Pydantic)
class StrategyParams(BaseModel):
    media_rapida_per: int
    media_lenta_per: int
    media_filtro_tendencia_per: int
    atr_periodo: int
    atr_multiplicador: float


class SignalResponse(BaseModel):
    ativo: str
    sinal: str
    data_sinal: Optional[datetime]
    preco_no_sinal: Optional[float]
    stop_sugerido: Optional[float]
    risco_pct: Optional[float]
    preco_atual: Optional[float]
    variacao_pct: Optional[float]


class Trade(BaseModel):
    Data_Compra: datetime = Field(alias="Data Compra")
    Preco_Compra: float = Field(alias="Preço Compra")
    Data_Venda: datetime = Field(alias="Data Venda")
    Preco_Venda: float = Field(alias="Preço Venda")
    Resultado_USD: float = Field(alias="Resultado ($)")
    Resultado_PCT: float = Field(alias="Resultado (%)")
    Capital_Acumulado: float = Field(alias="Capital Acumulado")


class BacktestMetrics(BaseModel):
    retorno_estrategia_pct: float
    retorno_bh_pct: float
    num_ciclos: int
    win_rate_pct: float
    max_drawdown_pct: float


class CapitalHistoryPoint(BaseModel):
    timestamp: datetime
    capital: float


class BacktestResponse(BaseModel):
    ativo: str
    params: StrategyParams
    metricas: BacktestMetrics
    historico_capital: List[CapitalHistoryPoint]
    historico_capital_bh: List[CapitalHistoryPoint]
    lista_trades: List[Trade]


# ==============================================================================
# --- LÓGICA DE DADOS (O CÉREBRO) ---
# ==============================================================================
def carregar_dados_binance(symbol: str, interval: str) -> pd.DataFrame:
    client = Client()
    start_date = (
        datetime.now() - timedelta(days=ANOS_DE_DADOS_BACKTEST * 365)
    ).strftime("%Y-%m-%d")
    try:
        klines = client.get_historical_klines(symbol, interval, start_date)
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(
            klines,
            columns=[
                "tempo_abertura",
                "abertura",
                "maxima",
                "minima",
                "fechamento",
                "volume",
                "tempo_fechamento",
                "volume_quote",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        numeric_cols = ["abertura", "maxima", "minima", "fechamento", "volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df["tempo_fechamento"] = pd.to_datetime(df["tempo_fechamento"], unit="ms")
        df.set_index("tempo_fechamento", inplace=True)
        return df
    except Exception as e:
        print(f"Erro ao buscar dados para {symbol}: {e}")
        return pd.DataFrame()


def calcular_indicadores(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc["media_rapida"] = (
        df_proc["fechamento"].rolling(window=params.media_rapida_per).mean()
    )
    df_proc["media_lenta"] = (
        df_proc["fechamento"].rolling(window=params.media_lenta_per).mean()
    )
    df_proc["media_filtro"] = (
        df_proc["fechamento"].rolling(window=params.media_filtro_tendencia_per).mean()
    )
    ranges = pd.concat(
        [
            df_proc["maxima"] - df_proc["minima"],
            (df_proc["maxima"] - df_proc["fechamento"].shift()).abs(),
            (df_proc["minima"] - df_proc["fechamento"].shift()).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    df_proc["atr"] = true_range.rolling(window=params.atr_periodo).mean()
    df_proc.dropna(inplace=True)
    return df_proc


def identificar_sinal_atual(
    df_processado: pd.DataFrame,
) -> tuple[str, Optional[datetime], Optional[float], Optional[float]]:
    for i in range(len(df_processado) - 1, 0, -1):
        atual, anterior = df_processado.iloc[i], df_processado.iloc[i - 1]
        if (
            anterior["media_rapida"] <= anterior["media_lenta"]
            and atual["media_rapida"] > atual["media_lenta"]
            and atual["fechamento"] > atual["media_filtro"]
        ):
            return "COMPRA", atual.name, atual["fechamento"], atual["atr"]
        elif (
            anterior["media_rapida"] >= anterior["media_lenta"]
            and atual["media_rapida"] < atual["media_lenta"]
        ):
            return "VENDA", atual.name, atual["fechamento"], None
    return "NEUTRO", None, None, None


def executar_backtest_completo(
    df_processado: pd.DataFrame, params: StrategyParams
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capital, posicionado, qtde_ativo, stop_loss = CAPITAL_INICIAL, False, 0.0, 0.0
    trades, historico_capital_raw = [], []
    for i in range(len(df_processado)):
        row = df_processado.iloc[i]
        valor_portfolio = capital if not posicionado else qtde_ativo * row["fechamento"]
        historico_capital_raw.append(
            {"timestamp": row.name.to_pydatetime(), "capital": valor_portfolio}
        )
        if i == 0:
            continue
        prev_row = df_processado.iloc[i - 1]
        sinal_compra = (
            prev_row["media_rapida"] > prev_row["media_lenta"]
            and prev_row["fechamento"] > prev_row["media_filtro"]
        )
        sinal_venda_cruz = prev_row["media_rapida"] < prev_row["media_lenta"]
        if not posicionado and sinal_compra:
            preco_compra = row["abertura"]
            if preco_compra > 0 and not np.isnan(preco_compra):
                qtde_compra = (capital / preco_compra) * (1 - TAXA_CORRETAGEM)
                capital, qtde_ativo, posicionado = 0.0, qtde_compra, True
                atr_compra = row["atr"] if not pd.isna(row["atr"]) else 0.0
                stop_loss = preco_compra - (atr_compra * params.atr_multiplicador)
                trades.append(
                    {"data": row.name, "tipo": "COMPRA", "preco": preco_compra}
                )
        elif posicionado:
            stop_ativado = row["minima"] < stop_loss
            if stop_ativado or sinal_venda_cruz:
                preco_saida = stop_loss if stop_ativado else row["abertura"]
                if preco_saida > 0 and not np.isnan(preco_saida):
                    capital = (qtde_ativo * preco_saida) * (1 - TAXA_CORRETAGEM)
                    qtde_ativo, posicionado = 0.0, False
                    tipo_venda = "VENDA_STOP_ATR" if stop_ativado else "VENDA_CRUZ"
                    trades.append(
                        {"data": row.name, "tipo": tipo_venda, "preco": preco_saida}
                    )
    return trades, historico_capital_raw


def formatar_lista_trades(trades: List[dict]) -> List[dict]:
    df_trades = pd.DataFrame(trades)
    compras = df_trades[df_trades["tipo"] == "COMPRA"].reset_index(drop=True)
    vendas = df_trades[df_trades["tipo"].str.contains("VENDA")].reset_index(drop=True)
    num_ciclos = min(len(compras), len(vendas))
    if num_ciclos == 0:
        return []
    lista_trades_detalhados = []
    capital_acumulado = CAPITAL_INICIAL
    for i in range(num_ciclos):
        compra, venda = compras.iloc[i], vendas.iloc[i]
        retorno = (
            capital_acumulado
            * (venda["preco"] / compra["preco"])
            * ((1 - TAXA_CORRETAGEM) ** 2)
        )
        resultado_abs = retorno - capital_acumulado
        resultado_pct = (
            (resultado_abs / capital_acumulado) * 100 if capital_acumulado > 0 else 0
        )
        lista_trades_detalhados.append(
            {
                "Data Compra": compra["data"],
                "Preço Compra": compra["preco"],
                "Data Venda": venda["data"],
                "Preço Venda": venda["preco"],
                "Resultado ($)": resultado_abs,
                "Resultado (%)": resultado_pct,
                "Capital Acumulado": retorno,
            }
        )
        capital_acumulado = retorno
    return lista_trades_detalhados


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "success", "message": "Bem-vindo à API do Código Ômega!"}


@app.post("/signal/{symbol}", response_model=SignalResponse)
def get_current_signal(
    symbol: str, params: StrategyParams, interval: str
) -> SignalResponse:
    df_raw = carregar_dados_binance(symbol.upper(), interval)
    if df_raw.empty:
        raise HTTPException(
            status_code=404, detail=f"Dados não encontrados para {symbol}."
        )
    df_processed = calcular_indicadores(df_raw, params)
    if df_processed.empty:
        raise HTTPException(
            status_code=400, detail="Não há dados suficientes para os indicadores."
        )

    sinal, data_sinal, preco_no_sinal, atr_no_sinal = identificar_sinal_atual(
        df_processed
    )

    stop_sugerido, risco_pct, preco_atual, variacao_pct = None, None, None, None
    if not df_processed.empty:
        preco_atual = df_processed.iloc[-1]["fechamento"]
        if preco_no_sinal and preco_no_sinal > 0:
            variacao_pct = ((preco_atual - preco_no_sinal) / preco_no_sinal) * 100
    if sinal == "COMPRA" and atr_no_sinal and preco_no_sinal:
        stop_sugerido = preco_no_sinal - (atr_no_sinal * params.atr_multiplicador)
        if preco_no_sinal > 0:
            risco_pct = ((preco_no_sinal - stop_sugerido) / preco_no_sinal) * 100
    return SignalResponse(
        ativo=symbol.upper(),
        sinal=sinal,
        data_sinal=data_sinal,
        preco_no_sinal=preco_no_sinal,
        stop_sugerido=stop_sugerido,
        risco_pct=risco_pct,
        preco_atual=preco_atual,
        variacao_pct=variacao_pct,
    )


@app.post("/backtest/{symbol}", response_model=BacktestResponse)
def get_full_backtest(
    symbol: str, params: StrategyParams, interval: str
) -> BacktestResponse:
    df_raw = carregar_dados_binance(symbol.upper(), interval)
    if df_raw.empty or len(df_raw) < 2:
        raise HTTPException(
            status_code=404, detail=f"Dados insuficientes para {symbol}."
        )
    df_processed = calcular_indicadores(df_raw, params)
    if df_processed.empty:
        raise HTTPException(
            status_code=400, detail="Não há dados suficientes para os indicadores."
        )
    trades_raw, historico_capital = executar_backtest_completo(df_processed, params)
    df_raw_bh = df_raw.iloc[1:]
    preco_inicial_bh = df_raw_bh["abertura"].iloc[0]
    preco_final_bh = df_raw_bh["fechamento"].iloc[-1]
    retorno_bh_pct = (
        ((preco_final_bh - preco_inicial_bh) / preco_inicial_bh) * 100.0
        if preco_inicial_bh > 0
        else 0
    )
    df_trades = pd.DataFrame(trades_raw) if trades_raw else pd.DataFrame()
    num_ciclos = min(
        len(df_trades[df_trades["tipo"] == "COMPRA"]),
        len(df_trades[df_trades["tipo"].str.contains("VENDA")]),
    )
    vitorias = 0
    if num_ciclos > 0:
        compras = df_trades[df_trades["tipo"] == "COMPRA"].reset_index(drop=True)
        vendas = df_trades[df_trades["tipo"].str.contains("VENDA")].reset_index(
            drop=True
        )
        vitorias = sum(
            1
            for i in range(num_ciclos)
            if vendas.iloc[i]["preco"] > compras.iloc[i]["preco"]
        )
    win_rate_pct = (vitorias / num_ciclos) * 100.0 if num_ciclos > 0 else 0.0
    df_capital = pd.DataFrame([item["capital"] for item in historico_capital])
    pico_anterior = df_capital[0].cummax()
    drawdown = (pico_anterior - df_capital[0]) / pico_anterior.replace(0, np.nan)
    max_drawdown_pct = float(drawdown.max() * 100.0) if drawdown.notna().any() else 0.0
    capital_final = (
        historico_capital[-1]["capital"] if historico_capital else CAPITAL_INICIAL
    )
    retorno_estrategia_pct = (
        (capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL
    ) * 100.0
    metricas = BacktestMetrics(
        retorno_estrategia_pct=retorno_estrategia_pct,
        retorno_bh_pct=retorno_bh_pct,
        num_ciclos=num_ciclos,
        win_rate_pct=win_rate_pct,
        max_drawdown_pct=max_drawdown_pct,
    )
    lista_trades_formatada = formatar_lista_trades(trades_raw)
    df_bh = pd.DataFrame(index=df_raw_bh.index)
    df_bh["capital"] = (CAPITAL_INICIAL / preco_inicial_bh) * df_raw_bh["fechamento"]
    historico_capital_bh = [
        {"timestamp": index.to_pydatetime(), "capital": row["capital"]}
        for index, row in df_bh.iterrows()
    ]
    return BacktestResponse(
        ativo=symbol.upper(),
        params=params,
        metricas=metricas,
        historico_capital=historico_capital,
        historico_capital_bh=historico_capital_bh,
        lista_trades=lista_trades_formatada,
    )
