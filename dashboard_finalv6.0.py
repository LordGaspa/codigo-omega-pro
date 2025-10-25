# -*- coding: utf-8 -*-
# CÓDIGO ÔMEGA - PROTÓTIPO (v13.8 - Correção de Taxa e Sincronia)
#
# OBJETIVO DESTA VERSÃO:
# - [CORREÇÃO CRÍTICA] Ajustar a TAXA_CORRETAGEM para um valor realista (0.075%).
# - Sincronizar o portfólio com a versão Pro mais recente.
# ----------------------------------------------------------------------------

import warnings
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from binance.client import Client
from typing import List, Dict, Any, Tuple, Optional

warnings.simplefilter(action="ignore", category=FutureWarning)

st.set_page_config(
    layout="wide",
    page_title="Código Ômega - Multi-Estratégia",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>[data-testid="stMetricValue"] {font-size: 2.2em;}</style>""",
    unsafe_allow_html=True,
)

# [CORREÇÃO] A taxa foi ajustada de 7.5% para 0.075%
CAPITAL_INICIAL, TAXA_CORRETAGEM, ANOS_DE_DADOS_BACKTEST = 1000.0, 0.00075, 8

# [SINCRONIZADO] Portfólio completo e atualizado
PARAMETROS_OTIMIZADOS = {
    "BTCUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 2.0 },
    "ETHUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 80, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.5 },
    "BCHUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 9, "MEDIA_LENTA_PER": 80, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.5 },
    "BNBUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 12, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.0 },
    "SOLUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 80, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.0 },
    "XRPUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 3.0 },
    "DOGEUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 12, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.0 },
    "TRXUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_4HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 20, "ATR_MULTIPLICADOR": 3.5 },
    "LINKUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_4HOUR, "MEDIA_RAPIDA_PER": 9, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 3.5 },
    "SHIBUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 20, "ATR_MULTIPLICADOR": 3.5 },
    "SUIUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 20, "ATR_MULTIPLICADOR": 3.5 },
    "XLMUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 20, "ATR_MULTIPLICADOR": 2.5 },
    "AVAXUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 20, "ATR_MULTIPLICADOR": 2.0 },
    "TAOUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 250, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.5 },
    "FETUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 15, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.5 },
    "WLDUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 9, "MEDIA_LENTA_PER": 80, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 3.0 },
    "ADAUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 80, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 2.0 },
    "HBARUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.5, },
    "UNIUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 12, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 3.5, },
    "NEARUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 120, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.0, },
    "SANDUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_12HOUR, "MEDIA_RAPIDA_PER": 12, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 250, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.5, },
    "AXSUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 5, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.5, },
    "INJUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_8HOUR, "MEDIA_RAPIDA_PER": 21, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 14, "ATR_MULTIPLICADOR": 3.5, },
    "RNDRUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 12, "MEDIA_LENTA_PER": 100, "MEDIA_FILTRO_TENDENCIA_PER": 200, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 3.0, },
    "CAKEUSDT": { "PERIODO_CANDLE": Client.KLINE_INTERVAL_6HOUR, "MEDIA_RAPIDA_PER": 15, "MEDIA_LENTA_PER": 60, "MEDIA_FILTRO_TENDENCIA_PER": 150, "ATR_PERIODO": 10, "ATR_MULTIPLICADOR": 2.5, },
}

@st.cache_data(ttl=60 * 60 * 6)
def carregar_dados_brutos(symbol: str, kline_interval: str) -> pd.DataFrame:
    client = Client()
    start_date = (
        datetime.now() - timedelta(days=ANOS_DE_DADOS_BACKTEST * 365)
    ).strftime("%Y-%m-%d")
    klines = client.get_historical_klines(symbol, kline_interval, start_date)
    if not klines:
        return pd.DataFrame()
    df = pd.DataFrame(
        klines,
        columns=[
            "tempo_abertura", "abertura", "maxima", "minima", "fechamento",
            "volume", "tempo_fechamento", "volume_quote", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    for col in ["abertura", "maxima", "minima", "fechamento", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["tempo_fechamento"] = pd.to_datetime(df["tempo_fechamento"], unit="ms")
    df.set_index("tempo_fechamento", inplace=True)
    return df


def processar_dados_com_indicadores(
    df: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    df_proc = df.copy()
    df_proc["media_rapida"] = (
        df_proc["fechamento"].rolling(window=params["MEDIA_RAPIDA_PER"]).mean()
    )
    df_proc["media_lenta"] = (
        df_proc["fechamento"].rolling(window=params["MEDIA_LENTA_PER"]).mean()
    )
    df_proc["media_filtro"] = (
        df_proc["fechamento"]
        .rolling(window=params["MEDIA_FILTRO_TENDENCIA_PER"])
        .mean()
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
    df_proc["atr"] = true_range.rolling(window=params["ATR_PERIODO"]).mean()
    df_proc.dropna(inplace=True)
    return df_proc


def executar_backtest_completo(
    df_processado: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[float, List[Dict[str, Any]], List[float]]:
    capital, posicionado, qtde_ativo, stop_loss = CAPITAL_INICIAL, False, 0.0, 0.0
    trades, historico_capital = [], []
    for i in range(len(df_processado)):
        row = df_processado.iloc[i]
        valor_portfolio = capital if not posicionado else qtde_ativo * row["fechamento"]
        historico_capital.append(valor_portfolio)
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
                stop_loss = preco_compra - (atr_compra * params["ATR_MULTIPLICADOR"])
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
    capital_final = (
        capital
        if not posicionado
        else (qtde_ativo * df_processado.iloc[-1]["fechamento"]) * (1 - TAXA_CORRETAGEM)
    )
    return float(capital_final), trades, historico_capital


def analisar_resultados_backtest(
    capital_final: float,
    trades: List[Dict[str, Any]],
    historico_capital: List[float],
    retorno_bh_pct: float,
) -> Dict[str, Any]:
    resultados = {
        "capital_final_estrategia": float(capital_final),
        "retorno_estrategia_pct": ((capital_final - CAPITAL_INICIAL) / CAPITAL_INICIAL)
        * 100.0,
        "retorno_bh_pct": retorno_bh_pct,
    }
    df_trades = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(columns=["data", "tipo", "preco"])
    )
    compras = df_trades[df_trades["tipo"] == "COMPRA"]
    vendas = df_trades[df_trades["tipo"].str.contains("VENDA")]
    num_trades = int(min(len(compras), len(vendas)))
    resultados["num_ciclos"] = num_trades
    if num_trades > 0:
        vitorias = sum(
            1
            for i in range(num_trades)
            if vendas.iloc[i]["preco"] > compras.iloc[i]["preco"]
        )
        resultados["win_rate_pct"] = (vitorias / num_trades) * 100.0
    else:
        resultados["win_rate_pct"] = 0.0
    df_capital = pd.DataFrame(historico_capital, columns=["capital"])
    if not df_capital.empty:
        pico_anterior = df_capital["capital"].cummax()
        drawdown = (pico_anterior - df_capital["capital"]) / pico_anterior.replace(
            0, np.nan
        )
        resultados["max_drawdown_pct"] = (
            float(drawdown.max() * 100.0) if drawdown.notna().any() else 0.0
        )
    else:
        resultados["max_drawdown_pct"] = 0.0
    return resultados


def encontrar_sinal_vigente(
    df_processado: pd.DataFrame,
) -> Tuple[str, Optional[datetime], Optional[float], Optional[float]]:
    for i in range(len(df_processado) - 1, 0, -1):
        atual, anterior = df_processado.iloc[i], df_processado.iloc[i - 1]
        if (
            anterior["media_rapida"] <= anterior["media_lenta"]
            and atual["media_rapida"] > atual["media_lenta"]
            and atual["fechamento"] > atual["media_filtro"]
        ):
            return "TENDÊNCIA DE ALTA 🟢", atual.name, atual["fechamento"], atual["atr"]
        elif (
            anterior["media_rapida"] >= anterior["media_lenta"]
            and atual["media_rapida"] < atual["media_lenta"]
        ):
            return "TENDÊNCIA DE BAIXA 🔴", atual.name, atual["fechamento"], None
    return "NEUTRO ⚪", None, None, None


def formatar_preco(preco: Optional[float]) -> str:
    if preco is not None:
        if preco < 0.01:
            return f"${preco:,.8f}"
        if preco < 1.0:
            return f"${preco:,.4f}"
        return f"${preco:,.2f}"
    return "$0.00"


def gerar_grafico_de_impacto(
    df_bruto: pd.DataFrame,
    df_processado: pd.DataFrame,
    historico_capital: List[float],
    symbol: str,
    resultados: Dict[str, Any],
) -> go.Figure:
    capital_bh = (CAPITAL_INICIAL / df_bruto["abertura"].iloc[0]) * df_bruto[
        "fechamento"
    ]
    capital_estrategia = pd.Series(
        historico_capital, index=df_processado.index[: len(historico_capital)]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Código Ômega",
            x=capital_estrategia.index,
            y=capital_estrategia,
            line=dict(color="#299958", width=3),
            fill="tozeroy",
            fillcolor="rgba(91, 203, 138, 0.5)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Buy & Hold",
            x=capital_bh.index,
            y=capital_bh,
            line=dict(color="#F39C12", width=2),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.4)",
        )
    )
    pico_valor = capital_estrategia.max() if not capital_estrategia.empty else 0
    texto_metricas = (
        f"<b>Retorno Ômega:</b> <span style='color: #299958;'>{resultados['retorno_estrategia_pct']:,.2f}%</span><br>"
        f"<b>Retorno B&H:</b> <span style='color: #F39C12;'>{resultados['retorno_bh_pct']:,.2f}%</span><br>"
        f"────────────────────<br>"
        f"<b>Pico de Capital:</b> ${pico_valor:,.2f}<br>"
        f"<b>Taxa de Acerto:</b> {resultados['win_rate_pct']:.2f}%<br>"
        f"<b>Nº de Trades:</b> {resultados['num_ciclos']}"
    )
    fig.add_annotation(
        text=texto_metricas,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        font=dict(size=16, color="black"),
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.75)",
        bordercolor="rgba(169, 169, 169, 0.5)",
        borderwidth=1,
        align="left",
    )
    fig.update_layout(
        title=dict(
            text=f"<b>Performance: Código Ômega vs. Mercado ({symbol})</b>",
            y=0.95,
            x=0.4,
            font=dict(size=24, color="black"),
        ),
        template="plotly_white",
        height=600,
        xaxis=dict(
            title="", tickfont=dict(size=14, color="black"), gridcolor="#D3D3D3"
        ),
        yaxis=dict(
            title="Capital (USDT)",
            tickfont=dict(size=14, color="black"),
            gridcolor="#D3D3D3",
        ),
        legend=dict(
            x=0.02,
            y=0.65,
            xanchor="left",
            yanchor="top",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.5)",
        ),
        hovermode="x unified",
        paper_bgcolor="#F0F0F0",
        plot_bgcolor="#F0F0F0",
    )
    return fig


def gerar_lista_de_trades(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df_trades = pd.DataFrame(trades)
    compras = df_trades[df_trades["tipo"] == "COMPRA"].reset_index(drop=True)
    vendas = df_trades[df_trades["tipo"].str.contains("VENDA")].reset_index(drop=True)
    num_ciclos = min(len(compras), len(vendas))
    if num_ciclos == 0:
        return pd.DataFrame()
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
    return (
        pd.DataFrame(lista_trades_detalhados)
        .sort_index(ascending=False)
        .reset_index(drop=True)
    )


def gerar_radar_de_sinais(barra_progresso: st.progress) -> pd.DataFrame:
    lista_de_sinais = []
    ativos = list(PARAMETROS_OTIMIZADOS.keys())
    for i, ativo in enumerate(ativos):
        params = PARAMETROS_OTIMIZADOS[ativo]
        barra_progresso.progress((i + 1) / len(ativos), text=f"Analisando {ativo}...")
        dados_brutos = carregar_dados_brutos(ativo, params["PERIODO_CANDLE"])
        if not dados_brutos.empty:
            dados_proc = processar_dados_com_indicadores(dados_brutos, params)
            if not dados_proc.empty:
                sinal, data, preco_sinal, _ = encontrar_sinal_vigente(dados_proc)
                preco_atual = dados_proc.iloc[-1]["fechamento"]
                variacao = (
                    ((preco_atual - preco_sinal) / preco_sinal) * 100
                    if preco_sinal and preco_sinal > 0
                    else 0
                )
                lista_de_sinais.append(
                    {
                        "Ativo": ativo,
                        "Último Sinal": sinal,
                        "Data do Sinal": data,
                        "Variação desde o Sinal (%)": variacao,
                    }
                )
    if not lista_de_sinais:
        return pd.DataFrame()
    df_sinais = pd.DataFrame(lista_de_sinais)
    df_sinais["Data do Sinal"] = pd.to_datetime(df_sinais["Data do Sinal"])
    return df_sinais.sort_values(by="Data do Sinal", ascending=False).reset_index(
        drop=True
    )


# --- INÍCIO DA EXECUÇÃO DA UI ---
st.title("🚀 Código Ômega - Multi-Estratégia")
with st.expander("📡 Radar de Sinais do Portfólio", expanded=True):
    barra_progresso_radar = st.progress(0, text="Iniciando análise do portfólio...")
    try:
        df_radar = gerar_radar_de_sinais(barra_progresso_radar)
        barra_progresso_radar.empty()
        if not df_radar.empty:
            st.dataframe(
                df_radar.style.format(
                    {
                        "Data do Sinal": lambda dt: (
                            dt.strftime("%d/%m/%Y %H:%M") if pd.notna(dt) else "—"
                        ),
                        "Variação desde o Sinal (%)": "{:,.2f}%",
                    }
                ).apply(
                    lambda row: [
                        (
                            "color: green"
                            if "ALTA" in row["Último Sinal"]
                            else (
                                "color: red"
                                if "BAIXA" in row["Último Sinal"]
                                else "color: black"
                            )
                        )
                    ]
                    * len(row),
                    axis=1,
                ),
                use_container_width=True,
            )
        else:
            st.warning("Não foi possível gerar o radar de sinais.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar o radar de sinais: {e}")
        barra_progresso_radar.empty()

st.markdown("---")
st.sidebar.title("Controles de Análise")
lista_ativos_sidebar = sorted(list(PARAMETROS_OTIMIZADOS.keys()))
ativo_selecionado = st.sidebar.selectbox(
    "Selecione o Ativo para Análise Detalhada:", lista_ativos_sidebar, index=0
)

params_ativo = PARAMETROS_OTIMIZADOS[ativo_selecionado]
dados_brutos_ativo = carregar_dados_brutos(
    ativo_selecionado, params_ativo["PERIODO_CANDLE"]
)

if dados_brutos_ativo.empty or len(dados_brutos_ativo) < 2:
    st.error(f"Não foi possível carregar dados para {ativo_selecionado}.")
else:
    dados_proc_ativo = processar_dados_com_indicadores(dados_brutos_ativo, params_ativo)
    preco_inicial_bh = dados_brutos_ativo["abertura"].iloc[1]
    preco_final_bh = dados_brutos_ativo["fechamento"].iloc[-1]
    retorno_bh_pct = (
        ((preco_final_bh - preco_inicial_bh) / preco_inicial_bh) * 100.0
        if preco_inicial_bh > 0
        else 0
    )
    with st.spinner(f"Executando backtest detalhado para {ativo_selecionado}..."):
        capital_final, trades, historico_capital = executar_backtest_completo(
            dados_proc_ativo, params_ativo
        )
        resultados = analisar_resultados_backtest(
            capital_final, trades, historico_capital, retorno_bh_pct
        )
        df_lista_trades = gerar_lista_de_trades(trades)
    sinal_vigente, data_sinal, preco_no_sinal, atr_no_sinal = encontrar_sinal_vigente(
        dados_proc_ativo
    )
    preco_atual = (
        dados_proc_ativo.iloc[-1]["fechamento"] if not dados_proc_ativo.empty else 0
    )
    st.header(f"Análise Detalhada: {ativo_selecionado}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Preço Atual", formatar_preco(preco_atual))
        st.markdown("---")
        st.subheader("Sinal Vigente:")
        if "ALTA" in sinal_vigente:
            st.markdown(
                f"<h2 style='text-align: center; color: green;'>{sinal_vigente.split('🟢')[0].strip()} 🟢</h2>",
                unsafe_allow_html=True,
            )
        elif "BAIXA" in sinal_vigente:
            st.markdown(
                f"<h2 style='text-align: center; color: red;'>{sinal_vigente.split('🔴')[0].strip()} 🔴</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<h2 style='text-align: center; color: black;'>{sinal_vigente.split('⚪')[0].strip()} ⚪</h2>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.subheader("Contexto do Último Sinal")
        if data_sinal and preco_no_sinal:
            variacao = (
                ((preco_atual - preco_no_sinal) / preco_no_sinal) * 100
                if preco_no_sinal > 0
                else 0
            )
            cor_variacao = "green" if variacao >= 0 else "red"
            st.markdown(
                f"**Início em:** {data_sinal.strftime('%d/%m/%y às %H:%M')}<br>**Preço no Sinal:** {formatar_preco(preco_no_sinal)}<br>**Variação:** <span style='color: {cor_variacao};'>{variacao:+.2f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Aguardando novo sinal de entrada.")
        st.markdown("---")
        st.subheader("Análise de Risco (Nova Entrada)")
        if "ALTA" in sinal_vigente and atr_no_sinal and preco_no_sinal:
            stop_calculado = preco_no_sinal - (
                atr_no_sinal * params_ativo["ATR_MULTIPLICADOR"]
            )
            risco_calculado = (
                ((preco_no_sinal - stop_calculado) / preco_no_sinal) * 100
                if preco_no_sinal > 0
                else 0
            )
            st.markdown(
                f"**Stop Sugerido:** <span style='color: orange;'>{formatar_preco(stop_calculado)}</span><br>**Risco:** <span style='color: orange;'>{risco_calculado:.2f}%</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color: red;'><b>ENTRADA NÃO RECOMENDADA</b></span>",
                unsafe_allow_html=True,
            )
    with col2:
        tab_performance, tab_lista = st.tabs(
            ["Análise de Performance", "Lista de Trades"]
        )
        with tab_performance:
            figura_de_impacto = gerar_grafico_de_impacto(
                dados_brutos_ativo,
                dados_proc_ativo,
                historico_capital,
                ativo_selecionado,
                resultados,
            )
            st.plotly_chart(figura_de_impacto, use_container_width=True)
        with tab_lista:
            st.subheader(f"Histórico Detalhado de Trades para {ativo_selecionado}")
            if not df_lista_trades.empty:
                st.dataframe(
                    df_lista_trades.style.format(
                        {
                            "Preço Compra": formatar_preco,
                            "Preço Venda": formatar_preco,
                            "Resultado ($)": "${:,.2f}",
                            "Resultado (%)": "{:.2f}%",
                            "Capital Acumulado": "${:,.2f}",
                            "Data Compra": "{:%d/%m/%Y %H:%M}",
                            "Data Venda": "{:%d/%m/%Y %H:%M}",
                        }
                    ).applymap(
                        lambda val: "color: green" if val > 0 else "color: red",
                        subset=["Resultado ($)", "Resultado (%)"],
                    )
                )
            else:
                st.info("Nenhum ciclo de trade (compra/venda) foi completado.")

