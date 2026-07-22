from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TRADING_DAYS = 252
NY_TZ = ZoneInfo("America/New_York")

st.set_page_config(
    page_title="Series bursátiles y efecto de noticias",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Series bursátiles y efecto de noticias")
st.caption(
    "Compara NVIDIA y otras acciones, caracteriza sus series de tiempo y explora "
    "cómo el sentimiento de las noticias se relaciona con sus retornos."
)

with st.expander("Alcance y advertencias", expanded=False):
    st.markdown(
        """
        - La aplicación es educativa y no constituye asesoría financiera.
        - Una correlación entre noticias y retornos **no demuestra causalidad**.
        - Los titulares obtenidos desde Yahoo Finance suelen ser recientes. Para un
          análisis histórico más sólido, carga un CSV propio de noticias.
        - VADER funciona mejor con titulares en inglés. Los resultados en español
          deben interpretarse con cautela.
        """
    )


# ---------------------------------------------------------------------------
# Utilidades de datos de mercado
# ---------------------------------------------------------------------------
def clean_tickers(raw_text: str) -> list[str]:
    """Convierte una cadena separada por comas en tickers únicos y válidos."""
    tickers: list[str] = []
    for token in raw_text.replace(";", ",").split(","):
        ticker = token.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Extrae un campo OHLCV de una descarga de yfinance con columnas simples o MultiIndex."""
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)
        level_1 = raw.columns.get_level_values(1)

        if field in level_0:
            out = raw[field].copy()
        elif field in level_1:
            out = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if field not in raw.columns:
            return pd.DataFrame()
        out = raw[[field]].copy()
        if len(tickers) == 1:
            out.columns = [tickers[0]]

    if isinstance(out, pd.Series):
        out = out.to_frame(name=tickers[0])

    out.columns = [str(col).upper() for col in out.columns]
    out.index = pd.to_datetime(out.index, errors="coerce")
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    return out.sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def download_market_data(
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Descarga datos diarios y devuelve tablas por campo."""
    if not tickers:
        return {}

    # yfinance interpreta end como fecha exclusiva.
    end_exclusive = end_date + timedelta(days=1)

    raw = yf.download(
        tickers=list(tickers),
        start=start_date.isoformat(),
        end=end_exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
        timeout=30,
    )

    fields: dict[str, pd.DataFrame] = {}
    for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        extracted = _extract_field(raw, field, list(tickers))
        if not extracted.empty:
            fields[field] = extracted

    return fields


def choose_price_table(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Usa precio ajustado cuando existe; de lo contrario, cierre."""
    prices = fields.get("Adj Close", pd.DataFrame()).copy()
    if prices.empty or prices.dropna(how="all").empty:
        prices = fields.get("Close", pd.DataFrame()).copy()
    return prices.dropna(how="all")


def series_diagnostics(series: pd.Series) -> dict[str, object]:
    """Calcula métricas descriptivas y una clasificación sencilla de la serie."""
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) < 20:
        return {
            "Observaciones": len(s),
            "Retorno total": np.nan,
            "Volatilidad anual": np.nan,
            "Máx. drawdown": np.nan,
            "Tendencia": "Datos insuficientes",
            "Pendiente anual": np.nan,
            "ADF p-valor": np.nan,
            "Estacionariedad": "Datos insuficientes",
            "Autocorr. retornos (1 día)": np.nan,
            "Tipo de serie": "Datos insuficientes",
        }

    returns = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = s.iloc[-1] / s.iloc[0] - 1
    annual_vol = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

    running_max = s.cummax()
    drawdown = s / running_max - 1
    max_drawdown = drawdown.min()

    # Tendencia log-lineal para hacer comparables las pendientes entre activos.
    x = np.arange(len(s), dtype=float)
    log_prices = np.log(s.clip(lower=np.finfo(float).eps))
    slope_daily = np.polyfit(x, log_prices, 1)[0]
    annual_slope = np.exp(slope_daily * TRADING_DAYS) - 1

    if annual_slope > 0.10:
        trend = "Alcista"
    elif annual_slope < -0.10:
        trend = "Bajista"
    else:
        trend = "Lateral"

    try:
        adf_pvalue = float(adfuller(s, autolag="AIC")[1])
        stationarity = "Estacionaria" if adf_pvalue < 0.05 else "No estacionaria"
    except (ValueError, np.linalg.LinAlgError):
        adf_pvalue = np.nan
        stationarity = "No concluyente"

    lag1_autocorr = returns.autocorr(lag=1) if len(returns) >= 3 else np.nan

    if annual_vol < 0.20:
        vol_label = "volatilidad baja"
    elif annual_vol < 0.40:
        vol_label = "volatilidad media"
    else:
        vol_label = "volatilidad alta"

    series_type = f"{trend.lower()}, {vol_label}, {stationarity.lower()}"

    return {
        "Observaciones": len(s),
        "Retorno total": total_return,
        "Volatilidad anual": annual_vol,
        "Máx. drawdown": max_drawdown,
        "Tendencia": trend,
        "Pendiente anual": annual_slope,
        "ADF p-valor": adf_pvalue,
        "Estacionariedad": stationarity,
        "Autocorr. retornos (1 día)": lag1_autocorr,
        "Tipo de serie": series_type,
    }


def diagnostics_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in prices.columns:
        result = series_diagnostics(prices[ticker])
        result["Ticker"] = ticker
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Ticker")


def make_candlestick(fields: dict[str, pd.DataFrame], ticker: str) -> go.Figure | None:
    required = ("Open", "High", "Low", "Close")
    if not all(field in fields and ticker in fields[field].columns for field in required):
        return None

    ohlc = pd.DataFrame(
        {
            "Open": fields["Open"][ticker],
            "High": fields["High"][ticker],
            "Low": fields["Low"][ticker],
            "Close": fields["Close"][ticker],
        }
    ).dropna()

    if ohlc.empty:
        return None

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"Velas diarias: {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Utilidades de noticias
# ---------------------------------------------------------------------------
@st.cache_resource
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _nested_value(value: object, key: str = "url") -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get(key)
        return str(candidate) if candidate else None
    return None


def parse_yahoo_news_item(item: dict) -> dict[str, object] | None:
    """Normaliza formatos antiguos y nuevos de noticias de yfinance."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item

    title = content.get("title") or item.get("title")
    if not title:
        return None

    provider_obj = content.get("provider")
    if isinstance(provider_obj, dict):
        source = provider_obj.get("displayName") or provider_obj.get("name")
    else:
        source = provider_obj
    source = source or item.get("publisher") or item.get("source") or "Sin fuente"

    published_raw = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )

    published_at: pd.Timestamp | None = None
    if isinstance(published_raw, (int, float, np.integer, np.floating)):
        published_at = pd.to_datetime(published_raw, unit="s", utc=True, errors="coerce")
    elif published_raw:
        published_at = pd.to_datetime(published_raw, utc=True, errors="coerce")

    canonical = _nested_value(content.get("canonicalUrl"))
    clickthrough = _nested_value(content.get("clickThroughUrl"))
    url = (
        canonical
        or clickthrough
        or content.get("link")
        or item.get("link")
        or item.get("url")
        or ""
    )

    summary = content.get("summary") or content.get("description") or item.get("summary") or ""

    return {
        "published_at": published_at,
        "title": str(title),
        "source": str(source),
        "url": str(url),
        "summary": str(summary),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yahoo_news(query: str, news_count: int) -> pd.DataFrame:
    search = yf.Search(
        query=query,
        max_results=1,
        news_count=news_count,
        include_research=False,
        enable_fuzzy_query=True,
        raise_errors=False,
    )
    parsed = []
    for item in search.news or []:
        if isinstance(item, dict):
            row = parse_yahoo_news_item(item)
            if row:
                parsed.append(row)

    if not parsed:
        return pd.DataFrame(columns=["published_at", "title", "source", "url", "summary"])

    news = pd.DataFrame(parsed)
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    return news.drop_duplicates(subset=["title", "published_at"]).sort_values(
        "published_at", ascending=False
    )


def prepare_uploaded_news(uploaded_file) -> pd.DataFrame:
    news = pd.read_csv(uploaded_file)
    normalized_columns = {str(col).strip().lower(): col for col in news.columns}

    date_col = next(
        (normalized_columns[name] for name in ("published_at", "date", "fecha", "datetime") if name in normalized_columns),
        None,
    )
    title_col = next(
        (normalized_columns[name] for name in ("title", "headline", "titulo", "titular") if name in normalized_columns),
        None,
    )

    if date_col is None or title_col is None:
        raise ValueError(
            "El CSV debe incluir una columna de fecha "
            "(published_at/date/fecha/datetime) y una de titular "
            "(title/headline/titulo/titular)."
        )

    source_col = next(
        (normalized_columns[name] for name in ("source", "publisher", "fuente") if name in normalized_columns),
        None,
    )
    url_col = next(
        (normalized_columns[name] for name in ("url", "link", "enlace") if name in normalized_columns),
        None,
    )
    summary_col = next(
        (normalized_columns[name] for name in ("summary", "description", "resumen") if name in normalized_columns),
        None,
    )

    prepared = pd.DataFrame(
        {
            "published_at": pd.to_datetime(news[date_col], utc=True, errors="coerce"),
            "title": news[title_col].astype(str),
            "source": news[source_col].astype(str) if source_col else "CSV cargado",
            "url": news[url_col].fillna("").astype(str) if url_col else "",
            "summary": news[summary_col].fillna("").astype(str) if summary_col else "",
        }
    )

    return prepared.dropna(subset=["published_at", "title"]).drop_duplicates(
        subset=["title", "published_at"]
    )


def add_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return news.copy()

    analyzer = get_sentiment_analyzer()
    result = news.copy()
    result["sentiment"] = result["title"].fillna("").map(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    result["sentiment_label"] = pd.cut(
        result["sentiment"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["Negativa", "Neutral", "Positiva"],
        include_lowest=True,
    )
    return result


def align_to_trading_day(
    timestamp: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
) -> pd.Timestamp | pd.NaT:
    """
    Asigna la noticia al día de mercado pertinente:
    - antes de las 16:00 ET y en día hábil: mismo día;
    - después del cierre o fin de semana: siguiente sesión.
    """
    if pd.isna(timestamp) or trading_dates.empty:
        return pd.NaT

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_ny = ts.tz_convert(NY_TZ)

    candidate = pd.Timestamp(ts_ny.date())
    normalized_dates = pd.DatetimeIndex(trading_dates).tz_localize(None).normalize()

    if ts_ny.hour >= 16:
        candidate += pd.Timedelta(days=1)

    position = normalized_dates.searchsorted(candidate, side="left")
    if position >= len(normalized_dates):
        return pd.NaT
    return normalized_dates[position]


def build_event_study(
    news: pd.DataFrame,
    target_prices: pd.Series,
    benchmark_prices: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Relaciona sentimiento diario con retornos anormales del activo."""
    target = pd.to_numeric(target_prices, errors="coerce").dropna()
    benchmark = pd.to_numeric(benchmark_prices, errors="coerce").dropna()

    aligned_prices = pd.concat(
        [target.rename("target"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()

    returns = aligned_prices.pct_change()
    abnormal_returns = (returns["target"] - returns["benchmark"]).rename("abnormal_return")

    work = news.dropna(subset=["published_at"]).copy()
    work["event_date"] = work["published_at"].map(
        lambda ts: align_to_trading_day(ts, aligned_prices.index)
    )
    work = work.dropna(subset=["event_date"])

    if work.empty:
        return pd.DataFrame(), abnormal_returns

    daily = (
        work.groupby("event_date", as_index=False)
        .agg(
            sentiment=("sentiment", "mean"),
            news_count=("title", "count"),
            positive_news=("sentiment_label", lambda x: int((x == "Positiva").sum())),
            negative_news=("sentiment_label", lambda x: int((x == "Negativa").sum())),
            headlines=("title", lambda x: " | ".join(x.astype(str).head(5))),
        )
        .sort_values("event_date")
    )

    abnormal = abnormal_returns.copy()
    next_day_abnormal = abnormal.shift(-1)
    car_0_2 = abnormal.rolling(window=3, min_periods=1).sum().shift(-2)

    daily = daily.set_index("event_date")
    daily["retorno_anormal_d0"] = abnormal.reindex(daily.index)
    daily["retorno_anormal_d1"] = next_day_abnormal.reindex(daily.index)
    daily["CAR_0_2"] = car_0_2.reindex(daily.index)
    daily = daily.reset_index()

    return daily, abnormal_returns


def correlation_or_nan(x: pd.Series, y: pd.Series) -> float:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")

    ticker_text = st.text_input(
        "Tickers para comparar",
        value="NVDA, AMD, AVGO, TSM, INTC",
        help="Usa símbolos de Yahoo Finance separados por comas.",
    )
    tickers = clean_tickers(ticker_text)

    today = date.today()
    default_start = today - timedelta(days=365 * 2)
    start_date = st.date_input(
        "Fecha inicial",
        value=default_start,
        max_value=today,
    )
    end_date = st.date_input(
        "Fecha final",
        value=today,
        min_value=start_date,
        max_value=today,
    )

    benchmark = st.text_input(
        "Benchmark",
        value="QQQ",
        help="Se usa para calcular retornos anormales.",
    ).strip().upper()

    selected_default = tickers.index("NVDA") if "NVDA" in tickers else 0
    target = st.selectbox(
        "Activo principal",
        options=tickers if tickers else ["NVDA"],
        index=selected_default if tickers else 0,
    )

    st.divider()
    news_source = st.radio(
        "Fuente de noticias",
        options=["Yahoo Finance (recientes)", "Cargar CSV histórico"],
    )

    news_count = st.slider(
        "Número máximo de titulares",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        disabled=news_source != "Yahoo Finance (recientes)",
    )

    uploaded_file = None
    if news_source == "Cargar CSV histórico":
        uploaded_file = st.file_uploader(
            "CSV de noticias",
            type=["csv"],
            help=(
                "Columnas obligatorias: date/published_at y title/headline. "
                "Opcionales: source, url y summary."
            ),
        )

    run_analysis = st.button("Analizar", type="primary", use_container_width=True)


if not tickers:
    st.error("Escribe al menos un ticker válido.")
    st.stop()

if start_date >= end_date:
    st.error("La fecha inicial debe ser anterior a la fecha final.")
    st.stop()

all_tickers = list(tickers)
if benchmark and benchmark not in all_tickers:
    all_tickers.append(benchmark)

if not run_analysis:
    st.info("Configura los parámetros en la barra lateral y pulsa **Analizar**.")
    st.stop()


# ---------------------------------------------------------------------------
# Descarga y análisis
# ---------------------------------------------------------------------------
with st.spinner("Descargando precios y preparando el análisis..."):
    try:
        market_fields = download_market_data(tuple(all_tickers), start_date, end_date)
    except Exception as exc:
        st.error(f"No fue posible descargar los datos de mercado: {exc}")
        st.stop()

prices_all = choose_price_table(market_fields)

if prices_all.empty:
    st.error(
        "No se obtuvieron precios. Verifica los tickers, el rango de fechas "
        "o inténtalo de nuevo más tarde."
    )
    st.stop()

available_tickers = [ticker for ticker in tickers if ticker in prices_all.columns]
missing_tickers = [ticker for ticker in tickers if ticker not in prices_all.columns]

if missing_tickers:
    st.warning("Sin datos para: " + ", ".join(missing_tickers))

if target not in prices_all.columns:
    st.error(f"No hay datos disponibles para el activo principal {target}.")
    st.stop()

if benchmark not in prices_all.columns:
    st.error(f"No hay datos disponibles para el benchmark {benchmark}.")
    st.stop()

prices = prices_all[available_tickers].dropna(how="all")

# KPIs principales
target_series = prices_all[target].dropna()
target_diag = series_diagnostics(target_series)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Último precio", f"{target_series.iloc[-1]:,.2f}")
kpi_2.metric("Retorno del periodo", f"{target_diag['Retorno total']:.2%}")
kpi_3.metric("Volatilidad anual", f"{target_diag['Volatilidad anual']:.2%}")
kpi_4.metric("Máximo drawdown", f"{target_diag['Máx. drawdown']:.2%}")

tab_compare, tab_diagnostics, tab_news, tab_data = st.tabs(
    ["Comparación", "Identificación de series", "Impacto de noticias", "Datos"]
)

with tab_compare:
    normalized = prices.copy()
    normalized = normalized.ffill().dropna(how="all")
    for column in normalized.columns:
        first_valid = normalized[column].dropna()
        if not first_valid.empty:
            normalized[column] = normalized[column] / first_valid.iloc[0] * 100

    fig_norm = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        labels={"value": "Índice base 100", "index": "Fecha", "variable": "Ticker"},
        title="Evolución comparada (base 100)",
    )
    fig_norm.update_layout(height=520, legend_title_text="Activo")
    st.plotly_chart(fig_norm, use_container_width=True)

    candlestick = make_candlestick(market_fields, target)
    if candlestick is not None:
        st.plotly_chart(candlestick, use_container_width=True)

    returns = prices.pct_change().dropna(how="all")
    rolling_vol = returns.rolling(21).std() * np.sqrt(TRADING_DAYS)
    fig_vol = px.line(
        rolling_vol,
        x=rolling_vol.index,
        y=rolling_vol.columns,
        labels={"value": "Volatilidad anualizada", "index": "Fecha", "variable": "Ticker"},
        title="Volatilidad móvil de 21 sesiones",
    )
    fig_vol.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vol, use_container_width=True)

with tab_diagnostics:
    diagnostics = diagnostics_table(prices)
    display_diag = diagnostics.copy()

    percent_columns = [
        "Retorno total",
        "Volatilidad anual",
        "Máx. drawdown",
        "Pendiente anual",
        "Autocorr. retornos (1 día)",
    ]
    formats = {column: "{:.2%}" for column in percent_columns}
    formats["ADF p-valor"] = "{:.4f}"

    st.dataframe(
        display_diag.style.format(formats, na_rep="—"),
        use_container_width=True,
    )

    st.markdown(
        """
        **Lectura rápida**
        - **Tendencia:** se estima con una pendiente log-lineal anualizada.
        - **Estacionariedad:** ADF con nivel de significancia del 5 %. En acciones,
          el precio suele ser no estacionario, mientras que los retornos suelen
          comportarse de forma más estable.
        - **Autocorrelación:** mide la relación lineal entre el retorno actual y el
          del día anterior.
        """
    )

    diag_download = diagnostics.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar diagnóstico CSV",
        data=diag_download,
        file_name="diagnostico_series.csv",
        mime="text/csv",
    )

with tab_news:
    try:
        if news_source == "Yahoo Finance (recientes)":
            with st.spinner("Consultando titulares recientes..."):
                news = fetch_yahoo_news(target, news_count)
        elif uploaded_file is not None:
            news = prepare_uploaded_news(uploaded_file)
        else:
            st.info("Carga un CSV para ejecutar el análisis histórico de noticias.")
            st.stop()
    except Exception as exc:
        st.error(f"No fue posible preparar las noticias: {exc}")
        st.stop()

    news = add_sentiment(news)

    if news.empty:
        st.warning(
            "No se encontraron titulares. Puedes cambiar el activo o cargar un CSV histórico."
        )
    else:
        event_study, abnormal_returns = build_event_study(
            news=news,
            target_prices=prices_all[target],
            benchmark_prices=prices_all[benchmark],
        )

        positive_share = (news["sentiment_label"] == "Positiva").mean()
        negative_share = (news["sentiment_label"] == "Negativa").mean()
        mean_sentiment = news["sentiment"].mean()

        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Titulares", f"{len(news):,}")
        n2.metric("Sentimiento medio", f"{mean_sentiment:.3f}")
        n3.metric("Noticias positivas", f"{positive_share:.1%}")
        n4.metric("Noticias negativas", f"{negative_share:.1%}")

        if event_study.empty:
            st.warning(
                "Las fechas de las noticias no coinciden con el periodo de precios seleccionado."
            )
        else:
            corr_d0 = correlation_or_nan(
                event_study["sentiment"], event_study["retorno_anormal_d0"]
            )
            corr_d1 = correlation_or_nan(
                event_study["sentiment"], event_study["retorno_anormal_d1"]
            )
            corr_car = correlation_or_nan(
                event_study["sentiment"], event_study["CAR_0_2"]
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Correlación sentimiento vs. retorno anormal D0",
                "—" if np.isnan(corr_d0) else f"{corr_d0:.3f}",
            )
            c2.metric(
                "Correlación sentimiento vs. retorno anormal D+1",
                "—" if np.isnan(corr_d1) else f"{corr_d1:.3f}",
            )
            c3.metric(
                "Correlación sentimiento vs. CAR [0,+2]",
                "—" if np.isnan(corr_car) else f"{corr_car:.3f}",
            )

            timeline = event_study.copy()
            timeline["retorno_anormal_d0_pct"] = timeline["retorno_anormal_d0"] * 100
            fig_events = go.Figure()
            fig_events.add_trace(
                go.Bar(
                    x=timeline["event_date"],
                    y=timeline["sentiment"],
                    name="Sentimiento medio",
                    yaxis="y",
                )
            )
            fig_events.add_trace(
                go.Scatter(
                    x=timeline["event_date"],
                    y=timeline["retorno_anormal_d0_pct"],
                    name="Retorno anormal D0 (%)",
                    mode="lines+markers",
                    yaxis="y2",
                )
            )
            fig_events.update_layout(
                title=f"Noticias y retorno anormal de {target} frente a {benchmark}",
                xaxis_title="Fecha de evento",
                yaxis=dict(title="Sentimiento"),
                yaxis2=dict(
                    title="Retorno anormal (%)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                height=520,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_events, use_container_width=True)

            scatter_data = event_study.dropna(
                subset=["sentiment", "retorno_anormal_d1"]
            )
            if len(scatter_data) >= 3:
                fig_scatter = px.scatter(
                    scatter_data,
                    x="sentiment",
                    y="retorno_anormal_d1",
                    size="news_count",
                    hover_data=["event_date", "headlines"],
                    trendline="ols",
                    labels={
                        "sentiment": "Sentimiento medio diario",
                        "retorno_anormal_d1": "Retorno anormal del día siguiente",
                        "news_count": "Número de noticias",
                    },
                    title="Relación entre sentimiento y retorno anormal del día siguiente",
                )
                fig_scatter.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info(
                    "Se necesitan al menos tres fechas con datos válidos para la regresión."
                )

            by_label = (
                news.assign(event_date=news["published_at"].map(
                    lambda ts: align_to_trading_day(ts, prices_all.index)
                ))
                .dropna(subset=["event_date"])
                .merge(
                    event_study[["event_date", "retorno_anormal_d0", "retorno_anormal_d1", "CAR_0_2"]],
                    on="event_date",
                    how="left",
                )
                .groupby("sentiment_label", observed=False)[
                    ["retorno_anormal_d0", "retorno_anormal_d1", "CAR_0_2"]
                ]
                .mean()
            )

            if not by_label.empty:
                by_label_plot = by_label.reset_index().melt(
                    id_vars="sentiment_label",
                    var_name="Ventana",
                    value_name="Retorno anormal medio",
                )
                fig_labels = px.bar(
                    by_label_plot,
                    x="sentiment_label",
                    y="Retorno anormal medio",
                    color="Ventana",
                    barmode="group",
                    labels={
                        "sentiment_label": "Clasificación del titular",
                        "Retorno anormal medio": "Retorno anormal medio",
                    },
                    title="Retorno anormal medio según sentimiento",
                )
                fig_labels.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig_labels, use_container_width=True)

            st.subheader("Eventos agregados por sesión")
            event_display = event_study.copy()
            st.dataframe(
                event_display.style.format(
                    {
                        "sentiment": "{:.3f}",
                        "retorno_anormal_d0": "{:.2%}",
                        "retorno_anormal_d1": "{:.2%}",
                        "CAR_0_2": "{:.2%}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
            )

        st.subheader("Titulares analizados")
        news_display = news.copy()
        news_display["published_at"] = news_display["published_at"].dt.strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        news_display = news_display[
            ["published_at", "source", "title", "sentiment", "sentiment_label", "url"]
        ]
        st.dataframe(
            news_display.style.format({"sentiment": "{:.3f}"}),
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("Enlace"),
            },
        )

        news_download = news_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar noticias analizadas",
            data=news_download,
            file_name=f"noticias_{target}.csv",
            mime="text/csv",
        )

with tab_data:
    st.subheader("Precios ajustados")
    st.dataframe(prices_all.sort_index(ascending=False), use_container_width=True)

    price_download = prices_all.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar precios CSV",
        data=price_download,
        file_name="precios_mercado.csv",
        mime="text/csv",
    )

    st.subheader("Rendimientos diarios")
    daily_returns = prices_all.pct_change().sort_index(ascending=False)
    st.dataframe(
        daily_returns.style.format("{:.3%}", na_rep="—"),
        use_container_width=True,
    )
