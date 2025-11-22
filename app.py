import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests


# ===================== 네이버 뉴스 API 키 설정 ===================== #
# 네이버 개발자 센터에서 발급받은 값으로 아래 두 줄을 바꿔 넣으세요.
NAVER_CLIENT_ID = "vjXipRSKeRApGyJjKQHt"
NAVER_CLIENT_SECRET = "lpBmqXkm1m"


# ===================== 공통 유틸 ===================== #

def clean_html(text: str) -> str:
    """네이버 검색 결과에 들어가는 <b> 태그 등 제거"""
    text = re.sub(r"<\/?b>", "", text)
    text = re.sub(r"&quot;", "\"", text)
    text = re.sub(r"&apos;", "'", text)
    text = re.sub(r"&amp;", "&", text)
    return text


# ===================== 네이버 뉴스 ===================== #

def get_naver_news(query: str, display: int = 6):
    """
    네이버 뉴스 검색 API를 이용해서 최근 뉴스 제목/요약/링크 가져오기.
    query: 회사 이름 (한글/영문 모두 가능)
    """
    if (
        not NAVER_CLIENT_ID
        or not NAVER_CLIENT_SECRET
        or "여기에_" in NAVER_CLIENT_ID
    ):
        return [], "⚠ 네이버 뉴스 API 키가 설정되지 않아 네이버 뉴스는 표시되지 않습니다."

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date",
    }

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        if res.status_code != 200:
            return [], f"⚠ 네이버 뉴스 API 호출 실패 (HTTP {res.status_code})"

        data = res.json()
        items = data.get("items", [])
        news_list = []
        for it in items:
            title = clean_html(it.get("title", "")) or "제목 없음"
            desc = clean_html(it.get("description", ""))
            link = it.get("link", "")
            news_list.append(
                {
                    "title": title,
                    "summary": desc,
                    "link": link,
                    "source": "네이버",
                }
            )
        if not news_list:
            return [], "네이버 뉴스 검색 결과가 없습니다."
        return news_list, ""
    except Exception as e:
        return [], f"⚠ 네이버 뉴스 API 오류: {e}"


# ===================== yfinance 뉴스 ===================== #

def get_yf_news(ticker_obj, limit: int = 6):
    try:
        raw = ticker_obj.news or []
    except Exception:
        raw = []
    news_list = []
    for it in raw[:limit]:
        title = it.get("title") or "제목 없음"
        link = it.get("link", "")
        pub = it.get("publisher", "")
        news_list.append(
            {
                "title": title,
                "summary": "",
                "link": link,
                "source": f"야후({pub})" if pub else "야후",
            }
        )
    return news_list


# ===================== 뉴스 요약/분위기 ===================== #

def summarize_news_combined(news_all):
    """
    뉴스 헤드라인을 텍스트로 요약 + 간단 긍/부정 카운트 반환
    """
    if not news_all:
        text = (
            "관련 뉴스 데이터를 찾지 못했습니다. "
            "다른 종목이나 기간을 시도해 보거나, 회사명을 조금 다르게 입력해 보세요."
        )
        return text, 0, 0

    lines = []
    pos_kw = ["상승", "호실적", "호재", "good", "beat", "record", "growth", "surge", "improve"]
    neg_kw = ["하락", "적자", "부진", "악재", "down", "loss", "slump", "weak", "lawsuit"]
    pos = neg = 0

    for i, n in enumerate(news_all, start=1):
        title = n["title"]
        src = n["source"]
        link = n["link"]
        lines.append(f"{i}. {title} ({src})")
        if link:
            lines.append(f"   링크: {link}")

        lt = title.lower()
        if any(k in lt for k in pos_kw):
            pos += 1
        if any(k in lt for k in neg_kw):
            neg += 1

    text = "📌 최근 주요 뉴스 헤드라인 요약:\n\n" + "\n".join(lines)
    text += "\n\n🧠 뉴스 분위기(아주 단순 키워드 기준):\n"
    if pos > neg:
        text += f"- 긍정 키워드가 더 많이 나타납니다. (긍정 {pos} vs 부정 {neg})\n"
    elif neg > pos:
        text += f"- 부정 키워드가 더 많이 나타납니다. (부정 {neg} vs 긍정 {pos})\n"
    else:
        text += "- 긍정/부정 키워드가 비슷하거나 뚜렷이 치우치지 않습니다.\n"

    text += (
        "\n초보자 팁: 뉴스 한두 개에 너무 휘둘리기보다는, "
        "실적(매출과 이익)과 장기 차트 흐름을 먼저 보고, "
        "뉴스는 참고자료로 활용하는 것이 좋습니다."
    )
    return text, pos, neg


# ===================== 주가/기술적 분석 ===================== #

def get_ticker_obj(ticker: str):
    return yf.Ticker(ticker)


def get_info_safe(ticker_obj) -> dict:
    raw = {}
    if hasattr(ticker_obj, "get_info"):
        try:
            raw = ticker_obj.get_info() or {}
        except Exception:
            raw = {}
    if not raw and hasattr(ticker_obj, "info"):
        try:
            raw = ticker_obj.info or {}
        except Exception:
            raw = {}
    return raw


def load_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(period=period)
    if df is None or df.empty:
        raise ValueError(
            "주가 데이터를 불러오지 못했습니다. 티커(symbol)를 다시 확인해 주세요.\n"
            "예) 미국: AAPL / 한국: 삼성전자 005930.KS"
        )
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index)
    return df


def add_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    df["Close"] = close

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    period = 14
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def calc_perf_stats(price_df: pd.DataFrame) -> dict:
    df = price_df.copy()
    df["Return"] = df["Close"].pct_change()
    if df["Return"].dropna().empty:
        return {"total_return": None, "annual_vol": None, "max_dd": None}

    total_return = (1 + df["Return"].dropna()).prod() - 1
    annual_vol = df["Return"].dropna().std() * np.sqrt(252)

    cum = (1 + df["Return"].fillna(0)).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "annual_vol": annual_vol,
        "max_dd": max_dd,
    }


def comment_from_technicals(df: pd.DataFrame) -> str:
    cols = [c for c in ["Close", "SMA_20", "SMA_60", "RSI_14"] if c in df.columns]
    clean = df[cols].dropna()
    if clean.empty:
        return "기술적 지표를 계산할 수 있는 기간이 너무 짧거나 데이터가 부족합니다. 기간을 더 길게 설정해 보세요."

    latest = clean.iloc[-1]
    comments = []

    if latest["SMA_20"] > latest["SMA_60"]:
        comments.append("20일선이 60일선 위에 있어 **중기적으로는 상승 추세 쪽에 가까운 모습**입니다.")
    elif latest["SMA_20"] < latest["SMA_60"]:
        comments.append("20일선이 60일선 아래에 있어 **중기적으로는 하락/조정 국면일 가능성**이 있습니다.")
    else:
        comments.append("20일선과 60일선이 비슷한 위치에 있어, 뚜렷한 추세보다는 **횡보 구간**에 가깝습니다.")

    rsi = latest["RSI_14"]
    if rsi >= 70:
        comments.append(f"RSI(14)가 {rsi:.1f}로, **단기 과열(과매수)** 구간에 가까워 보입니다. 단기 조정 가능성을 유의하세요.")
    elif rsi <= 30:
        comments.append(f"RSI(14)가 {rsi:.1f}로, **단기 과매도** 구간으로 볼 수 있습니다. 기술적 반등 여지가 생길 수 있습니다.")
    else:
        comments.append(f"RSI(14)가 {rsi:.1f}로, 과열/과매도 신호는 뚜렷하지 않습니다.")

    comments.append(
        "\n※ 위 내용은 단순 기술적 지표 기준의 참고용 코멘트이며, "
        "미래 수익을 보장하거나 매수·매도를 직접적으로 추천하는 것이 아닙니다."
    )
    return "\n\n".join(comments)


# ===================== 재무 분석 (요약표 + 한국어) ===================== #

def load_fundamentals(ticker_obj) -> dict:
    data = {}
    try:
        data["financials"] = ticker_obj.financials
    except Exception:
        data["financials"] = pd.DataFrame()

    try:
        data["balance_sheet"] = ticker_obj.balance_sheet
    except Exception:
        data["balance_sheet"] = pd.DataFrame()

    try:
        data["cashflow"] = ticker_obj.cashflow
    except Exception:
        data["cashflow"] = pd.DataFrame()

    return data


def build_financial_summary(fin: pd.DataFrame, info: dict) -> pd.DataFrame:
    """
    초보자가 보기 편한 한국어 재무 요약표 생성
    (매출, 영업이익, 순이익 + PER, ROE 등)
    """
    rows = []

    if fin is not None and not fin.empty:
        col = fin.columns[0]  # 가장 최근 열

        def get_row(name):
            return fin.loc[name, col] if name in fin.index else None

        revenue = get_row("Total Revenue") or get_row("Revenue")
        op_income = get_row("Operating Income")
        net_income = get_row("Net Income")

        if revenue is not None:
            rows.append(["매출(최근 연도)", f"{revenue:,.0f}", "기업이 벌어들인 총 매출 규모"])
        if op_income is not None:
            rows.append(["영업이익(최근 연도)", f"{op_income:,.0f}", "본업에서 벌어들인 이익"])
        if net_income is not None:
            rows.append(["당기순이익(최근 연도)", f"{net_income:,.0f}", "세금 등 모두 반영한 최종 이익"])

        if revenue and op_income:
            op_margin = op_income / revenue
            rows.append(["영업이익률", f"{op_margin * 100:.2f}%", "매출 대비 영업이익 비율"])
        if revenue and net_income:
            net_margin = net_income / revenue
            rows.append(["순이익률", f"{net_margin * 100:.2f}%", "매출 대비 순이익 비율"])
    else:
        rows.append(["재무제표", "데이터 없음", "야후에서 재무 데이터를 가져오지 못했습니다."])

    # 요약 지표(info)
    pe = info.get("trailingPE")
    roe = info.get("returnOnEquity")
    pb = info.get("priceToBook")
    div_yield = info.get("dividendYield")

    if pe:
        rows.append(["PER", f"{pe:.2f}", "주가수익비율(낮을수록 상대적으로 저평가일 수 있음)"])
    if roe:
        rows.append(["ROE", f"{roe * 100:.2f}%", "자기자본이익률(높을수록 수익성 우수)"])
    if pb:
        rows.append(["PBR", f"{pb:.2f}", "주가순자산비율(1배 근처는 보통, 높을수록 프리미엄)"])
    if div_yield:
        rows.append(["배당수익률", f"{div_yield * 100:.2f}%", "배당 기준 연 수익률"])

    df = pd.DataFrame(rows, columns=["항목", "값", "설명"])
    return df


def financial_comment_text(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "재무 요약 정보를 불러오지 못했습니다."

    lines = []
    roe_row = summary_df[summary_df["항목"] == "ROE"]
    per_row = summary_df[summary_df["항목"] == "PER"]
    margin_row = summary_df[summary_df["항목"] == "영업이익률"]

    if not roe_row.empty:
        roe_val = float(roe_row["값"].iloc[0].replace("%", ""))
        if roe_val > 15:
            lines.append("ROE가 15% 이상으로, **자기자본 수익성이 우수한 편**입니다.")
        elif roe_val < 5:
            lines.append("ROE가 5% 이하로, 수익성이 다소 낮은 편일 수 있습니다.")

    if not per_row.empty:
        per_val = float(per_row["값"].iloc[0])
        if per_val < 10:
            lines.append("PER가 10배 이하로, **이익 대비 주가 수준이 낮은 편(저평가 가능성)**일 수 있습니다.")
        elif per_val > 30:
            lines.append("PER가 30배 이상으로, **성장성 기대가 반영된 고평가 구간**일 수 있습니다.")

    if not margin_row.empty:
        m_val = float(margin_row["값"].iloc[0].replace("%", ""))
        if m_val > 20:
            lines.append("영업이익률이 20% 이상으로, **본업 경쟁력이 상당히 높은 편**입니다.")
        elif m_val < 5:
            lines.append("영업이익률이 5% 이하로, 경쟁이 치열하거나 수익성이 낮은 사업일 수 있습니다.")

    lines.append(
        "\n⚠️ 단일 재무 지표만으로는 기업의 모든 가치를 판단하기 어렵습니다. "
        "여러 해 추세와 동종업계 비교를 함께 보는 것이 좋습니다."
    )
    return "\n".join(lines)


# ===================== 섹터/테마 분석 ===================== #

def analyze_sector_theme(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()

    growth = ["technology", "software", "semiconductor", "internet", "ai", "communication"]
    green = ["renewable", "clean", "solar", "wind", "green"]
    health = ["health", "biotech", "pharmaceutical", "medical"]

    text = []

    if any(k in s or k in i for k in growth):
        text.append("이 회사는 **성장 섹터(IT/테크/AI 등)**에 속해 있어, 현재 시장 트렌드와 맞물려 관심을 받기 좋은 편입니다.")
    elif any(k in s or k in i for k in green):
        text.append("이 회사는 **친환경/에너지 전환 관련 섹터**에 속해 있어, 중장기 정책/테마와 연결될 수 있는 분야입니다.")
    elif any(k in s or k in i for k in health):
        text.append("이 회사는 **헬스케어/바이오 섹터**에 속해 있어, 인구 구조 변화와 의료 수요 증가 측면에서 장기 성장 테마를 가질 수 있습니다.")
    else:
        text.append("섹터/업종만 놓고 보면 '요즘 테마'에 딱 맞는 유형은 아닐 수 있으나, 개별 기업의 경쟁력과 밸류에이션을 중심으로 볼 필요가 있습니다.")

    text.append(
        "다만, 섹터가 좋다고 해서 모든 기업이 좋은 것은 아니며, "
        "실적(매출/이익), 재무구조, 경쟁 환경 등을 함께 검토하는 것이 중요합니다."
    )
    return "\n\n".join(text)


# ===================== 투자 적합도 (1~5단계) ===================== #

def assess_investment_suitability(info, perf, ind_df, fin_summary, news_pos, news_neg):
    """
    여러 요소를 단순 규칙으로 합쳐 1~5점 적합도 점수 계산 (학습용).
    절대 실제 투자 판단용으로 쓰면 안 됨.
    """
    score = 3.0  # 중립에서 시작

    # 시가총액 (크면 안정성 가중치 +)
    mcap = info.get("marketCap")
    if mcap:
        if mcap > 1e11:  # 아주 대형
            score += 0.5
        elif mcap > 1e10:
            score += 0.3

    # 기간 수익률 / 변동성 / 최대낙폭
    tr = perf.get("total_return")
    vol = perf.get("annual_vol")
    dd = perf.get("max_dd")

    if tr is not None:
        if tr > 0.5:
            score += 0.7
        elif tr > 0.1:
            score += 0.4
        elif tr < -0.3:
            score -= 0.7
        elif tr < -0.1:
            score -= 0.4

    if vol is not None:
        if vol > 0.6:
            score -= 0.5
        elif vol < 0.2:
            score += 0.2

    if dd is not None:
        if dd < -0.5:
            score -= 0.6
        elif dd > -0.2:
            score += 0.2

    # 차트 추세 / RSI
    cols = [c for c in ["SMA_20", "SMA_60", "RSI_14"] if c in ind_df.columns]
    clean = ind_df[cols].dropna()
    if not clean.empty:
        latest = clean.iloc[-1]
        if latest["SMA_20"] > latest["SMA_60"]:
            score += 0.3
        else:
            score -= 0.2
        rsi = latest["RSI_14"]
        if rsi > 70:
            score -= 0.3
        elif rsi < 30:
            score += 0.1

    # 재무 (당기순이익, ROE)
    if not fin_summary.empty:
        ni_row = fin_summary[fin_summary["항목"] == "당기순이익(최근 연도)"]
        if not ni_row.empty:
            try:
                ni_val = float(ni_row["값"].iloc[0].replace(",", ""))
                if ni_val > 0:
                    score += 0.4
                else:
                    score -= 0.6
            except Exception:
                pass

        roe_row = fin_summary[fin_summary["항목"] == "ROE"]
        if not roe_row.empty:
            try:
                roe_val = float(roe_row["값"].iloc[0].replace("%", ""))
                if roe_val > 15:
                    score += 0.4
                elif roe_val < 5:
                    score -= 0.3
            except Exception:
                pass

    # 뉴스 분위기
    if news_pos > news_neg:
        score += 0.2
    elif news_neg > news_pos:
        score -= 0.2

    # 범위 제한
    score = max(1.0, min(5.0, score))

    if score >= 4.5:
        grade = "매우 적합 (5단계 중 상위)"
    elif score >= 3.8:
        grade = "비교적 적합"
    elif score >= 3.0:
        grade = "보통 (중립)"
    elif score >= 2.0:
        grade = "비교적 부적합"
    else:
        grade = "매우 부적합"

    detail = (
        "⚠️ 이 적합도 평가는 간단한 규칙 기반 **학습용 점수**입니다. "
        "실제 매수·매도 의사결정에 직접 사용하면 안 되며, "
        "반드시 추가적인 기업 분석과 본인의 판단이 필요합니다."
    )
    return {"score": score, "grade": grade, "detail": detail}


# ===================== Streamlit UI 구성 ===================== #

def show_help_tab():
    st.subheader("❓ 전체 사용법 & 기본 개념 정리")

    st.markdown(
        """
        ### 1. 티커(symbol) 입력 방법
        - 미국 주식: `AAPL`, `TSLA`, `MSFT` 처럼 심볼만 입력  
        - 한국 주식: `종목코드 + .KS(코스피)`, `.KQ(코스닥)`  
          - 삼성전자 → `005930.KS`  
          - SK하이닉스 → `000660.KS`  

        ### 2. 각 탭의 의미
        - **📌 개요**: 회사 기본 정보, 섹터/업종, 테마 관점 의견, 투자 적합도 요약  
        - **💰 재무 분석**: 한국어 재무 요약 표 + 해설  
        - **📈 차트/기술적**: 종가 + 이동평균선, RSI 차트 & 기술적 관점 설명과 해석법  
        - **📰 뉴스 & 전망**: 네이버 + 야후 뉴스 헤드라인과 분위기 요약  

        ### 3. 반드시 기억할 점
        - 이 도구는 어디까지나 **학습/연습용 보조 도구**입니다.  
        - 과거 데이터와 단순 지표만으로는 미래 수익을 보장할 수 없습니다.  
        - 실제 투자 전에는 본인이 추가 리서치와 고민을 충분히 한 뒤 판단해야 합니다.
        """
    )


def show_summary_header(ticker: str, info: dict, price_df: pd.DataFrame, perf: dict):
    last_close = float(price_df["Close"].iloc[-1])
    first_close = float(price_df["Close"].iloc[0])
    total_return = (last_close / first_close - 1) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("종목 코드", ticker)
    col2.metric("최근 종가", f"{last_close:,.2f}")
    col3.metric("기간 수익률", f"{total_return:.2f}%")

    mcap = info.get("marketCap")
    col4.metric("시가총액(추정)", f"{mcap:,.0f}" if mcap else "정보 없음")

    st.markdown(
        f"**{info.get('longName', info.get('shortName', '회사명 정보 없음'))}**  \n"
        f"섹터: {info.get('sector', '정보 없음')} / 업종: {info.get('industry', '정보 없음')}"
    )

    if perf.get("annual_vol") is not None and perf.get("max_dd") is not None:
        st.caption(
            f"- 연간 변동성(단순 추정): 약 {perf['annual_vol'] * 100:.2f}%  "
            f"/ 최대 낙폭: {perf['max_dd'] * 100:.2f}%"
        )


def show_suitability_section(suitability: dict):
    st.subheader("📊 투자 적합도 요약 (1~5단계, 학습용)")

    score = suitability["score"]
    grade = suitability["grade"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("적합도 점수 (1~5)", f"{score:.1f}")
        st.write(grade)
    with col2:
        df = pd.DataFrame({"적합도": [score]}, index=["점수"])
        st.bar_chart(df, use_container_width=True)

    st.caption(suitability["detail"])


def show_overview_tab(info: dict, user_level_kor: str, suitability: dict):
    st.subheader("📌 회사 개요")

    st.write(f"- 회사명: **{info.get('longName', info.get('shortName', '정보 없음'))}**")
    st.write(f"- 섹터(Sector): **{info.get('sector', '정보 없음')}**")
    st.write(f"- 업종(Industry): **{info.get('industry', '정보 없음')}**")

    st.markdown("---")
    st.subheader("🌐 섹터/테마 관점 코멘트")
    st.write(
        analyze_sector_theme(
            info.get("sector", ""),
            info.get("industry", "")
        )
    )

    st.markdown("---")
    show_suitability_section(suitability)

    if user_level_kor == "초보자":
        st.info("초보자 팁: '어떤 회사인가?'를 먼저 이해한 뒤, 재무제표와 차트를 함께 보는 것이 좋습니다.")


def show_financial_tab(fundamentals: dict, fin_summary: pd.DataFrame, fin_comment: str):
    st.subheader("💰 재무 요약 표 (한국어)")

    st.table(fin_summary)

    st.markdown("### 재무 해석 코멘트")
    st.write(fin_comment)

    # 필요하면 원본 재무제표도 펼쳐 볼 수 있게
    fin = fundamentals.get("financials", pd.DataFrame())
    bs = fundamentals.get("balance_sheet", pd.DataFrame())
    cf = fundamentals.get("cashflow", pd.DataFrame())

    with st.expander("원본 재무제표(영문)도 보고 싶다면 클릭"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**손익계산서 (일부)**")
            if fin is not None and not fin.empty:
                st.dataframe(fin.head(12))
            else:
                st.write("손익계산서 데이터를 불러오지 못했습니다.")
        with col2:
            st.markdown("**재무상태표 (일부)**")
            if bs is not None and not bs.empty:
                st.dataframe(bs.head(12))
            else:
                st.write("재무상태표 데이터를 불러오지 못했습니다.")

        st.markdown("**현금흐름표 (일부)**")
        if cf is not None and not cf.empty:
            st.dataframe(cf.head(12))
        else:
            st.write("현금흐름표 데이터를 불러오지 못했습니다.")


def show_chart_tab(price_df: pd.DataFrame):
    st.subheader("📈 가격 + 이동평균선 차트")
    chart_df = price_df[["Close", "SMA_20", "SMA_60"]].dropna()
    st.line_chart(chart_df)

    st.subheader("📉 RSI(14)")
    st.line_chart(price_df[["RSI_14"]])

    st.markdown("### 이번 종목 차트 요약 분석")
    tech_text = comment_from_technicals(price_df)
    st.write(tech_text)

    st.markdown("### 차트 해석 기초 (도움말)")
    with st.expander("차트와 지표를 어떻게 해석하면 좋은지 궁금하다면 펼쳐보세요."):
        st.markdown(
            """
            - **종가 + 이동평균선**
              - 20일선이 60일선 위에 있으면 단기적으로 상승 추세 쪽으로 해석하는 경우가 많습니다.  
              - 20일선이 60일선 아래에 있으면 단기/중기 조정 또는 하락 추세일 가능성을 봅니다.  
            - **RSI(14)**  
              - 70 이상: 과열/과매수 구간 → 단기 조정 가능성 주의  
              - 30 이하: 과매도 구간 → 기술적 반등 가능성을 참고  
            - 지표는 어디까지나 **보조수단**이며,  
              실적·뉴스·펀더멘털과 함께 종합적으로 판단해야 합니다.
            """
        )


def show_news_tab(combined_news, news_summary_text: str, naver_msg: str):
    st.subheader("📰 뉴스 & 전망")

    if naver_msg:
        st.caption(naver_msg)

    st.write(news_summary_text)

    st.markdown("---")
    st.subheader("개별 뉴스 목록")

    if combined_news:
        for i, n in enumerate(combined_news, start=1):
            st.markdown(f"**{i}. {n['title']}** ({n['source']})")
            if n["summary"]:
                st.write(n["summary"])
            if n["link"]:
                st.markdown(f"[기사 링크 열기]({n['link']})")
            st.markdown("---")
    else:
        st.write("표시할 뉴스가 없습니다.")

    st.info(
        "뉴스와 요약, 분위기 분석은 어디까지나 참고용이며, "
        "실제 투자 판단은 반드시 본인의 추가 분석과 판단에 의해 이루어져야 합니다."
    )


def main():
    st.set_page_config(page_title="주식 AI 분석 도우미", layout="wide")
    st.title("📊 주식 AI 분석 도우미 (한국/미국 + 뉴스/재무/차트/적합도)")

    st.markdown(
        """
        이 도구는 **초보자부터 전문가까지** 모두가 참고할 수 있도록 만든  
        주식 **기본 정보 + 재무 + 차트 + 뉴스 + 섹터/테마 + 투자 적합도(학습용)** 분석 보조 툴입니다.  

        ⚠️ **중요 안내**  
        - 이 서비스는 공부/연습용이며, 특정 종목에 대한 매수/매도 추천이 아닙니다.  
        - 모든 투자 결정과 책임은 전적으로 사용자 본인에게 있습니다.
        """
    )

    st.sidebar.header("🔧 설정 & 사용법")

    user_level_kor = st.sidebar.radio(
        "사용자 수준",
        ["초보자", "전문가"],
        index=0,
        help="표현 방식과 도움말의 친절함 정도가 달라집니다."
    )
    level = "beginner" if user_level_kor == "초보자" else "expert"

    st.sidebar.markdown(
        """
        **티커(symbol) 입력 예시**

        - 미국:  
          - 애플 → `AAPL`  
          - 테슬라 → `TSLA`  

        - 한국(야후 파이낸스 형식):  
          - 삼성전자 → `005930.KS`  
          - SK하이닉스 → `000660.KS`  
        """
    )

    ticker = st.sidebar.text_input(
        "종목 티커(symbol)",
        value="AAPL",
        help="한국 종목은 종목코드 + .KS / .KQ 형식으로 입력합니다. 예: 005930.KS"
    )

    period = st.sidebar.selectbox(
        "차트 기간",
        options=["1y", "3y", "5y"],
        index=1,
        help="과거 몇 년치 주가를 기준으로 차트/성과를 볼지 선택합니다."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**사용 순서**\n\n1. 종목 티커 입력\n2. 기간 선택\n3. 아래 [분석 실행] 버튼 클릭")

    run = st.sidebar.button("🔍 분석 실행")

    if not run:
        st.info("왼쪽 사이드바에서 **티커와 기간을 설정한 뒤 [분석 실행] 버튼을 눌러주세요.**")
        show_help_tab()
        return

    # 데이터 로딩
    try:
        with st.spinner("데이터 불러오는 중..."):
            tk = get_ticker_obj(ticker)
            info = get_info_safe(tk)
            price = load_price_history(ticker, period=period)
            price = add_indicators(price)
            perf = calc_perf_stats(price)
            fundamentals = load_fundamentals(tk)
            fin_summary = build_financial_summary(fundamentals.get("financials", pd.DataFrame()), info)
            fin_comment = financial_comment_text(fin_summary)

        with st.spinner("뉴스 불러오는 중..."):
            company_name = info.get("longName", info.get("shortName", ticker))
            naver_news, naver_msg = get_naver_news(company_name)
            yf_news = get_yf_news(tk)
            combined_news = naver_news + yf_news
            news_summary_text, news_pos, news_neg = summarize_news_combined(combined_news)

        suitability = assess_investment_suitability(info, perf, price, fin_summary, news_pos, news_neg)

        st.success("✅ 데이터 불러오기 완료")
    except Exception as e:
        st.error(f"데이터 불러오는 중 에러 발생: {e}")
        return

    show_summary_header(ticker, info, price, perf)

    tab_overview, tab_fin, tab_chart, tab_news, tab_help = st.tabs(
        ["📌 개요", "💰 재무 분석", "📈 차트/기술적", "📰 뉴스 & 전망", "❓ 도움말"]
    )

    with tab_overview:
        show_overview_tab(info, user_level_kor, suitability)

    with tab_fin:
        show_financial_tab(fundamentals, fin_summary, fin_comment)

    with tab_chart:
        show_chart_tab(price)

    with tab_news:
        show_news_tab(combined_news, news_summary_text, naver_msg)

    with tab_help:
        show_help_tab()


if __name__ == "__main__":
    main()
