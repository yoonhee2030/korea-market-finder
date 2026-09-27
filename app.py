import io
import os
import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel

st.set_page_config(
    page_title="あなたの商品、韓国では？",
    page_icon="🇰🇷",
    layout="centered",
)

st.markdown("""
<style>
.block-container {max-width: 720px; padding-top: 1.6rem; padding-bottom: 3rem;}
h1 {font-size: 2.2rem !important; line-height: 1.24 !important;}
.sub {font-size: 1.03rem; color: #56616f; line-height: 1.7; margin-bottom: 1.5rem;}
.step {font-size: .82rem; font-weight: 800; letter-spacing: .08em; color: #d94435; margin-top: 1.4rem;}
.product-card {border:1px solid #e4e8ee; border-radius:16px; padding:14px; margin:10px 0; background:white;}
.meta {color:#66717f; font-size:.88rem;}
.price {font-size:1.15rem; font-weight:800; color:#c83227;}
.footer {text-align:center; color:#7a8491; font-size:.82rem; padding-top:2rem;}
</style>
""", unsafe_allow_html=True)


class JapaneseProductText(BaseModel):
    title_ja: str
    source_ja: str


class MarketingCopy(BaseModel):
    headline_ja: str
    keywords_ja: list[str]
    copy_ja: str


class ProductSelection(BaseModel):
    indices: list[int]


class ProductAnalysis(BaseModel):
    product_name: str
    category_ja: str
    category_ko: str
    manufacturer: str
    volume: str
    features_ja: list[str]
    shopping_query_ko: str
    shopping_queries_ko: list[str]


def get_secret(name: str) -> str:
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        pass
    return value or os.getenv(name, "")


def compress_image(data: bytes, max_side: int = 1600, quality: int = 82) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def analyze_japanese_product(photo_bytes: bytes) -> ProductAnalysis:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません。")

    client = genai.Client(api_key=api_key)

    prompt = """
あなたは日本の商品を韓国市場で比較するための食品リサーチアシスタントです。
添付画像から、写真で確認できる範囲だけで商品を整理してください。
不明な情報は空文字にしてください。誇張や推測はしないでください。

shopping_query_ko は、韓国で実際に販売されている類似商品を
Google Shoppingで探すための自然な韓国語検索語にしてください。
ブランド名そのものではなく、商品カテゴリ・味・形態・用途など
類似性の高い要素を優先してください。

返す項目:
- product_name: 日本語の商品名
- category_ja: 日本語の商品カテゴリー
- category_ko: 韓国語の商品カテゴリー
- manufacturer: メーカー/販売者
- volume: 内容量
- features_ja: 写真から確認できる主な特徴 最大4個
- shopping_query_ko: 韓国語のショッピング検索語 1つ
- shopping_queries_ko: ブランド名を避け、カテゴリ・味・形態・用途を変えた韓国語検索語を3つ
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=photo_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProductAnalysis,
        ),
    )
    return ProductAnalysis.model_validate_json(response.text)


def make_marketing_copy(a: ProductAnalysis) -> MarketingCopy:
    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    prompt = f"""日本の食品メーカー向けに、次の商品が韓国市場で訴求する場合に使える広告・検索キーワード案を日本語で作成してください。
実際の広告実績だと断定せず、商品の写真から確認できた特徴と韓国での検索カテゴリーをもとにした「訴求キーワード案」としてください。
短く、展示会で一目で分かる表現にしてください。
商品: {a.product_name}
カテゴリー: {a.category_ja}
特徴: {", ".join(a.features_ja)}
韓国検索カテゴリー: {a.category_ko}
headline_ja: 12文字程度
keywords_ja: 4〜6個、各10文字以内
copy_ja: 1文、35文字以内
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=MarketingCopy),
    )
    return MarketingCopy.model_validate_json(response.text)


def search_korean_products(query: str, num: int = 15) -> list[dict]:
    api_key = get_secret("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY が設定されていません。")

    response = requests.post(
        "https://google.serper.dev/shopping",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        json={
            "q": query,
            "gl": "kr",
            "hl": "ko",
            "num": num,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("shopping", []) or []


def collect_candidates(queries: list[str]) -> list[dict]:
    merged = []
    seen = set()

    for query in queries:
        for item in search_korean_products(query, 12):
            title = str(item.get("title", "")).strip()
            source = str(item.get("source", "") or item.get("seller", "")).strip()
            image = str(item.get("imageUrl") or item.get("image") or item.get("thumbnail") or "").strip()

            key = (title.lower(), source.lower(), image)
            if not title or key in seen:
                continue

            seen.add(key)
            merged.append(item)

    return merged[:35]


def select_diverse_similar_products(a: ProductAnalysis, candidates: list[dict], limit: int = 10) -> list[dict]:
    if not candidates:
        return []

    compact = []
    for idx, item in enumerate(candidates):
        compact.append({
            "index": idx,
            "title": item.get("title", ""),
            "source": item.get("source", "") or item.get("seller", ""),
            "price": item.get("price", ""),
        })

    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    prompt = f"""あなたは韓国食品市場の比較担当者です。
日本の商品と比較するため、候補から最大{limit}件を選んでください。

元の商品:
商品名: {a.product_name}
メーカー: {a.manufacturer}
カテゴリー: {a.category_ja}
特徴: {", ".join(a.features_ja)}

選定ルール:
1. 食品だけを選ぶ。玩具、雑貨、容器、調理器具は除外。
2. 元の商品と同一商品・同一ブランドの輸入販売ページは原則除外。
3. 同じブランド・同じシリーズ・同じセット商品の重複は1件まで。
4. カテゴリー、味、形状、食べ方が近いものを優先。
5. できるだけ異なるブランドの商品を選び、市場の選択肢が分かるようにする。
6. 韓国で販売されている国内・輸入ブランドの両方を含めてよい。
7. 明らかに同じ商品の容量違い・セット数違いは重複扱い。

候補:
{compact}

indices には候補の index を、良い順に最大{limit}個返してください。
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ProductSelection,
        ),
    )
    selected = ProductSelection.model_validate_json(response.text).indices

    result = []
    used = set()
    for idx in selected:
        if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in used:
            result.append(candidates[idx])
            used.add(idx)
        if len(result) >= limit:
            break

    return result


def translate_product_text(title: str, source: str) -> JapaneseProductText:
    api_key = get_secret("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = f"""以下の韓国ショッピング検索結果を、日本の食品メーカーが自然に読める日本語にしてください。
商品名はブランド名・数量・容量・味など重要情報を残してください。
販売元名は固有のサービス名（Coupang、AliExpressなど）はそのまま、説明的な韓国語だけ日本語にしてください。
説明や補足は不要です。

商品名: {title}
販売元: {source}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JapaneseProductText,
        ),
    )
    return JapaneseProductText.model_validate_json(response.text)


@st.cache_data(ttl=3600, show_spinner=False)
def japanese_result_text(title: str, source: str) -> tuple[str, str]:
    try:
        translated = translate_product_text(title, source)
        return translated.title_ja or title, translated.source_ja or source
    except Exception:
        return title, source


@st.cache_data(ttl=21600, show_spinner=False)
def krw_to_jpy_rate() -> float | None:
    try:
        response = requests.get("https://open.er-api.com/v6/latest/KRW", timeout=10)
        response.raise_for_status()
        return float(response.json()["rates"]["JPY"])
    except Exception:
        return None


def parse_krw_price(value) -> int | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def normalize_price(value):
    if value is None:
        return ""
    return str(value).strip()


def product_image_url(item: dict) -> str:
    return (
        item.get("imageUrl")
        or item.get("image")
        or item.get("thumbnail")
        or ""
    )


def product_link(item: dict) -> str:
    return (
        item.get("link")
        or item.get("productUrl")
        or item.get("url")
        or ""
    )


st.caption("KOREA MARKET FINDER · KEA")
st.title("あなたの商品、韓国では？")
st.markdown(
    '<div class="sub">商品を1枚撮影すると、AIが特徴を確認し、'
    '韓国で販売されている類似商品を検索します。</div>',
    unsafe_allow_html=True,
)

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "shopping_results" not in st.session_state:
    st.session_state.shopping_results = []
if "marketing" not in st.session_state:
    st.session_state.marketing = None


st.markdown('<div class="step">STEP 01 · 撮る</div>', unsafe_allow_html=True)
st.subheader("商品の写真を撮影")
photo = st.camera_input("商品の正面がよく見えるように撮影してください")

if photo:
    photo_bytes = compress_image(photo.getvalue())
    st.success("撮影しました。")

    if st.button("🇰🇷 韓国の類似商品を探す", type="primary", width="stretch"):
        try:
            with st.spinner("AIが商品を確認し、韓国の商品を検索しています…"):
                analysis = analyze_japanese_product(photo_bytes)
                queries = analysis.shopping_queries_ko or [analysis.shopping_query_ko]
                candidates = collect_candidates(queries[:3])
                results = select_diverse_similar_products(analysis, candidates, 10)
                marketing = make_marketing_copy(analysis)
                st.session_state.analysis = analysis
                st.session_state.shopping_results = results
                st.session_state.marketing = marketing
            st.rerun()
        except Exception as e:
            st.error("検索中にエラーが発生しました。")
            st.code(str(e))


if st.session_state.analysis:
    a = st.session_state.analysis

    if st.session_state.marketing:
        m = st.session_state.marketing
        st.markdown('<div class="step">KOREA MARKET · 訴求キーワード</div>', unsafe_allow_html=True)
        st.subheader(m.headline_ja)
        st.markdown("　".join([f"**#{k}**" for k in m.keywords_ja]))
        st.caption("韓国市場向けの訴求キーワード案です。実際の広告出稿データではありません。")
        st.write(m.copy_ja)

    st.markdown('<div class="step">STEP 02 · 確認</div>', unsafe_allow_html=True)
    st.subheader("この商品を確認しました")

    st.markdown(
        f"""
        <div class="product-card">
          <b>{a.product_name or '商品名を確認できませんでした'}</b><br>
          <span class="meta">{a.category_ja or ''} {(' · ' + a.manufacturer) if a.manufacturer else ''} {(' · ' + a.volume) if a.volume else ''}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if a.features_ja:
        st.write(" / ".join(a.features_ja))

    with st.expander("検索キーワード"):
        st.write(a.shopping_query_ko)


    st.markdown('<div class="step">STEP 03 · 見つける</div>', unsafe_allow_html=True)
    st.subheader("🇰🇷 韓国で販売されている類似商品 10選")

    results = st.session_state.shopping_results

    if not results:
        st.warning("類似商品を見つけられませんでした。別の写真でお試しください。")
    else:
        st.caption("日本円は現在の為替レートによる参考換算です。韓国ウォンの販売価格も併記しています。価格・販売状況は販売先でご確認ください。")

        for item in results:
            title = item.get("title", "商品")
            source = item.get("source", "") or item.get("seller", "")
            title_ja, source_ja = japanese_result_text(title, source)
            price = normalize_price(item.get("price"))
            krw_price = parse_krw_price(item.get("price"))
            rate = krw_to_jpy_rate()
            jpy_price = round(krw_price * rate) if krw_price and rate else None
            image_url = product_image_url(item)
            link = product_link(item)

            c1, c2 = st.columns([1, 2.3])
            with c1:
                if image_url:
                    st.image(image_url, width="stretch")
            with c2:
                st.markdown(f"**{title_ja}**")
                if price:
                    st.markdown(f'<div class="price">{price}</div>', unsafe_allow_html=True)
                if source:
                    st.caption(source)
                if link:
                    st.link_button("韓国の販売ページを見る →", link, width="stretch")

            st.divider()


    st.markdown('<div class="step">STEP 04 · 知る</div>', unsafe_allow_html=True)
    st.subheader("韓国市場をもっと見る")
    st.info(
        "次の段階では、この商品カテゴリーの韓国市場・流通チャネル・"
        "消費トレンドなどの情報もご覧いただけるようにします。"
    )


st.markdown(
    """
    <div class="footer">
      KEA デジタル流通センター<br>
      崔 允僖（チェ・ユニ） 部長 · ian@gokea.org
    </div>
    """,
    unsafe_allow_html=True,
)
