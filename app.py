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


class ProductAnalysis(BaseModel):
    product_name: str
    category_ja: str
    category_ko: str
    manufacturer: str
    volume: str
    features_ja: list[str]
    shopping_query_ko: str


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


def search_korean_products(query: str, num: int = 10) -> list[dict]:
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
                results = search_korean_products(analysis.shopping_query_ko)
                st.session_state.analysis = analysis
                st.session_state.shopping_results = results
            st.rerun()
        except Exception as e:
            st.error("検索中にエラーが発生しました。")
            st.code(str(e))


if st.session_state.analysis:
    a = st.session_state.analysis

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
    st.subheader("🇰🇷 韓国では、こんな商品があります")

    results = st.session_state.shopping_results[:5]

    if not results:
        st.warning("類似商品を見つけられませんでした。別の写真でお試しください。")
    else:
        st.caption("Google Shopping の韓国向け検索結果を表示しています。価格・販売状況は販売先でご確認ください。")

        for item in results:
            title = item.get("title", "商品")
            source = item.get("source", "") or item.get("seller", "")
            price = normalize_price(item.get("price"))
            image_url = product_image_url(item)
            link = product_link(item)

            c1, c2 = st.columns([1, 2.3])
            with c1:
                if image_url:
                    st.image(image_url, width="stretch")
            with c2:
                st.markdown(f"**{title}**")
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
