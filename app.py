import streamlit as st

st.set_page_config(page_title="あなたの商品、韓国では？", page_icon="🇰🇷", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 720px; padding-top: 2.2rem;}
h1 {font-size: 2.25rem !important; line-height: 1.25 !important;}
.sub {font-size: 1.08rem; color: #555; margin-bottom: 1.6rem;}
.step {font-weight: 700; color: #444; margin-top: 1.4rem;}
</style>
""", unsafe_allow_html=True)

st.caption("KOREA MARKET FINDER")
st.title("あなたの商品、韓国では？")
st.markdown('<div class="sub">商品を撮影すると、韓国で販売されている類似商品や市場情報を探します。</div>', unsafe_allow_html=True)

st.markdown('<div class="step">01　商品を撮影</div>', unsafe_allow_html=True)
photo = st.camera_input("商品の正面がよく見えるように撮影してください")

if photo:
    st.success("撮影しました。")
    st.markdown('<div class="step">02　韓国の類似商品を探す</div>', unsafe_allow_html=True)
    st.info("次のステップで Gemini と Google 検索を接続します。")

st.divider()
st.caption("KEA デジタル流通センター")
