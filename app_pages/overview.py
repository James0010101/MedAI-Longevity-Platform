"""
app_pages/overview.py — Landing page: promo video, key facts, features and a call to action.
"""
import streamlit as st
from core.branding import BRAND_BLUE, BRAND_CYAN, BRAND_NAVY, PROMO_VIDEO_PATH, logo_data_uri
from core.i18n import get_current_language, t

lang = get_current_language()

st.markdown(
    f"""
<style>
.ov-hero {{ background: linear-gradient(135deg, {BRAND_NAVY} 0%, #25418A 55%, {BRAND_BLUE} 100%);
  border-radius: 26px; padding: 38px 42px; color: #fff; position: relative; overflow: hidden;
  box-shadow: 0 18px 44px rgba(30,47,82,.28); }}
.ov-hero::after {{ content:""; position:absolute; right:-80px; top:-80px; width:340px; height:340px; border-radius:50%;
  background: radial-gradient(circle, {BRAND_CYAN}66 0%, transparent 70%); }}
.ov-hero-top {{ display:flex; align-items:center; gap:18px; position:relative; z-index:1; }}
.ov-hero-top img {{ filter: drop-shadow(0 6px 16px rgba(0,0,0,.35)); }}
.ov-hero h1 {{ color:#fff; margin:0; padding:0; font-size:2.3rem; line-height:1.1; letter-spacing:-1px; }}
.ov-pill {{ display:inline-block; margin:22px 0 12px; padding:7px 16px; border-radius:30px; font-weight:700; font-size:.85rem;
  background: rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3); position:relative; z-index:1; }}
.ov-tag {{ font-size:1.3rem; font-weight:500; line-height:1.4; max-width:760px; color:#E6EEFF; position:relative; z-index:1; }}
.ov-step {{ margin:34px 0 12px; font-size:.95rem; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:{BRAND_BLUE}; }}
.ov-stats {{ display:flex; gap:16px; margin:26px 0 6px; flex-wrap:wrap; }}
.ov-stat {{ flex:1; min-width:170px; background:#fff; border-radius:20px; padding:20px 22px; text-align:center;
  box-shadow:0 8px 26px rgba(30,47,82,.08); border:1px solid #E3ECF8; }}
.ov-stat b {{ display:block; font-size:2.1rem; font-weight:800; letter-spacing:-1px;
  background:linear-gradient(135deg,{BRAND_BLUE},{BRAND_CYAN}); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.ov-stat span {{ font-size:.9rem; color:#5B7399; font-weight:600; }}
.ov-feat {{ background:#fff; border-radius:20px; padding:22px; min-height:270px; border:1px solid #E3ECF8;
  box-shadow:0 8px 26px rgba(30,47,82,.07); }}
.ov-ico {{ width:52px; height:52px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:1.6rem;
  background:linear-gradient(135deg,{BRAND_CYAN}33,{BRAND_BLUE}26); margin-bottom:12px; }}
.ov-feat h4 {{ margin:0 0 8px; font-size:1.1rem; color:{BRAND_NAVY}; }}
.ov-feat p {{ margin:0; color:#34496E; font-size:.95rem; line-height:1.5; }}
[data-testid="stVideo"] {{ border-radius:22px; overflow:hidden; box-shadow:0 18px 44px rgba(30,47,82,.22); border:6px solid #fff; }}
[data-testid="stPageLink"] a {{ background:linear-gradient(135deg,{BRAND_BLUE},#4A86FF); color:#fff !important; justify-content:center;
  padding:16px 24px; border-radius:16px; box-shadow:0 10px 26px rgba(47,107,255,.38); font-size:1.15rem; }}
[data-testid="stPageLink"] a p {{ color:#fff; font-weight:800; font-size:1.15rem; }}
[data-testid="stPageLink"] a:hover {{ filter:brightness(1.08); }}
</style>
<div class="ov-hero">
  <div class="ov-hero-top"><img src="{logo_data_uri()}" width="68" height="68" alt="MedAI Longevity logo"/>
  <h1>{t('app_title', lang=lang)}</h1></div>
  <div class="ov-pill">🎬 {t('ov_badge', lang=lang)}</div>
  <div class="ov-tag">{t('ov_tagline', lang=lang)}</div>
</div>
<div class="ov-step">▶ {t('ov_watch', lang=lang)}</div>
""",
    unsafe_allow_html=True,
)

_, col_video, _ = st.columns([1, 10, 1])
with col_video:
    st.video(str(PROMO_VIDEO_PATH))
    st.caption(t("ov_video_note", lang=lang))

stats = [("4 240", "ov_s1"), ("3", "ov_s2"), ("41", "ov_s3"), ("A4", "ov_s4")]
st.markdown(
    '<div class="ov-stats">'
    + "".join(f'<div class="ov-stat"><b>{n}</b><span>{t(k, lang=lang)}</span></div>' for n, k in stats)
    + "</div>",
    unsafe_allow_html=True,
)

st.write("")
features = [("ov_f1", "🫀"), ("ov_f2", "🔍"), ("ov_f3", "🎯"), ("ov_f4", "📄")]
for col, (key, icon) in zip(st.columns(len(features)), features):
    col.markdown(
        f'<div class="ov-feat"><div class="ov-ico">{icon}</div><h4>{t(key + "_t", lang=lang)}</h4>'
        f'<p>{t(key + "_d", lang=lang)}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(f'<div class="ov-step">🩺 {t("ov_then", lang=lang)}</div>', unsafe_allow_html=True)
_, col_cta, _ = st.columns([1, 2, 1])
with col_cta:
    st.page_link("app_pages/screening.py", label=t("ov_cta", lang=lang), icon="🩺", use_container_width=True)
    st.caption(t("ov_cta_hint", lang=lang))

st.divider()
st.caption(t("ov_author", lang=lang))
st.caption(t("ov_disclaimer", lang=lang))
