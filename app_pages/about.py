"""
app_pages/about.py — Project Overview, Societal Problem, Solution & Educational Startup Context.
"""
import streamlit as st
from core.i18n import get_current_language, t

lang = get_current_language()
st.title(f"ℹ️ {t('about_title', lang=lang)}")
st.caption(t("about_mission", lang=lang))

# 1. Problem Statement
with st.container(border=True):
    st.subheader(t("about_sec_1_title", lang=lang))
    st.markdown(t("about_sec_1_text", lang=lang))

# 2. Solution & Technology
with st.container(border=True):
    st.subheader(t("about_sec_2_title", lang=lang))
    st.markdown(t("about_sec_2_text", lang=lang))

# 3. Roadmap & Implementation
with st.container(border=True):
    st.subheader(t("about_sec_3_title", lang=lang))
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(f"##### {t('about_phase_1_title', lang=lang)}")
        st.write(t("about_phase_1_item_1", lang=lang))
        st.write(t("about_phase_1_item_2", lang=lang))
        st.write(t("about_phase_1_item_3", lang=lang))
        st.write(t("about_phase_1_item_4", lang=lang))
    with col_r2:
        st.markdown(f"##### {t('about_phase_2_title', lang=lang)}")
        st.write(t("about_phase_2_item_1", lang=lang))
        st.write(t("about_phase_2_item_2", lang=lang))
        st.write(t("about_phase_2_item_3", lang=lang))
    with col_r3:
        st.markdown(f"##### {t('about_phase_3_title', lang=lang)}")
        st.write(t("about_phase_3_item_1", lang=lang))
        st.write(t("about_phase_3_item_2", lang=lang))
        st.write(t("about_phase_3_item_3", lang=lang))

# 4. Project Team & Institution
with st.container(border=True):
    st.subheader(t("about_sec_4_title", lang=lang))
    st.caption(t("about_team_agency", lang=lang))
    st.caption(t("about_team_contest", lang=lang))
    st.divider()
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"##### {t('about_author_title', lang=lang)}")
        st.markdown(t("about_author_name", lang=lang))
        st.markdown(t("about_author_inst", lang=lang))
        st.markdown(t("about_author_field", lang=lang))
    with col_t2:
        st.markdown(f"##### {t('about_mentor_title', lang=lang)}")
        st.markdown(t("about_mentor_name", lang=lang))
        st.markdown(t("about_mentor_contact", lang=lang))
        st.markdown(t("about_mentor_repo", lang=lang))

st.divider()
st.caption(t("footer_disclaimer", lang=lang))
