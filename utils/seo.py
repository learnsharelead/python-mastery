import streamlit as st
import json

def inject_seo_meta(title, description, keywords=None, schema_type="TechArticle"):
    """Injects JSON-LD Structured Data and sets page config."""
    st.set_page_config(
        page_title=title,
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    schema = {
        "@context": "https://schema.org",
        "@type": schema_type,
        "headline": title,
        "description": description,
        "author": {"@type": "Organization", "name": "Python Mastery Hub"},
        "publisher": {"@type": "Organization", "name": "Python Mastery Hub"}
    }
    
    if keywords:
        schema["keywords"] = keywords

    st.markdown(f"""
    <script type="application/ld+json">{json.dumps(schema)}</script>
    <div style="display:none;"><h1>{title}</h1><p>{description}</p></div>
    """, unsafe_allow_html=True)
