import streamlit as st

def apply_custom_css():
    st.markdown("""
<style>
    /* ==========================================================================
       RESET & BASICS
       ========================================================================== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Remove Streamlit Bloat */
    #MainMenu, footer, header {visibility: hidden; height: 0;}
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    .stDeployButton {display: none;}
    
    /* App Background */
    .stApp {
        background-color: #f8fafc !important;
    }
    
    /* ==========================================================================
       COMPACT LAYOUT - REDUCED WHITESPACE
       ========================================================================== */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    
    header { visibility: hidden !important; height: 0 !important; }
    
    /* Reduce gap between elements */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* Reduce Markdown spacing */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    .stMarkdown p {
        margin-bottom: 0.4rem !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        padding-top: 0 !important;
    }
    
    /* Reduce divider spacing */
    hr {
        margin: 0.5rem 0 !important;
    }

    /* ==========================================================================
       TYPOGRAPHY - COMPACT
       ========================================================================== */
    h1 {
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        letter-spacing: -0.02em !important;
        color: #1e293b !important;
        margin-bottom: 0.3rem !important;
    }
    
    h2 {
        font-size: 1.2rem !important;
        color: #1e293b !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    h3 {
        font-size: 1rem !important;
        color: #334155 !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    h4 {
        font-size: 0.9rem !important;
        color: #475569 !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.1rem !important;
    }
    
    p, li, label {
        font-size: 14px !important;
        line-height: 1.4 !important;
        color: #475569 !important;
    }
    
    /* ==========================================================================
       CARDS & SURFACES - COMPACT
       ========================================================================== */
    .content-card {
        background: #ffffff !important;
        padding: 0.8rem !important;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem !important;
    }
    
    .feature-box {
        background: #ffffff !important;
        padding: 0.8rem !important;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem !important;
    }

    /* BENTO BOX COLORS - Compact */
    .bento-blue { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important; }
    .bento-purple { background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%) !important; }
    .bento-orange { background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%) !important; }
    .bento-green { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important; }
    
    .bento-icon { font-size: 24px; margin-bottom: 5px; }
    .bento-title { font-weight: 700; font-size: 15px; color: #1e3a8a; margin-bottom: 2px; }
    .bento-desc { font-size: 13px; color: #64748b; line-height: 1.3; }

    /* ==========================================================================
       TABS - COMPACT
       ========================================================================== */
    .stTabs {
        margin-top: 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #f1f5f9;
        padding: 3px !important;
        border-radius: 8px;
        margin-bottom: 0.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 30px !important;
        padding: 0 10px !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Tab panel padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.3rem !important;
    }

    /* ==========================================================================
       CODE BLOCKS - COMPACT
       ========================================================================== */
    .stCodeBlock {
        margin-top: 0.3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stCodeBlock pre {
        padding: 0.6rem !important;
        font-size: 13px !important;
        line-height: 1.3 !important;
    }

    /* ==========================================================================
       BUTTONS - COMPACT
       ========================================================================== */
    .stButton button {
        padding: 0.3rem 0.8rem !important;
        font-size: 13px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    /* ==========================================================================
       ALERTS - COMPACT
       ========================================================================== */
    .stAlert {
        padding: 0.5rem 0.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stAlert p {
        margin: 0 !important;
    }

    /* ==========================================================================
       EXPANDERS - COMPACT
       ========================================================================== */
    .streamlit-expanderHeader {
        font-size: 14px !important;
        padding: 0.4rem 0.6rem !important;
    }
    
    .streamlit-expanderContent {
        padding: 0.5rem 0.8rem !important;
    }

    /* ==========================================================================
       COLUMNS - COMPACT GAP
       ========================================================================== */
    [data-testid="column"] {
        padding: 0 0.3rem !important;
    }

    /* ==========================================================================
       TABLES - COMPACT
       ========================================================================== */
    .stTable {
        font-size: 13px !important;
    }
    
    .stTable th, .stTable td {
        padding: 0.3rem 0.5rem !important;
    }

    /* ==========================================================================
       GRAPHVIZ - COMPACT
       ========================================================================== */
    .stGraphVizChart {
        margin: 0.3rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def show_header():
    st.markdown("""
<div style="
    display: flex; 
    align-items: center; 
    justify-content: center; 
    gap: 12px; 
    padding: 10px 0;
    margin-bottom: 10px;
">
    <div style="font-size: 42px;">🐍</div>
    <div style="display: flex; flex-direction: column;">
        <h1 style='
            margin: 0 !important; 
            padding: 0 !important; 
            font-size: 32px !important; 
            font-weight: 800; 
            letter-spacing: -1px; 
            color: #111827;
            line-height: 1;
        '>
            Python <span style='background: linear-gradient(135deg, #306998 0%, #FFD43B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Mastery</span>
        </h1>
        <div style="font-size: 14px; color: #64748b; font-weight: 500; margin-top: 3px;">
            From Hello World to Neural Networks
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def show_footer():
    """Display a professional copyright footer."""
    current_year = 2024
    st.markdown("---")
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 20px 0;
        margin-top: 20px;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    ">
        <p style="margin: 0 0 8px 0 !important; font-size: 14px; color: #64748b;">
            © {current_year} <strong>Python Mastery Hub</strong>. All rights reserved.
        </p>
        <p style="margin: 0 !important; font-size: 13px; color: #94a3b8;">
            Created with ❤️ by <a href="mailto:vikas.singh.info@gmail.com" style="color: #3b82f6; text-decoration: none;">vikas.singh.info@gmail.com</a>
        </p>
        <div style="margin-top: 10px; display: flex; justify-content: center; gap: 15px;">
            <a href="https://github.com/learnsharelead/python-mastery" target="_blank" style="color: #64748b; text-decoration: none; font-size: 13px;">
                📦 GitHub
            </a>
            <span style="color: #cbd5e1;">|</span>
            <a href="mailto:vikas.singh.info@gmail.com" style="color: #64748b; text-decoration: none; font-size: 13px;">
                ✉️ Contact
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

