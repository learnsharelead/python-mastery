import streamlit as st
from streamlit_option_menu import option_menu

def show_top_nav(current_page="Home"):
    """Renders the consistent top navigation bar."""
    
    # Hide default sidebar nav
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none !important;}
            section[data-testid="stSidebar"] {display: none !important;}
        </style>
    """, unsafe_allow_html=True)
    
    page_map = {
        "Home": "Home.py",
        "Beginner": "pages/1_🌱_Beginner.py",
        "Intermediate": "pages/2_🚀_Intermediate.py",
        "Advanced": "pages/3_🏗️_Advanced.py",
        "AI Integration": "pages/4_🧠_AI_Integration.py"
    }
    
    options = list(page_map.keys())
    
    try:
        default_index = options.index(current_page)
    except ValueError:
        default_index = 0

    selected = option_menu(
        menu_title=None,
        options=options,
        icons=["house", "flower1", "rocket", "building", "cpu"],
        default_index=default_index,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "4px 8px !important", 
                "background-color": "#ffffff", 
                "border-radius": "10px", 
                "margin-bottom": "10px", 
                "box-shadow": "0 1px 3px rgba(0,0,0,0.05)",
                "border": "1px solid #e2e8f0"
            },
            "icon": {"color": "#3b82f6", "font-size": "14px"}, 
            "nav-link": {
                "font-size": "13px", 
                "text-align": "center", 
                "margin": "0 2px", 
                "--hover-color": "#f1f5f9", 
                "color": "#64748b",
                "padding": "6px 12px",
                "border-radius": "6px"
            },
            "nav-link-selected": {
                "background-color": "#eff6ff", 
                "color": "#1e40af", 
                "font-weight": "600"
            },
        },
        key="top_nav_menu"
    )
    
    if selected != current_page:
        target_file = page_map[selected]
        st.switch_page(target_file)
