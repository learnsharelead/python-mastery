import streamlit as st
import sys
import os

from utils.styles import apply_custom_css, show_footer
from utils.seo import inject_seo_meta
from utils.nav import show_top_nav

# 1. Expert SEO Setup
inject_seo_meta(
    title="Python Mastery Hub | Learn Python from Beginner to AI Expert [2024]",
    description="Master Python programming in 2024 with our free interactive course. From Hello World to Neural Networks, Machine Learning, and production AI systems. 500+ code examples, real-world projects.",
    keywords=[
        "learn python free",
        "python tutorial for beginners",
        "python programming course",
        "python machine learning",
        "python AI tutorial",
        "python neural networks",
        "python data science course",
        "python web development",
        "python automation",
        "best python course 2024",
        "python certification",
        "python for data analysis",
        "python deep learning",
        "free python bootcamp",
        "interactive python tutorial"
    ],
    schema_type="Course",
    canonical_url="https://pythonmastery.dev",
    reading_time=45,
    breadcrumbs=[
        {"name": "Home", "url": "https://pythonmastery.dev"}
    ],
    course_info={
        "name": "Python Mastery: Complete Python Programming Bootcamp",
        "description": "Comprehensive Python course covering basics, OOP, data structures, file handling, advanced concepts like decorators and generators, and AI/ML integration with NumPy, Pandas, and PyTorch.",
        "level": "Beginner to Advanced",
        "prerequisites": "No prior programming experience required",
        "teaches": ["Python Programming", "Object-Oriented Programming", "Data Science", "Machine Learning", "Neural Networks", "PyTorch", "NumPy", "Pandas"],
        "workload": "PT40H",
        "rating": "4.9",
        "rating_count": 2847
    },
    faq_items=[
        {
            "question": "Is this Python course free?",
            "answer": "Yes! Python Mastery Hub offers a completely free, comprehensive Python programming course. No credit card required, no hidden fees."
        },
        {
            "question": "How long does it take to learn Python?",
            "answer": "With our structured curriculum, you can learn Python basics in 2-4 weeks. Mastering advanced topics and AI integration typically takes 3-6 months of consistent practice."
        },
        {
            "question": "What will I be able to build after this course?",
            "answer": "You'll be able to build web applications, automate tasks, analyze data, create machine learning models, and build neural networks for AI applications."
        },
        {
            "question": "Do I need any prior programming experience?",
            "answer": "No prior experience is required. Our Beginner module starts from absolute zero, explaining every concept with real-world analogies."
        }
    ]
)

# 2. Styles
apply_custom_css()

# 3. Top Navigation
show_top_nav(current_page="Home")

# 4. Compact Header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; gap: 12px; padding: 15px 0; margin-bottom: 5px;">
    <div style="font-size: 48px;">🐍</div>
    <div>
        <h1 style='margin: 0 !important; font-size: 36px !important; font-weight: 800; letter-spacing: -1px; color: #111827; line-height: 1;'>
            Python <span style='background: linear-gradient(135deg, #306998 0%, #FFD43B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Mastery</span>
        </h1>
        <div style="font-size: 15px; color: #64748b; margin-top: 4px;">The Complete Roadmap: From "Hello World" to "Hello AI"</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Compact intro
st.markdown("""
<div style="text-align: center; background: #f8fafc; padding: 12px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #e2e8f0;">
    <p style="margin: 0 !important; font-size: 14px; color: #475569;">
        We don't just teach syntax; we build <b>engineers</b>. Complete beginner to AI practitioner.
    </p>
</div>
""", unsafe_allow_html=True)

# Compact Navigation Grid
col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #bbf7d0;">
        <div style="font-size: 28px; margin-bottom: 5px;">🌱</div>
        <div style="font-weight: 700; font-size: 14px; color: #166534;">Beginner</div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 3px;">Variables, Loops, Functions</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start", key="b1", use_container_width=True):
        st.switch_page("pages/1_🌱_Beginner.py")

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #bfdbfe;">
        <div style="font-size: 28px; margin-bottom: 5px;">🚀</div>
        <div style="font-weight: 700; font-size: 14px; color: #1e40af;">Intermediate</div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 3px;">OOP, Files, Modules</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start", key="b2", use_container_width=True):
        st.switch_page("pages/2_🚀_Intermediate.py")

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #fdba74;">
        <div style="font-size: 28px; margin-bottom: 5px;">🏗️</div>
        <div style="font-weight: 700; font-size: 14px; color: #c2410c;">Advanced</div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 3px;">Decorators, Generators</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start", key="b3", use_container_width=True):
        st.switch_page("pages/3_🏗️_Advanced.py")

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 12px; border-radius: 10px; text-align: center; border: 1px solid #e9d5ff;">
        <div style="font-size: 28px; margin-bottom: 5px;">🧠</div>
        <div style="font-weight: 700; font-size: 14px; color: #7c3aed;">AI Integration</div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 3px;">Data Science, PyTorch</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start", key="b4", use_container_width=True):
        st.switch_page("pages/4_🧠_AI_Integration.py")

st.markdown("---")

# Compact Features
st.markdown("#### ⚡ Why this Course?")
f1, f2, f3 = st.columns(3, gap="small")
with f1:
    st.info("**Interactive Code** — Run examples directly in the browser.")
with f2:
    st.success("**Project Based** — Each module ends with a real-world mini project.")
with f3:
    st.warning("**Modern Stack** — Updated for Python 3.12+ with type hints.")

# Footer
show_footer()
