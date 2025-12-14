import streamlit as st
import json
from datetime import datetime

def inject_seo_meta(
    title: str,
    description: str,
    keywords: list = None,
    schema_type: str = "TechArticle",
    canonical_url: str = None,
    og_image: str = None,
    author: str = "Python Mastery Hub",
    date_published: str = None,
    date_modified: str = None,
    breadcrumbs: list = None,
    faq_items: list = None,
    course_info: dict = None,
    reading_time: int = None
):
    """
    Expert-level SEO injection for Streamlit pages.
    
    Implements:
    - JSON-LD Structured Data (Schema.org) - Works in Streamlit
    - Hidden semantic HTML for crawlers
    - Rich snippets for SERP features
    - FAQ Schema for featured snippets
    - Course Schema for educational content
    - BreadcrumbList for navigation
    
    Note: Streamlit doesn't support traditional meta tags in <head>.
    This implementation focuses on JSON-LD structured data which
    search engines can read from the page body.
    """
    
    # Set page config with SEO-optimized title
    st.set_page_config(
        page_title=title,
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Current date for freshness signals
    current_date = datetime.now().strftime("%Y-%m-%d")
    date_published = date_published or "2024-01-01"
    date_modified = date_modified or current_date
    
    # Default canonical URL
    base_url = "https://pythonmastery.dev"
    if not canonical_url:
        canonical_url = base_url
    
    # Default OG image
    if not og_image:
        og_image = f"{base_url}/assets/og-image.png"
    
    # Primary Article/TechArticle Schema
    main_schema = {
        "@context": "https://schema.org",
        "@type": schema_type,
        "@id": f"{canonical_url}#article",
        "headline": title,
        "description": description,
        "author": {
            "@type": "Organization",
            "@id": f"{base_url}#organization",
            "name": author,
            "url": base_url,
            "logo": {
                "@type": "ImageObject",
                "url": f"{base_url}/assets/logo.png"
            }
        },
        "publisher": {
            "@type": "Organization",
            "@id": f"{base_url}#organization",
            "name": "Python Mastery Hub",
            "logo": {
                "@type": "ImageObject",
                "url": f"{base_url}/assets/logo.png",
                "width": 600,
                "height": 60
            }
        },
        "datePublished": date_published,
        "dateModified": date_modified,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": canonical_url
        },
        "image": {
            "@type": "ImageObject",
            "url": og_image,
            "width": 1200,
            "height": 630
        },
        "inLanguage": "en-US",
        "isAccessibleForFree": True,
        "educationalLevel": "Beginner to Advanced",
        "learningResourceType": "Interactive Tutorial"
    }
    
    if keywords:
        main_schema["keywords"] = ", ".join(keywords) if isinstance(keywords, list) else keywords
    
    if reading_time:
        main_schema["timeRequired"] = f"PT{reading_time}M"
    
    # WebSite Schema for sitelinks
    website_schema = {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "@id": f"{base_url}#website",
        "name": "Python Mastery Hub",
        "url": base_url,
        "description": "Learn Python from beginner to AI expert with interactive tutorials and real-world projects",
        "publisher": {"@id": f"{base_url}#organization"},
        "potentialAction": {
            "@type": "SearchAction",
            "target": {
                "@type": "EntryPoint",
                "urlTemplate": f"{base_url}/search?q={{search_term_string}}"
            },
            "query-input": "required name=search_term_string"
        }
    }
    
    # BreadcrumbList Schema
    breadcrumb_schema = None
    if breadcrumbs:
        breadcrumb_items = []
        for i, crumb in enumerate(breadcrumbs, 1):
            breadcrumb_items.append({
                "@type": "ListItem",
                "position": i,
                "name": crumb.get("name"),
                "item": crumb.get("url", f"{base_url}/{crumb.get('name', '').lower().replace(' ', '-')}")
            })
        breadcrumb_schema = {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": breadcrumb_items
        }
    
    # FAQPage Schema for featured snippets
    faq_schema = None
    if faq_items:
        faq_entities = []
        for faq in faq_items:
            faq_entities.append({
                "@type": "Question",
                "name": faq.get("question"),
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq.get("answer")
                }
            })
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": faq_entities
        }
    
    # Course Schema for educational content
    course_schema = None
    if course_info:
        course_schema = {
            "@context": "https://schema.org",
            "@type": "Course",
            "@id": f"{canonical_url}#course",
            "name": course_info.get("name", title),
            "description": course_info.get("description", description),
            "provider": {
                "@type": "Organization",
                "name": "Python Mastery Hub",
                "sameAs": base_url
            },
            "educationalLevel": course_info.get("level", "Beginner"),
            "coursePrerequisites": course_info.get("prerequisites", "None"),
            "teaches": course_info.get("teaches", keywords),
            "hasCourseInstance": {
                "@type": "CourseInstance",
                "courseMode": "online",
                "courseWorkload": course_info.get("workload", "PT10H")
            }
        }
        if course_info.get("rating"):
            course_schema["aggregateRating"] = {
                "@type": "AggregateRating",
                "ratingValue": course_info.get("rating"),
                "bestRating": "5",
                "worstRating": "1",
                "ratingCount": course_info.get("rating_count", 100)
            }
    
    # Organization Schema
    org_schema = {
        "@context": "https://schema.org",
        "@type": "Organization",
        "@id": f"{base_url}#organization",
        "name": "Python Mastery Hub",
        "url": base_url,
        "logo": f"{base_url}/assets/logo.png",
        "sameAs": [
            "https://github.com/python-mastery",
            "https://twitter.com/pythonmastery",
            "https://www.youtube.com/@pythonmastery"
        ],
        "contactPoint": {
            "@type": "ContactPoint",
            "contactType": "customer support",
            "email": "support@pythonmastery.dev"
        }
    }
    
    # Combine all schemas
    schemas = [main_schema, website_schema, org_schema]
    if breadcrumb_schema:
        schemas.append(breadcrumb_schema)
    if faq_schema:
        schemas.append(faq_schema)
    if course_schema:
        schemas.append(course_schema)
    
    # Build hidden semantic content with JSON-LD
    # All elements are completely hidden but readable by search engines
    breadcrumb_html = ""
    if breadcrumbs:
        crumbs = "".join([f'<li><a href="{c.get("url", "#")}">{c.get("name")}</a></li>' for c in breadcrumbs])
        breadcrumb_html = f'<nav aria-label="breadcrumb"><ol>{crumbs}</ol></nav>'
    
    # JSON-LD scripts (these are properly parsed by search engines)
    json_ld_scripts = ""
    for schema in schemas:
        json_ld_scripts += f'<script type="application/ld+json">{json.dumps(schema)}</script>\n'
    
    # Inject everything in a completely hidden container
    st.markdown(f"""
    <div style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;" aria-hidden="true">
        {json_ld_scripts}
        <article itemscope itemtype="https://schema.org/{schema_type}">
            <h1 itemprop="headline">{title}</h1>
            <p itemprop="description">{description}</p>
            <meta itemprop="datePublished" content="{date_published}">
            <meta itemprop="dateModified" content="{date_modified}">
            <meta itemprop="author" content="{author}">
            <meta itemprop="keywords" content="{', '.join(keywords) if keywords else ''}">
            {breadcrumb_html}
        </article>
    </div>
    """, unsafe_allow_html=True)


def inject_faq_section(faq_items: list, expanded: bool = False):
    """
    Render an FAQ section with Schema.org markup for featured snippets.
    """
    faq_schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": []
    }
    
    with st.expander("❓ Frequently Asked Questions", expanded=expanded):
        for faq in faq_items:
            st.markdown(f"**Q: {faq['question']}**")
            st.markdown(f"A: {faq['answer']}")
            st.markdown("---")
            faq_schema["mainEntity"].append({
                "@type": "Question",
                "name": faq["question"],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq["answer"]
                }
            })
    
    # Inject FAQ schema (hidden)
    st.markdown(f"""
    <div style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;">
        <script type="application/ld+json">{json.dumps(faq_schema)}</script>
    </div>
    """, unsafe_allow_html=True)


def inject_how_to_schema(title: str, steps: list, total_time: str = "PT30M"):
    """
    Inject HowTo Schema for step-by-step guides (enables rich results).
    """
    how_to_schema = {
        "@context": "https://schema.org",
        "@type": "HowTo",
        "name": title,
        "totalTime": total_time,
        "step": []
    }
    
    for i, step in enumerate(steps, 1):
        how_to_schema["step"].append({
            "@type": "HowToStep",
            "position": i,
            "name": step.get("name"),
            "text": step.get("text"),
            "url": step.get("url", "")
        })
    
    st.markdown(f"""
    <div style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;">
        <script type="application/ld+json">{json.dumps(how_to_schema)}</script>
    </div>
    """, unsafe_allow_html=True)
