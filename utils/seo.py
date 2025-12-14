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
    - JSON-LD Structured Data (Schema.org)
    - Open Graph meta tags
    - Twitter Card meta tags
    - Semantic HTML for crawlers
    - Rich snippets for SERP features
    - FAQ Schema for featured snippets
    - Course Schema for educational content
    - BreadcrumbList for navigation
    """
    
    # Set page config
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
    
    # Combine all schemas into a graph
    schemas = [main_schema, website_schema, org_schema]
    if breadcrumb_schema:
        schemas.append(breadcrumb_schema)
    if faq_schema:
        schemas.append(faq_schema)
    if course_schema:
        schemas.append(course_schema)
    
    # Build meta tags HTML
    meta_tags = f"""
    <!-- Primary Meta Tags -->
    <title>{title}</title>
    <meta name="title" content="{title}">
    <meta name="description" content="{description}">
    <meta name="keywords" content="{', '.join(keywords) if keywords else ''}">
    <meta name="author" content="{author}">
    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1">
    <meta name="language" content="English">
    <meta name="revisit-after" content="7 days">
    <link rel="canonical" href="{canonical_url}">
    
    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="article">
    <meta property="og:url" content="{canonical_url}">
    <meta property="og:title" content="{title}">
    <meta property="og:description" content="{description}">
    <meta property="og:image" content="{og_image}">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:site_name" content="Python Mastery Hub">
    <meta property="og:locale" content="en_US">
    <meta property="article:published_time" content="{date_published}">
    <meta property="article:modified_time" content="{date_modified}">
    <meta property="article:author" content="{author}">
    <meta property="article:section" content="Programming">
    
    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:url" content="{canonical_url}">
    <meta name="twitter:title" content="{title}">
    <meta name="twitter:description" content="{description}">
    <meta name="twitter:image" content="{og_image}">
    <meta name="twitter:site" content="@pythonmastery">
    <meta name="twitter:creator" content="@pythonmastery">
    
    <!-- Additional SEO -->
    <meta name="googlebot" content="index, follow">
    <meta name="bingbot" content="index, follow">
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """
    
    # Keywords as hidden semantic tags
    keyword_tags = ""
    if keywords:
        for kw in keywords[:10]:  # Limit to 10 keywords
            keyword_tags += f'<meta property="article:tag" content="{kw}">\n'
    
    # JSON-LD structured data
    json_ld_scripts = ""
    for schema in schemas:
        json_ld_scripts += f'<script type="application/ld+json">{json.dumps(schema, indent=2)}</script>\n'
    
    # Hidden semantic HTML for crawlers
    semantic_html = f"""
    <div style="display:none;" aria-hidden="true">
        <h1 id="page-title">{title}</h1>
        <p id="page-description">{description}</p>
        <nav aria-label="breadcrumb">
            <ol>{"".join([f'<li><a href="{c.get("url", "#")}">{c.get("name")}</a></li>' for c in (breadcrumbs or [{"name": "Home", "url": base_url}])])}</ol>
        </nav>
        <article itemscope itemtype="https://schema.org/{schema_type}">
            <meta itemprop="headline" content="{title}">
            <meta itemprop="description" content="{description}">
            <meta itemprop="datePublished" content="{date_published}">
            <meta itemprop="dateModified" content="{date_modified}">
        </article>
    </div>
    """
    
    # Inject all SEO elements
    st.markdown(f"""
    {meta_tags}
    {keyword_tags}
    {json_ld_scripts}
    {semantic_html}
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
    
    # Inject FAQ schema
    st.markdown(f"""
    <script type="application/ld+json">{json.dumps(faq_schema)}</script>
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
    <script type="application/ld+json">{json.dumps(how_to_schema)}</script>
    """, unsafe_allow_html=True)
