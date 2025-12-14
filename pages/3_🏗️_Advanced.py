import streamlit as st
from utils.styles import apply_custom_css
from utils.seo import inject_seo_meta
from utils.nav import show_top_nav

# SEO & Styles
inject_seo_meta(
    title="Python Advanced - Decorators, Generators & Async",
    description="Master Python's advanced features: Decorators, Generators, Context Managers, and Concurrency.",
    keywords=["Python Decorators", "Generators", "Async Python", "Advanced Python"]
)
apply_custom_css()
show_top_nav(current_page="Advanced")

# Header
st.markdown("""
<div style="text-align: center; padding: 12px; background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); border-radius: 10px; margin-bottom: 10px;">
    <h2 style="margin: 0 !important; font-size: 1.4rem !important;">🏗️ Advanced: Expert-Level Python</h2>
    <p style="margin: 5px 0 0 0 !important; font-size: 14px; color: #c2410c;">Unlock the "magic". Write libraries, frameworks, high-performance code.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Decorators", "⚡ Generators", "🚪 Context Mgr", "🧵 Concurrency", "📐 Types"])

# =============================================================================
# TAB 1: DECORATORS
# =============================================================================
with tab1:
    st.markdown("## 🎨 Chapter 1: Decorators")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 Definition</h4>
            <p style="margin: 0 !important; font-size: 13px;">A function that <b>wraps</b> another function to add behavior, without changing original code.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy: Iron Man Suit</h4>
            <p style="margin: 0 !important; font-size: 13px;">Tony Stark + Suit = Enhanced abilities. Tony unchanged, suit adds powers.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🖼️ How Decorators Work", expanded=True):
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.code("""
# Step 1: Original function
def say_hello():
    return "Hello!"

# Step 2: Decorator adds "sparkles"
def make_fancy(func):
    def wrapper():
        result = func()
        return f"✨ {result} ✨"
    return wrapper

# Step 3: Apply manually
say_hello = make_fancy(say_hello)
print(say_hello())  # ✨ Hello! ✨
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=LR; node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
                o [label="Original\\nFunction", fillcolor="#dbeafe"];
                d [label="Decorator", fillcolor="#fef3c7"];
                e [label="Enhanced\\nFunction", fillcolor="#dcfce7"];
                o -> d [label="in"]; d -> e [label="out"];
            }
            """)

    with st.expander("✨ The `@` Syntax (Clean Way)", expanded=True):
        st.code("""
def make_fancy(func):
    def wrapper():
        return f"✨ {func()} ✨"
    return wrapper

@make_fancy  # Same as: say_hello = make_fancy(say_hello)
def say_hello():
    return "Hello!"

print(say_hello())  # ✨ Hello! ✨
        """, language="python")

    with st.expander("⏱️ Real Example: Timer Decorator", expanded=False):
        st.code("""
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done!"

slow_function()  # slow_function took 1.0012s
        """, language="python")

    with st.expander("📝 Real Example: Logger Decorator", expanded=False):
        st.code("""
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling: {func.__name__}")
        print(f"  Args: {args}")
        result = func(*args, **kwargs)
        print(f"  Returned: {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

add(5, 3)
# Calling: add
#   Args: (5, 3)
#   Returned: 8
        """, language="python")

    with st.expander("🏭 INDUSTRY: Decorators in Production", expanded=False):
        st.markdown("""
        ### 🏢 How Companies Use Decorators
        
        | Use Case | Example | Used By |
        |---|---|---|
        | **Authentication** | `@login_required` | Django, Flask |
        | **Caching** | `@lru_cache`, `@cached` | Every web app |
        | **Rate Limiting** | `@rate_limit(100/min)` | APIs, Twitter |
        | **Retry Logic** | `@retry(max_attempts=3)` | AWS SDK, tenacity |
        | **Monitoring** | `@traced`, `@metered` | Datadog, New Relic |
        """)
        st.code("""
# Production retry decorator (like Netflix's resilience patterns)
from functools import wraps
import time

def retry(max_attempts=3, delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    time.sleep(delay * (backoff ** attempts))
            return wrapper
        return wrapper
    return decorator

@retry(max_attempts=5, delay=0.5)
def call_external_api():
    # If this fails, it retries with exponential backoff
    response = requests.get("https://api.unreliable.com")
    return response.json()
        """, language="python")


# =============================================================================
# TAB 2: GENERATORS
# =============================================================================
with tab2:
    st.markdown("## ⚡ Chapter 2: Generators")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 Definition</h4>
            <p style="margin: 0 !important; font-size: 13px;">A function using <code>yield</code> to produce values <b>one at a time</b>, saving memory.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy: Pancake Cook</h4>
            <p style="margin: 0 !important; font-size: 13px;"><b>List:</b> Make 100 pancakes, stack all. <b>Generator:</b> Make 1, serve, repeat.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📊 List vs Generator Comparison", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❌ List: Memory Heavy**")
            st.code("""
def squares_list(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result

# 1M items ALL in RAM!
data = squares_list(1000000)
            """, language="python")
            st.error("Uses ~8MB+ for 1M integers")
        with col2:
            st.markdown("**✅ Generator: Memory Efficient**")
            st.code("""
def squares_gen(n):
    for i in range(n):
        yield i * i

# Just a generator object
data = squares_gen(1000000)

print(next(data))  # 0
print(next(data))  # 1
            """, language="python")
            st.success("Almost zero extra memory!")

    with st.expander("🔄 How `yield` Works", expanded=False):
        st.graphviz_chart("""
        digraph {
            rankdir=TB; node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
            call [label="Call function", shape=oval, fillcolor="#e0f2fe"];
            gen [label="Returns generator object\\n(no code runs yet!)", fillcolor="#dbeafe"];
            next [label="next() called", fillcolor="#fef3c7"];
            run [label="Runs until yield", fillcolor="#dcfce7"];
            pause [label="Pauses, returns value", fillcolor="#bbf7d0"];
            call -> gen -> next -> run -> pause;
            pause -> next [style=dashed, label="next()"];
        }
        """)

    with st.expander("♾️ Infinite Generator Example", expanded=False):
        st.code("""
def infinite_counter(start=0):
    current = start
    while True:  # Runs forever!
        yield current
        current += 1

counter = infinite_counter(10)
print(next(counter))  # 10
print(next(counter))  # 11
print(next(counter))  # 12
# Can go forever without storing!
        """, language="python")

    with st.expander("🏭 INDUSTRY: Generators in Data Pipelines", expanded=False):
        st.markdown("""
        ### 🏢 Where Generators Shine in Production
        
        | Use Case | Why Generators | Example |
        |---|---|---|
        | **Log Processing** | Process TB of logs without OOM | Splunk, ELK |
        | **ETL Pipelines** | Stream data through transforms | Airflow, Spark |
        | **API Pagination** | Fetch pages on-demand | REST API clients |
        | **File Parsing** | Read massive CSVs line by line | Pandas chunking |
        """)
        st.code("""
# Production ETL Pipeline with Generators
def read_large_csv(filepath):
    \"\"\"Read 100GB CSV without loading into memory\"\"\"
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

def transform_records(records):
    \"\"\"Apply transformations\"\"\"
    for record in records:
        record['price'] = float(record['price']) * 1.1  # 10% markup
        record['processed_at'] = datetime.now().isoformat()
        yield record

def filter_valid(records):
    \"\"\"Filter out invalid records\"\"\"
    for record in records:
        if float(record['price']) > 0:
            yield record

# Chain generators (no memory explosion!)
pipeline = filter_valid(transform_records(read_large_csv("100gb_sales.csv")))

for record in pipeline:
    save_to_database(record)  # Processes one at a time
        """, language="python")


# =============================================================================
# TAB 3: CONTEXT MANAGERS
# =============================================================================
with tab3:
    st.markdown("## 🚪 Chapter 3: Context Managers")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 Definition</h4>
            <p style="margin: 0 !important; font-size: 13px;">Auto-handles <b>setup</b> and <b>teardown</b>. Opens resources, cleans up even if error.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy: Automatic Door</h4>
            <p style="margin: 0 !important; font-size: 13px;">Opens on entry, closes on exit — no matter if you walk or run!</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❌ vs ✅ With and Without `with`", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❌ Without `with`**")
            st.code("""
file = open("data.txt", "r")
content = file.read()
# What if error here? File never closes!
file.close()
            """, language="python")
        with col2:
            st.markdown("**✅ With `with`**")
            st.code("""
with open("data.txt", "r") as file:
    content = file.read()
    # Even if error, file closes!

# File auto-closed here
            """, language="python")

    with st.expander("🛠️ Create Your Own Context Manager", expanded=False):
        st.code("""
from contextlib import contextmanager
import time

@contextmanager
def timer_context():
    print("⏱️ Timer started...")
    start = time.time()
    
    yield  # Your code runs here
    
    end = time.time()
    print(f"⏱️ Done in {end - start:.4f}s")

# Usage
with timer_context():
    time.sleep(1.5)
    print("Doing work...")

# Output:
# ⏱️ Timer started...
# Doing work...
# ⏱️ Done in 1.5012s
        """, language="python")

    with st.expander("🏭 INDUSTRY: Context Managers in Production", expanded=False):
        st.markdown("""
        ### 🏢 Production Context Manager Patterns
        
        | Use Case | Why Context Manager | Example |
        |---|---|---|
        | **Database Connections** | Auto-close connections | SQLAlchemy, psycopg2 |
        | **Transactions** | Auto-commit/rollback | Django ORM |
        | **Distributed Locks** | Prevent race conditions | Redis locks |
        | **Temp Resources** | Cleanup temp files | tempfile module |
        | **Feature Flags** | Scoped feature toggles | LaunchDarkly |
        """)
        st.code("""
# Production Database Transaction Pattern
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker

@contextmanager
def db_transaction():
    \"\"\"Production-grade database transaction manager\"\"\"
    session = SessionLocal()
    try:
        yield session
        session.commit()  # Success: commit
    except Exception:
        session.rollback()  # Error: rollback
        raise
    finally:
        session.close()  # Always close

# Usage
with db_transaction() as db:
    db.add(User(name="Alice"))
    db.add(Order(user_id=1, total=100))
    # If ANY error occurs, BOTH operations roll back!

# Distributed Lock (prevents race conditions in microservices)
@contextmanager
def redis_lock(lock_name, timeout=10):
    \"\"\"Distributed lock across multiple servers\"\"\"
    lock = redis_client.lock(lock_name, timeout=timeout)
    acquired = lock.acquire(blocking=True)
    try:
        if acquired:
            yield
        else:
            raise Exception("Could not acquire lock")
    finally:
        if acquired:
            lock.release()

with redis_lock("payment_processing"):
    # Only ONE server can run this at a time
    process_payment(user_id=123)
        """, language="python")


# =============================================================================
# TAB 4: CONCURRENCY
# =============================================================================
with tab4:
    st.markdown("## 🧵 Chapter 4: Concurrency")

    st.markdown("""
    <div style="background: #fef2f2; padding: 12px; border-radius: 8px; border-left: 4px solid #dc2626;">
        <h4 style="color: #991b1b; margin: 0 0 8px 0 !important;">⚠️ Advanced Territory</h4>
        <p style="margin: 0 !important; font-size: 13px;">Complex topic! We cover key concepts to get you started.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📚 Key Definitions", expanded=True):
        st.markdown("""
        | Term | Meaning | Analogy |
        |---|---|---|
        | **Synchronous** | One after another | Single cashier line |
        | **Asynchronous** | Start task, do other things, come back | Order food, browse while cooking |
        | **Concurrency** | Managing multiple tasks | 1 waiter, 5 tables |
        | **Parallelism** | Running tasks simultaneously | 5 waiters, 5 tables |
        """)

    with st.expander("🧵 Threading: For I/O-Bound Tasks", expanded=False):
        st.code("""
import threading
import time

def download(name):
    print(f"Starting: {name}")
    time.sleep(2)  # Simulate network
    print(f"Done: {name}")

# Create threads
t1 = threading.Thread(target=download, args=("file_A",))
t2 = threading.Thread(target=download, args=("file_B",))

t1.start()
t2.start()

t1.join()
t2.join()

print("All done!")
# Both run "at same time" (concurrent)
        """, language="python")

    with st.expander("⚡ Async/Await: Modern Concurrency", expanded=False):
        st.code("""
import asyncio

async def fetch(url):
    print(f"Fetching {url}...")
    await asyncio.sleep(1)  # Non-blocking!
    return f"Data from {url}"

async def main():
    results = await asyncio.gather(
        fetch("api/users"),
        fetch("api/posts"),
        fetch("api/comments")
    )
    print(f"Got {len(results)} results")

# asyncio.run(main())
# All 3 in ~1s, not 3s!
        """, language="python")

    with st.expander("🏭 INDUSTRY: High-Scale Concurrency", expanded=False):
        st.markdown("""
        ### 🏢 Production Concurrency Patterns
        
        | Use Case | Pattern | Example |
        |---|---|---|
        | **Web Servers** | Async I/O | FastAPI, uvicorn (10K+ RPS) |
        | **Background Jobs** | Task queues | Celery, RQ, Dramatiq |
        | **Microservices** | Async messaging | Kafka consumers |
        | **Data Processing** | Parallel workers | multiprocessing |
        | **Real-time** | WebSockets | Discord, Slack |
        """)
        st.code("""
# FastAPI handles 10,000+ requests/second with async
from fastapi import FastAPI
import httpx  # Async HTTP client

app = FastAPI()

@app.get("/aggregate")
async def aggregate_data():
    # Fetch from 3 microservices CONCURRENTLY
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            client.get("http://users-service/api/users/123"),
            client.get("http://orders-service/api/orders?user=123"),
            client.get("http://payments-service/api/balance/123"),
        )
    
    # 3 requests in ~50ms instead of ~150ms
    return {
        "user": results[0].json(),
        "orders": results[1].json(),
        "balance": results[2].json()
    }

# Celery: Distributed Task Queue (used by Instagram)
from celery import Celery

app = Celery('tasks', broker='redis://localhost')

@app.task
def send_email(to, subject, body):
    # Runs in background worker (not blocking web server)
    email_client.send(to, subject, body)

# Trigger from anywhere
send_email.delay("user@email.com", "Welcome!", "Hello...")
        """, language="python")


# =============================================================================
# TAB 5: TYPE HINTING
# =============================================================================
with tab5:
    st.markdown("## 📐 Chapter 5: Type Hinting")

    st.markdown("""
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What Are Type Hints?</h4>
        <p style="margin: 0 !important; font-size: 13px;">Optional annotations showing expected types. Python doesn't enforce, but tools like MyPy can check.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("❌ vs ✅ Without and With Types", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❌ Without Types**")
            st.code("""
def process(data):
    # What is data? List? Dict? String?
    return data.upper()
            """, language="python")
            st.error("Errors found at runtime")
        with col2:
            st.markdown("**✅ With Types**")
            st.code("""
def process(data: str) -> str:
    # Clear: str in, str out
    return data.upper()
            """, language="python")
            st.success("Tools catch errors before running")

    with st.expander("📚 Common Type Hints", expanded=False):
        st.code("""
from typing import List, Dict, Optional, Tuple

# Variables
name: str = "Alice"
age: int = 30
price: float = 9.99
active: bool = True

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections
scores: List[int] = [90, 85, 78]
user: Dict[str, str] = {"name": "Alice"}

# Optional (can be None)
def find(id: int) -> Optional[Dict]:
    if id == 1:
        return {"name": "Admin"}
    return None
        """, language="python")

    with st.expander("🏭 INDUSTRY: Types in Enterprise Python", expanded=False):
        st.markdown("""
        ### 🏢 Why Big Companies Mandate Type Hints
        
        | Benefit | Why It Matters |
        |---|---|
        | **Fewer Bugs** | Catch type errors before deployment |
        | **Better Docs** | Self-documenting code |
        | **IDE Support** | Autocomplete, refactoring |
        | **Large Teams** | Reduces miscommunication |
        | **CI/CD** | MyPy checks in pipelines |
        """)
        st.code("""
# Pydantic: Runtime Type Validation (FastAPI uses this)
from pydantic import BaseModel, EmailStr, validator
from typing import Optional

class User(BaseModel):
    name: str
    email: EmailStr
    age: int
    bio: Optional[str] = None
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

# Validation happens automatically!
user = User(name="Alice", email="alice@example.com", age=25)

# This raises ValidationError at runtime
try:
    bad_user = User(name="Bob", email="invalid", age=-5)
except Exception as e:
    print(e)  # Shows exactly what's wrong

# Dataclasses with types (built-in Python)
from dataclasses import dataclass

@dataclass
class Order:
    order_id: int
    items: list[str]
    total: float
    shipped: bool = False

order = Order(order_id=1, items=["Book", "Pen"], total=25.99)
        """, language="python")

