import streamlit as st
from utils.styles import apply_custom_css
from utils.seo import inject_seo_meta
from utils.nav import show_top_nav

# Expert SEO & Styles
inject_seo_meta(
    title="Python AI & Machine Learning Tutorial | NumPy, Pandas, PyTorch & Production ML",
    description="Build production AI systems with Python. Master NumPy, Pandas, Neural Networks, and PyTorch. Learn MLOps, model serving, LLM integration, and real-world AI architectures used by Google, Meta, and OpenAI.",
    keywords=[
        "python machine learning tutorial",
        "python AI programming",
        "numpy tutorial python",
        "pandas data science",
        "pytorch neural network",
        "python deep learning",
        "tensorflow vs pytorch",
        "python data science course",
        "production machine learning",
        "MLOps python tutorial",
        "python LLM integration",
        "RAG retrieval augmented generation",
        "python model deployment",
        "fastapi machine learning",
        "python neural network from scratch",
        "transformer architecture python",
        "GPT integration python",
        "python AI agents"
    ],
    schema_type="TechArticle",
    canonical_url="https://pythonmastery.dev/ai-integration",
    reading_time=120,
    breadcrumbs=[
        {"name": "Home", "url": "https://pythonmastery.dev"},
        {"name": "AI Integration", "url": "https://pythonmastery.dev/ai-integration"}
    ],
    course_info={
        "name": "Python AI Integration: From Data Science to Production ML",
        "description": "Bridge Python with AI. Learn NumPy, Pandas, Neural Networks, PyTorch, and build production-grade AI systems like Google, Meta, and OpenAI.",
        "level": "Intermediate to Advanced",
        "prerequisites": "Python fundamentals, basic math (linear algebra helpful)",
        "teaches": ["NumPy", "Pandas", "Neural Networks", "PyTorch", "MLOps", "Model Serving", "LLM Integration", "Production AI"],
        "workload": "PT20H",
        "rating": "4.9",
        "rating_count": 1256
    },
    faq_items=[
        {
            "question": "What is the difference between NumPy and Pandas?",
            "answer": "NumPy provides fast numerical arrays and mathematical operations - it's the backbone of scientific Python. Pandas builds on NumPy to provide DataFrames for tabular data with labels, making data manipulation and analysis more intuitive. Use NumPy for numerical computing, Pandas for data wrangling."
        },
        {
            "question": "Should I learn TensorFlow or PyTorch?",
            "answer": "PyTorch is recommended for most learners in 2024. It's more intuitive, dominates research, and is now production-ready. PyTorch is used by OpenAI (GPT), Meta (LLaMA), Tesla, and most AI startups. TensorFlow is still strong in enterprise and mobile deployment."
        },
        {
            "question": "What is RAG (Retrieval-Augmented Generation)?",
            "answer": "RAG combines LLMs with a retrieval system to ground answers in your own documents. First, relevant documents are retrieved using embeddings and vector search, then the LLM generates answers using that context. This reduces hallucinations and enables company-specific AI assistants."
        },
        {
            "question": "How do I deploy a machine learning model to production?",
            "answer": "Export your model (TorchScript, ONNX), wrap it in an API (FastAPI), containerize with Docker, deploy to cloud (AWS, GCP, Azure), and add monitoring (Prometheus, Evidently). Use caching for performance and implement proper error handling and fallbacks."
        },
        {
            "question": "What is MLOps and why is it important?",
            "answer": "MLOps applies DevOps practices to machine learning: version control for data and models, automated training pipelines, model registries, A/B testing for deployments, and monitoring for drift. It's essential for maintaining ML systems in production - Netflix, Uber, and Airbnb all have dedicated MLOps teams."
        }
    ]
)
apply_custom_css()
show_top_nav(current_page="AI Integration")

# Header
st.markdown("""
<div style="text-align: center; padding: 12px; background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-radius: 10px; margin-bottom: 10px;">
    <h2 style="margin: 0 !important; font-size: 1.4rem !important;">🧠 AI Integration: From Notebooks to Production</h2>
    <p style="margin: 5px 0 0 0 !important; font-size: 14px; color: #7c3aed;">Learn what companies like Google, Meta, and OpenAI actually use.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Pandas", "🧮 NumPy", "🕸️ Neural Nets", "🔥 PyTorch", "🏭 Industry AI", "🤖 Project"])

# =============================================================================
# TAB 1: PANDAS
# =============================================================================
with tab1:
    st.markdown("## 📊 Chapter 1: Pandas for Data Engineering")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What is Pandas?</h4>
            <p style="margin: 0 !important; font-size: 13px;">Python's "Excel on steroids". Clean, analyze, transform tabular data with code.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">🏢 Industry Usage</h4>
            <p style="margin: 0 !important; font-size: 13px;">Used by Netflix, Spotify, and every data team for ETL pipelines and analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🗃️ The DataFrame: Core Structure", expanded=True):
        st.graphviz_chart("""
        digraph {
            node [fontname="Arial", fontsize=10, shape=plaintext];
            df [label=<
                <table border="1" cellborder="1" cellspacing="0" cellpadding="6">
                    <tr><td bgcolor="#dbeafe"><b>Index</b></td><td bgcolor="#dbeafe"><b>Name</b></td><td bgcolor="#dbeafe"><b>Age</b></td><td bgcolor="#dbeafe"><b>City</b></td></tr>
                    <tr><td>0</td><td>Alice</td><td>28</td><td>NYC</td></tr>
                    <tr><td>1</td><td>Bob</td><td>34</td><td>London</td></tr>
                </table>
            >];
        }
        """)
        st.code("""
import pandas as pd

data = {"Name": ["Alice", "Bob"], "Age": [28, 34], "City": ["NYC", "London"]}
df = pd.DataFrame(data)
        """, language="python")

    with st.expander("🔧 Common Operations", expanded=False):
        st.code("""
df = pd.read_csv("sales.csv")

# Exploration
df.head(); df.describe(); df.info()

# Selection & Filtering
adults = df[df["Age"] > 30]
paris_adults = df[(df["Age"] > 25) & (df["City"] == "Paris")]

# Aggregation
df.groupby("Country")["Revenue"].sum()
        """, language="python")

    with st.expander("🏭 INDUSTRY: Real Data Engineering Pipelines", expanded=True):
        st.markdown("""
        ### 🏢 How Companies Actually Use Pandas
        
        | Company | Use Case |
        |---|---|
        | **Netflix** | Analyzing viewing patterns, A/B test results |
        | **Uber** | Cleaning ride data, surge pricing calculations |
        | **Airbnb** | Property analytics, pricing optimization |
        | **Banks** | Fraud detection data prep, risk scoring |
        """)
        
        st.markdown("#### ETL Pipeline Pattern (Extract-Transform-Load)")
        st.code("""
import pandas as pd
from sqlalchemy import create_engine

# EXTRACT: From multiple sources
sales_df = pd.read_csv("s3://bucket/sales.csv")
customers_df = pd.read_sql("SELECT * FROM customers", engine)
api_data = pd.DataFrame(requests.get("https://api.com/data").json())

# TRANSFORM: Clean & Enrich
df = sales_df.merge(customers_df, on="customer_id")
df["revenue"] = df["quantity"] * df["price"]
df["date"] = pd.to_datetime(df["date_str"])
df = df.dropna(subset=["revenue"])

# LOAD: To data warehouse
df.to_sql("sales_enriched", warehouse_engine, if_exists="replace")
df.to_parquet("s3://bucket/processed/sales.parquet")
        """, language="python")

        st.markdown("#### Large-Scale Data Processing with Chunking")
        st.code("""
# Processing 100GB+ files that don't fit in memory
chunk_size = 100_000
results = []

for chunk in pd.read_csv("huge_file.csv", chunksize=chunk_size):
    # Process each chunk
    chunk_result = chunk.groupby("category")["sales"].sum()
    results.append(chunk_result)

final = pd.concat(results).groupby(level=0).sum()
        """, language="python")


# =============================================================================
# TAB 2: NUMPY
# =============================================================================
with tab2:
    st.markdown("## 🧮 Chapter 2: NumPy for High-Performance Computing")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What is NumPy?</h4>
            <p style="margin: 0 !important; font-size: 13px;">Fast arrays & math. Backbone of ALL scientific Python.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">🏢 Industry Usage</h4>
            <p style="margin: 0 !important; font-size: 13px;">Every ML framework (PyTorch, TensorFlow) is built on NumPy concepts.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⚡ Vectorization: No Loops!", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**❌ Python Loop**")
            st.code('result = [x*2 for x in data]', language="python")
        with col2:
            st.markdown("**✅ NumPy**")
            st.code('result = data * 2  # 100x faster', language="python")

    with st.expander("🏭 INDUSTRY: High-Performance Computing", expanded=True):
        st.markdown("""
        ### 🏢 Production NumPy Patterns
        
        | Domain | Application |
        |---|---|
        | **Quantitative Finance** | Portfolio optimization, risk calculations |
        | **Scientific Research** | Simulations, signal processing |
        | **Computer Vision** | Image manipulation before ML |
        | **Audio Processing** | Spotify's audio feature extraction |
        """)
        
        st.markdown("#### Memory-Efficient Operations")
        st.code("""
import numpy as np

# In-place operations (no extra memory)
data = np.random.randn(10_000_000)
np.multiply(data, 2, out=data)  # Modify data directly

# Memory-mapped files (work with 100GB+ arrays)
big_array = np.memmap("huge_data.npy", dtype='float32', 
                       mode='r', shape=(1_000_000_000,))

# Process in chunks without loading all into RAM
chunk_means = [big_array[i:i+1000000].mean() 
               for i in range(0, len(big_array), 1000000)]
        """, language="python")

        st.markdown("#### Broadcasting for Efficient Computation")
        st.code("""
# Normalize 1 million images (shape: 1M x 224 x 224 x 3)
images = np.random.rand(1_000_000, 224, 224, 3)
mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
std = np.array([0.229, 0.224, 0.225])   # ImageNet std

# Broadcasting: applies to all 1M images at once!
normalized = (images - mean) / std
        """, language="python")


# =============================================================================
# TAB 3: NEURAL NETWORKS
# =============================================================================
with tab3:
    st.markdown("## 🕸️ Chapter 3: Neural Networks - Theory to Production")

    st.markdown("""
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What is a Neural Network?</h4>
        <p style="margin: 0 !important; font-size: 13px;">Layers of "neurons" that learn patterns. Powers ChatGPT, self-driving cars, and Spotify recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🖼️ Architecture Diagram", expanded=True):
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [shape=circle, style=filled, fontname="Arial", fontsize=9];
            subgraph cluster_i {
                label="Input"; style=filled; fillcolor="#dbeafe";
                i1 [label=""]; i2 [label=""]; i3 [label=""];
            }
            subgraph cluster_h {
                label="Hidden"; style=filled; fillcolor="#fef3c7";
                h1 [label=""]; h2 [label=""]; h3 [label=""];
            }
            subgraph cluster_o {
                label="Output"; style=filled; fillcolor="#dcfce7";
                o1 [label=""]; o2 [label=""];
            }
            i1 -> h1; i1 -> h2; i1 -> h3;
            i2 -> h1; i2 -> h2; i2 -> h3;
            i3 -> h1; i3 -> h2; i3 -> h3;
            h1 -> o1; h1 -> o2;
            h2 -> o1; h2 -> o2;
            h3 -> o1; h3 -> o2;
        }
        """)

    with st.expander("📚 Key Concepts", expanded=False):
        st.markdown("""
        | Concept | Meaning |
        |---|---|
        | **Neuron** | Sum(inputs × weights) + bias → activation |
        | **Weight/Bias** | Learnable parameters |
        | **Activation** | Non-linear function (ReLU, Sigmoid) |
        | **Loss** | How wrong the prediction is |
        | **Backprop** | Algorithm to calculate gradients |
        """)

    with st.expander("🏭 INDUSTRY: Modern Architecture Revolution", expanded=True):
        st.markdown("""
        ### 🏢 The Architectures That Changed Everything
        
        | Architecture | Year | Revolution | Used By |
        |---|---|---|---|
        | **CNN** | 2012 | Image recognition | Tesla Autopilot, Medical imaging |
        | **LSTM/GRU** | 2014 | Sequence data | Siri, Google Translate (old) |
        | **Transformer** | 2017 | Attention mechanism | GPT, BERT, everything modern |
        | **Diffusion** | 2020 | Image generation | DALL-E, Stable Diffusion, Midjourney |
        """)
        
        st.markdown("#### The Transformer Architecture (Powers ChatGPT)")
        st.graphviz_chart("""
        digraph {
            rankdir=TB; node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
            input [label="Input Tokens\\n'Hello world'", fillcolor="#dbeafe"];
            embed [label="Token Embedding\\n+ Position Encoding", fillcolor="#e0f2fe"];
            attn [label="Multi-Head\\nSelf-Attention", fillcolor="#fef3c7"];
            ff [label="Feed Forward\\nNetwork", fillcolor="#dcfce7"];
            norm [label="Layer Norm\\n+ Residual", fillcolor="#f3f4f6"];
            out [label="Output Tokens", fillcolor="#e9d5ff"];
            input -> embed -> attn -> ff -> norm;
            norm -> attn [style=dashed, label="Nx layers"];
            norm -> out;
        }
        """)

        st.markdown("#### Why Transformers Won")
        st.code("""
# The magic: Self-Attention
# "The cat sat on the mat because it was tired"
# Attention lets the model figure out "it" refers to "cat"

# Simplified attention calculation:
Query = Linear(input)   # What am I looking for?
Key = Linear(input)     # What do I contain?
Value = Linear(input)   # What do I actually say?

Attention = softmax(Query @ Key.T / sqrt(d)) @ Value
        """, language="python")


# =============================================================================
# TAB 4: PYTORCH
# =============================================================================
with tab4:
    st.markdown("## 🔥 Chapter 4: PyTorch - From Research to Production")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What is PyTorch?</h4>
            <p style="margin: 0 !important; font-size: 13px;">Deep learning framework by Meta. Most popular for research AND production.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">🏢 Who Uses It?</h4>
            <p style="margin: 0 !important; font-size: 13px;">OpenAI (GPT), Meta (LLaMA), Tesla, Uber, Airbnb, Microsoft.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🧱 Tensors & Autograd", expanded=True):
        st.code("""
import torch

x = torch.tensor([3.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([6.]) → d(x²)/dx = 2x = 6
        """, language="python")

    with st.expander("🏭 INDUSTRY: Production PyTorch Patterns", expanded=True):
        st.markdown("""
        ### 🏢 How Companies Deploy PyTorch Models
        
        | Stage | Tool | Purpose |
        |---|---|---|
        | **Training** | PyTorch + Lightning | Fast experimentation |
        | **Tracking** | MLflow, Weights & Biases | Log experiments |
        | **Serving** | TorchServe, Triton | REST API for models |
        | **Optimization** | TensorRT, ONNX | Speed up inference |
        | **Edge** | PyTorch Mobile | Phones, IoT devices |
        """)
        
        st.markdown("#### Production Training Script Pattern")
        st.code("""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import wandb

class ProductionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(config)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        
        # Auto-logged to W&B
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.config["lr"],
            weight_decay=0.01
        )

# Training with distributed GPU
trainer = pl.Trainer(
    accelerator="gpu",
    devices=8,           # 8 GPUs
    strategy="ddp",      # Distributed Data Parallel
    precision="16-mixed", # Mixed precision (faster)
    max_epochs=100,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor="val_loss"),
        pl.callbacks.EarlyStopping(patience=5)
    ]
)
trainer.fit(model, train_loader, val_loader)
        """, language="python")
        
        st.markdown("#### Model Export for Production")
        st.code("""
# Export to TorchScript (optimized, no Python dependency)
scripted = torch.jit.script(model)
scripted.save("model_production.pt")

# Export to ONNX (cross-platform)
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    dynamic_axes={"input": {0: "batch_size"}}
)

# Convert to TensorRT (NVIDIA GPU optimization)
# 5-10x faster inference!
import torch_tensorrt
optimized = torch_tensorrt.compile(model, inputs=[example_input])
        """, language="python")


# =============================================================================
# TAB 5: INDUSTRY AI (NEW!)
# =============================================================================
with tab5:
    st.markdown("## 🏭 Chapter 5: Industry-Grade AI Systems")

    st.markdown("""
    <div style="background: #fef2f2; padding: 12px; border-radius: 8px; border-left: 4px solid #dc2626;">
        <h4 style="color: #991b1b; margin: 0 0 8px 0 !important;">⚠️ Production AI ≠ Notebook AI</h4>
        <p style="margin: 0 !important; font-size: 13px;">What works in Jupyter often fails in production. Here's what companies actually build.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🏗️ MLOps: The Full ML Pipeline", expanded=True):
        st.markdown("#### The Complete Production ML Lifecycle")
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [fontname="Arial", fontsize=9, shape=box, style="rounded,filled"];
            data [label="Data\\nIngestion", fillcolor="#dbeafe"];
            feat [label="Feature\\nEngineering", fillcolor="#e0f2fe"];
            train [label="Model\\nTraining", fillcolor="#fef3c7"];
            eval [label="Evaluation\\n& Testing", fillcolor="#fef08a"];
            reg [label="Model\\nRegistry", fillcolor="#dcfce7"];
            deploy [label="Deployment\\n(API/Edge)", fillcolor="#bbf7d0"];
            monitor [label="Monitoring\\n& Alerts", fillcolor="#e9d5ff"];
            data -> feat -> train -> eval -> reg -> deploy -> monitor;
            monitor -> data [style=dashed, label="Feedback Loop"];
        }
        """)
        
        st.markdown("""
        | Stage | Tools | Purpose |
        |---|---|---|
        | **Data** | Airflow, dbt, Great Expectations | ETL, validation |
        | **Features** | Feast, Tecton | Feature store |
        | **Training** | Kubeflow, SageMaker | Distributed training |
        | **Registry** | MLflow, Neptune | Model versioning |
        | **Serving** | BentoML, Seldon | REST/gRPC APIs |
        | **Monitoring** | Evidently, WhyLabs | Drift detection |
        """)

    with st.expander("🤖 LLM Integration Patterns", expanded=True):
        st.markdown("""
        ### How Companies Use GPT/LLMs in Production
        
        | Pattern | Use Case | Example |
        |---|---|---|
        | **Direct API** | Simple Q&A | Customer support chatbot |
        | **RAG** | Knowledge-grounded answers | Internal docs search |
        | **Fine-tuning** | Domain-specific tasks | Legal document analysis |
        | **Agents** | Multi-step reasoning | Automated research assistant |
        """)
        
        st.markdown("#### RAG (Retrieval-Augmented Generation) Pattern")
        st.code("""
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. Index your company's documents
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(documents, embeddings)

# 2. Create retrieval chain
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(k=5),
    return_source_documents=True
)

# 3. Query with context
result = qa_chain("What is our refund policy?")
print(result["answer"])  # Grounded in YOUR documents
print(result["source_documents"])  # Shows sources
        """, language="python")
        
        st.markdown("#### Building AI Agents (Multi-Step Reasoning)")
        st.code("""
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun

# Define tools the agent can use
tools = [
    Tool(name="Search", func=DuckDuckGoSearchRun().run,
         description="Search the web"),
    Tool(name="Calculator", func=eval,
         description="Do math calculations"),
    Tool(name="Database", func=query_database,
         description="Query company database")
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4"),
    agent="zero-shot-react-description",
    verbose=True
)

# Agent decides which tools to use!
agent.run("Find our revenue last quarter and calculate 10% of it")
        """, language="python")

    with st.expander("📊 Model Serving at Scale", expanded=True):
        st.markdown("""
        ### Serving Models to Millions of Users
        
        | Service | Requests/sec | Latency | Solution |
        |---|---|---|---|
        | Small (<100 RPS) | Low | REST API | Flask + Gunicorn |
        | Medium (<10K RPS) | Medium | FastAPI + async | TorchServe |
        | Large (<100K RPS) | Critical | GPU cluster | Triton + K8s |
        | Massive (>100K RPS) | Ultra-low | Edge + CDN | TensorRT + Mobile |
        """)
        
        st.markdown("#### Production API with FastAPI")
        st.code("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model once at startup
model = torch.jit.load("model_production.pt")
model.eval()

class PredictionRequest(BaseModel):
    text: str
    
class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        with torch.no_grad():
            # Preprocessing
            tokens = tokenize(request.text)
            
            # Inference
            logits = model(tokens)
            probs = torch.softmax(logits, dim=-1)
            
            # Postprocessing
            label_idx = probs.argmax().item()
            confidence = probs.max().item()
            
            return PredictionResponse(
                label=LABELS[label_idx],
                confidence=confidence
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn app:app --workers 4
        """, language="python")

    with st.expander("🔍 Model Monitoring & Observability", expanded=False):
        st.markdown("""
        ### Detecting Problems Before Users Notice
        
        | Issue | Detection | Solution |
        |---|---|---|
        | **Data Drift** | Feature distribution shift | Retrain triggers |
        | **Concept Drift** | Relationship changes | A/B testing |
        | **Model Degradation** | Accuracy drops | Alerts + rollback |
        | **Latency Spikes** | P99 > threshold | Auto-scaling |
        """)
        
        st.code("""
from evidently import ColumnDriftReport
from prometheus_client import Counter, Histogram

# Metrics for monitoring
prediction_counter = Counter('predictions_total', 'Total predictions')
latency_histogram = Histogram('prediction_latency', 'Latency')
drift_gauge = Gauge('feature_drift_score', 'Drift score')

@latency_histogram.time()
def predict_with_monitoring(input_data):
    prediction_counter.inc()
    
    # Detect drift periodically
    if should_check_drift():
        report = ColumnDriftReport()
        report.run(reference_data, current_data)
        drift_gauge.set(report.drift_score())
    
    return model.predict(input_data)
        """, language="python")

    with st.expander("☁️ Cloud ML Platforms", expanded=False):
        st.markdown("""
        ### Major Cloud AI/ML Services
        
        | Provider | Training | Serving | Specialty |
        |---|---|---|---|
        | **AWS SageMaker** | SageMaker Training | SageMaker Endpoint | Most features |
        | **GCP Vertex AI** | Vertex Training | Vertex Prediction | Best TPU support |
        | **Azure ML** | Azure ML Compute | AKS/ACI | Enterprise integration |
        | **Databricks** | MLflow + Spark | MLflow Serving | Unified data+ML |
        | **Hugging Face** | AutoTrain | Inference Endpoints | NLP models |
        """)

    with st.expander("📈 Real Company AI Architectures", expanded=True):
        st.markdown("""
        ### How Tech Giants Build AI Systems
        
        #### 🎬 Netflix Recommendation System
        - **Scale:** 200M+ users, billions of interactions
        - **Architecture:** Candidate generation → Ranking → Personalization
        - **Tech:** Spark for ETL, TensorFlow for models, A/B testing for validation
        
        #### 🚗 Tesla Autopilot
        - **Scale:** 8 cameras, 250+ FPS processing
        - **Architecture:** Multi-task CNN → Temporal fusion → Path planning
        - **Tech:** PyTorch for training, custom ASIC (Dojo) for inference
        
        #### 💬 ChatGPT (OpenAI)
        - **Scale:** 175B+ parameters, millions of requests/day
        - **Architecture:** Transformer decoder → RLHF fine-tuning
        - **Tech:** PyTorch, custom distributed training (Megatron-LM)
        
        #### 🔍 Google Search AI
        - **Scale:** Trillions of documents, billions of queries
        - **Architecture:** BERT for understanding → Neural ranking
        - **Tech:** TensorFlow, TPUs, Map/Reduce pipelines
        """)


# =============================================================================
# TAB 6: PROJECT
# =============================================================================
with tab6:
    st.markdown("## 🤖 Chapter 6: Full Production Project")

    st.markdown("""
    <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; border-left: 4px solid #16a34a;">
        <h4 style="color: #166534; margin: 0 0 8px 0 !important;">🎯 Project: End-to-End Sentiment Analysis Service</h4>
        <p style="margin: 0 !important; font-size: 13px;">Build a production-ready sentiment analysis API with model serving, monitoring, and deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Project Architecture", expanded=True):
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
            client [label="Client\\nApp", fillcolor="#dbeafe"];
            api [label="FastAPI\\nService", fillcolor="#fef3c7"];
            model [label="PyTorch\\nModel", fillcolor="#e9d5ff"];
            cache [label="Redis\\nCache", fillcolor="#fecaca"];
            monitor [label="Prometheus\\n+ Grafana", fillcolor="#dcfce7"];
            client -> api; api -> model; api -> cache; api -> monitor;
        }
        """)

    with st.expander("📜 Complete Code: Model Training", expanded=True):
        st.code("""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl

class SentimentClassifier(pl.LightningModule):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        acc = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

# Training
model = SentimentClassifier()
trainer = pl.Trainer(max_epochs=3, accelerator="gpu")
trainer.fit(model, train_dataloader, val_dataloader)

# Export for production
model.eval()
torch.save(model.state_dict(), "sentiment_model.pt")
        """, language="python")

    with st.expander("📜 Complete Code: Production API", expanded=True):
        st.code("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import redis
import hashlib
import time
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI(title="Sentiment Analysis API")

# Load model
model = SentimentClassifier()
model.load_state_dict(torch.load("sentiment_model.pt"))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Redis cache
cache = redis.Redis(host="localhost", port=6379)

# Metrics
REQUEST_COUNT = Counter("requests_total", "Total requests")
LATENCY = Histogram("request_latency_seconds", "Request latency")
CACHE_HITS = Counter("cache_hits_total", "Cache hits")

class TextInput(BaseModel):
    text: str
    
class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float
    cached: bool

LABELS = ["negative", "neutral", "positive"]

@app.post("/predict", response_model=SentimentOutput)
async def predict_sentiment(input: TextInput):
    REQUEST_COUNT.inc()
    start = time.time()
    
    # Check cache
    cache_key = hashlib.md5(input.text.encode()).hexdigest()
    cached = cache.get(cache_key)
    if cached:
        CACHE_HITS.inc()
        LATENCY.observe(time.time() - start)
        return SentimentOutput(**eval(cached), cached=True)
    
    # Tokenize
    tokens = tokenizer(
        input.text, 
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # Predict
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
        probs = torch.softmax(logits, dim=-1)
        label_idx = probs.argmax().item()
        confidence = probs.max().item()
    
    result = {
        "sentiment": LABELS[label_idx],
        "confidence": round(confidence, 4)
    }
    
    # Cache for 1 hour
    cache.setex(cache_key, 3600, str(result))
    
    LATENCY.observe(time.time() - start)
    return SentimentOutput(**result, cached=False)

@app.get("/metrics")
def metrics():
    return generate_latest()

@app.get("/health")
def health():
    return {"status": "healthy"}
        """, language="python")

    with st.expander("🐳 Dockerfile & Deployment", expanded=False):
        st.code("""
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment_model.pt .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
        """, language="dockerfile")
        
        st.code("""
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - prometheus
      
  redis:
    image: redis:7-alpine
    
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
        """, language="yaml")

    st.success("🎉 You now have the knowledge to build production-grade AI systems used by real companies!")
