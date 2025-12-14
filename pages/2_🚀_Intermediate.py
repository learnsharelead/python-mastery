import streamlit as st
from utils.styles import apply_custom_css
from utils.seo import inject_seo_meta
from utils.nav import show_top_nav

# Expert SEO & Styles
inject_seo_meta(
    title="Python Intermediate Tutorial | OOP, File I/O, Modules & Error Handling",
    description="Level up your Python skills with Object-Oriented Programming, file handling, exception handling, and modular code. Build real applications with classes, inheritance, and external packages.",
    keywords=[
        "python OOP tutorial",
        "python classes and objects",
        "python inheritance example",
        "python file handling",
        "python read write files",
        "python exception handling",
        "python try except",
        "python modules import",
        "python pip packages",
        "python data structures",
        "python dictionary tutorial",
        "python list comprehension",
        "python CSV JSON parsing",
        "object oriented python",
        "python error handling best practices"
    ],
    schema_type="TechArticle",
    canonical_url="https://pythonmastery.dev/intermediate",
    reading_time=75,
    breadcrumbs=[
        {"name": "Home", "url": "https://pythonmastery.dev"},
        {"name": "Intermediate", "url": "https://pythonmastery.dev/intermediate"}
    ],
    course_info={
        "name": "Python Intermediate Module: OOP, Files & Modules",
        "description": "Level up with Object Oriented Programming, File I/O, Error Handling, and Modules. Learn to structure code for real-world applications.",
        "level": "Intermediate",
        "prerequisites": "Basic Python knowledge (variables, loops, functions)",
        "teaches": ["OOP", "Classes", "Inheritance", "File I/O", "Exception Handling", "Modules", "pip"],
        "workload": "PT12H",
        "rating": "4.8",
        "rating_count": 987
    },
    faq_items=[
        {
            "question": "What is Object-Oriented Programming (OOP) in Python?",
            "answer": "OOP is a programming paradigm where you create 'classes' as blueprints and 'objects' as instances. A class defines attributes (data) and methods (functions) that belong together. Example: a Dog class with name attribute and bark() method."
        },
        {
            "question": "What is the difference between a list and dictionary in Python?",
            "answer": "Lists are ordered collections accessed by index (0, 1, 2...), like numbered slots. Dictionaries store key-value pairs accessed by labels, like labeled drawers. Use lists for ordered data, dictionaries for named lookups."
        },
        {
            "question": "How do I handle errors in Python without crashing?",
            "answer": "Use try/except blocks. Put risky code in 'try:', and handle the error in 'except:'. You can catch specific errors like 'except ValueError:' or use 'finally:' for cleanup code that always runs."
        },
        {
            "question": "What is the 'with' statement used for in Python?",
            "answer": "The 'with' statement is used for resource management, automatically handling setup and cleanup. Most commonly used with files: 'with open(file) as f:' ensures the file is properly closed even if an error occurs."
        }
    ]
)
apply_custom_css()
show_top_nav(current_page="Intermediate")

# Header
st.markdown("""
<div style="text-align: center; padding: 12px; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 10px; margin-bottom: 10px;">
    <h2 style="margin: 0 !important; font-size: 1.4rem !important;">🚀 Intermediate: Building Real Applications</h2>
    <p style="margin: 5px 0 0 0 !important; font-size: 14px; color: #1e40af;">From "Scripts" to "Software". Structure your code for the real world.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Data Structures", "🤖 OOP", "📂 Files", "🛡️ Errors", "📦 Modules"])

# =============================================================================
# TAB 1: DATA STRUCTURES
# =============================================================================
with tab1:
    st.markdown("## 📚 Chapter 1: Data Structures")

    st.markdown("""
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">🧠 Core Idea</h4>
        <p style="margin: 0 !important; font-size: 13px;">Store multiple related pieces of data together. <b>Lists</b> = ordered by number. <b>Dicts</b> = accessed by label.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Lists: The Ordered Shelf", expanded=True):
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("**Analogy:** Bookshelf with numbered slots (0, 1, 2...).")
            st.code("""
fruits = ["Apple", "Banana", "Cherry"]

print(fruits[0])   # Apple (first)
print(fruits[-1])  # Cherry (last)

fruits.append("Date")       # Add to end
fruits.insert(0, "Apricot") # Add at position
fruits.remove("Banana")     # Remove by value
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=TB; node [fontname="Arial", fontsize=10, shape=record, style=filled, fillcolor="#e0f2fe"];
                list [label="<0> 0 | <1> 1 | <2> 2"];
                data [label="<0> Apple | <1> Banana | <2> Cherry", fillcolor="#fff"];
                list:0 -> data:0; list:1 -> data:1; list:2 -> data:2;
            }
            """)

    with st.expander("🏷️ Dictionaries: The Filing Cabinet", expanded=True):
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("**Analogy:** Labeled drawers. Access by name, not position.")
            st.code("""
user = {
    "name": "Alice",
    "age": 30,
    "city": "London"
}

print(user["name"])     # Alice
user["age"] = 31        # Update
user["email"] = "a@b.c" # Add new key
del user["city"]        # Remove

for key, val in user.items():
    print(f"{key}: {val}")
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=LR; node [fontname="Arial", fontsize=10, shape=record, style=filled];
                dict [label="{ {name | Alice} | {age | 30} | {city | London} }", fillcolor="#dcfce7"];
            }
            """)

    with st.expander("🔒 Tuples & Sets", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tuples:** Immutable (can't change after creation).")
            st.code('coords = (10.5, 20.3)\nprint(coords[0])  # 10.5\n# coords[0] = 99  # ERROR!', language="python")
        with col2:
            st.markdown("**Sets:** Unique items only.")
            st.code('tags = {"a", "b", "a"}\nprint(tags)  # {"a", "b"}\n# Duplicates auto-removed', language="python")


# =============================================================================
# TAB 2: OOP
# =============================================================================
with tab2:
    st.markdown("## 🤖 Chapter 2: Object Oriented Programming")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 What is OOP?</h4>
            <p style="margin: 0 !important; font-size: 13px;">Create a <b>blueprint (Class)</b> and build <b>instances (Objects)</b> from it.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy</h4>
            <p style="margin: 0 !important; font-size: 13px;"><b>Class</b> = Car blueprint. <b>Object</b> = Your red Toyota.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 Key Terminology", expanded=True):
        st.markdown("""
        | Term | Meaning | Example |
        |---|---|---|
        | **Class** | Blueprint | `class Dog:` |
        | **Object** | Instance from blueprint | `buddy = Dog()` |
        | **Attribute** | Variable inside object | `self.name` |
        | **Method** | Function inside object | `def bark(self):` |
        | `__init__` | Constructor (auto-runs on creation) | Sets initial values |
        | `self` | Reference to current object | "My name is..." |
        """)

    with st.expander("🖼️ Class Diagram", expanded=False):
        st.graphviz_chart("""
        digraph {
            rankdir=TB; node [fontname="Arial", fontsize=10, shape=record, style=filled];
            subgraph cluster_class {
                label="CLASS: Dog"; style=filled; fillcolor="#eff6ff";
                attrs [label="Attributes\\n---\\nname, breed, age", fillcolor="#dbeafe"];
                methods [label="Methods\\n---\\nbark(), fetch()", fillcolor="#bfdbfe"];
            }
            subgraph cluster_objs {
                label="OBJECTS"; style=filled; fillcolor="#f0fdf4";
                o1 [label="buddy\\nname='Buddy'", fillcolor="#dcfce7"];
                o2 [label="max\\nname='Max'", fillcolor="#dcfce7"];
            }
            attrs -> o1 [style=dashed]; attrs -> o2 [style=dashed];
        }
        """)

    with st.expander("💻 Writing Your First Class", expanded=True):
        st.code("""
class Dog:
    def __init__(self, name, breed, age):
        self.name = name    # Attribute
        self.breed = breed
        self.age = age
        print(f"🐕 {self.name} was born!")
    
    def bark(self):         # Method
        return f"{self.name} says: WOOF!"
    
    def birthday(self):
        self.age += 1
        return f"Happy Birthday {self.name}! Now {self.age}."

# Create objects
buddy = Dog("Buddy", "Golden", 3)
max_dog = Dog("Max", "Beagle", 5)

print(buddy.bark())        # Buddy says: WOOF!
print(max_dog.birthday())  # Happy Birthday Max! Now 6.
        """, language="python")

    with st.expander("🧬 Inheritance: Parent → Child", expanded=False):
        st.markdown("Child class inherits from parent. Can override methods.")
        st.code("""
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return "..."

class Cat(Animal):
    def speak(self):
        return f"{self.name} says: Meow!"

class Dog(Animal):
    def speak(self):
        return f"{self.name} says: Woof!"

whiskers = Cat("Whiskers")
rex = Dog("Rex")
print(whiskers.speak())  # Meow!
print(rex.speak())       # Woof!
        """, language="python")


# =============================================================================
# TAB 3: FILES
# =============================================================================
with tab3:
    st.markdown("## 📂 Chapter 3: File Input/Output")

    st.markdown("""
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">🧠 Why Files?</h4>
        <p style="margin: 0 !important; font-size: 13px;">Variables die when script ends. Files = <b>permanent storage</b> on disk.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 Reading Files", expanded=True):
        st.code("""
# 'with' auto-closes file (even if error!)
with open("diary.txt", "r") as file:
    content = file.read()  # Entire content as string
    print(content)

# Read line by line
with open("diary.txt", "r") as file:
    for line in file:
        print(line.strip())
        """, language="python")

    with st.expander("✍️ Writing Files", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**`'w'` Mode: Overwrite**")
            st.warning("⚠️ Erases existing content!")
            st.code('with open("notes.txt", "w") as f:\n    f.write("New content")', language="python")
        with col2:
            st.markdown("**`'a'` Mode: Append**")
            st.success("✅ Adds to end, keeps existing.")
            st.code('with open("log.txt", "a") as f:\n    f.write("New entry\\n")', language="python")

    with st.expander("📊 CSV Files", expanded=False):
        st.code("""
import csv

# Write CSV
data = [["Name", "Age"], ["Alice", 30], ["Bob", 25]]
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Read CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)  # ['Name', 'Age'], ['Alice', '30']...
        """, language="python")

    with st.expander("🔧 JSON Files", expanded=False):
        st.code("""
import json

# Write JSON
user = {"name": "Alice", "level": 42}
with open("user.json", "w") as f:
    json.dump(user, f, indent=4)

# Read JSON
with open("user.json", "r") as f:
    loaded = json.load(f)
    print(loaded["name"])  # Alice
        """, language="python")


# =============================================================================
# TAB 4: ERROR HANDLING
# =============================================================================
with tab4:
    st.markdown("## 🛡️ Chapter 4: Error Handling")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #fef2f2; padding: 12px; border-radius: 8px; border-left: 4px solid #dc2626;">
            <h4 style="color: #991b1b; margin: 0 0 8px 0 !important;">💥 The Problem</h4>
            <p style="margin: 0 !important; font-size: 13px;">User types "abc" when you expect number → <b>Crash!</b></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; border-left: 4px solid #16a34a;">
            <h4 style="color: #166534; margin: 0 0 8px 0 !important;">🛡️ The Solution</h4>
            <p style="margin: 0 !important; font-size: 13px;"><code>try/except</code> = Safety net. Catch errors gracefully.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🎯 Basic try/except", expanded=True):
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.code("""
try:
    num = int(input("Enter number: "))
    print(f"You entered: {num}")
except ValueError:
    print("That's not a valid number!")

print("Program continues...")
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=TB; node [fontname="Arial", fontsize=9, shape=box, style="rounded,filled"];
                t [label="try block", fillcolor="#dbeafe"];
                c [label="Error?", shape=diamond, fillcolor="#fef3c7"];
                e [label="except block", fillcolor="#dcfce7"];
                f [label="continue", shape=oval, fillcolor="#f3f4f6"];
                t -> c; c -> e [label="Yes"]; c -> f [label="No"]; e -> f;
            }
            """)

    with st.expander("🎭 Multiple Exception Types", expanded=False):
        st.code("""
try:
    with open("data.txt", "r") as f:
        num = int(f.read())
        result = 100 / num

except FileNotFoundError:
    print("File doesn't exist!")
except ValueError:
    print("File doesn't contain a number!")
except ZeroDivisionError:
    print("Can't divide by zero!")
except Exception as e:
    print(f"Unexpected: {e}")
        """, language="python")

    with st.expander("🔒 The `finally` Block", expanded=False):
        st.code("""
try:
    print("Connecting...")
    raise Exception("Connection failed!")
except Exception as e:
    print(f"Error: {e}")
finally:
    # ALWAYS runs, error or not
    print("Cleanup: Closing resources...")
        """, language="python")


# =============================================================================
# TAB 5: MODULES
# =============================================================================
with tab5:
    st.markdown("## 📦 Chapter 5: Modules & Packages")

    st.markdown("""
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">🧠 Core Idea</h4>
        <p style="margin: 0 !important; font-size: 13px;">Split code into files. <b>Import</b> what you need. Like chapters in a book.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔧 Built-in Modules", expanded=True):
        st.code("""
import math
print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.14159...

import random
print(random.randint(1, 100))
print(random.choice(["A", "B", "C"]))

from datetime import datetime
print(datetime.now())
        """, language="python")

    with st.expander("📁 Create Your Own Module", expanded=False):
        st.markdown("**File: `my_utils.py`**")
        st.code('PI = 3.14159\n\ndef greet(name):\n    return f"Hello, {name}!"', language="python")
        st.markdown("**File: `main.py`**")
        st.code('import my_utils\nprint(my_utils.greet("Alice"))\nprint(my_utils.PI)\n\n# Or specific import\nfrom my_utils import greet\nprint(greet("Bob"))', language="python")

    with st.expander("📥 Install External Packages (pip)", expanded=False):
        st.code("pip install requests pandas flask pytest", language="bash")
        st.code("""
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200
        """, language="python")
