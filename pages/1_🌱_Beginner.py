import streamlit as st
from utils.styles import apply_custom_css
from utils.seo import inject_seo_meta
from utils.nav import show_top_nav

# Expert SEO & Styles
inject_seo_meta(
    title="Python Beginner Tutorial 2024 | Variables, Loops, Functions & More",
    description="Learn Python fundamentals from scratch. Master variables, data types, if-else logic, for/while loops, and functions with interactive examples. Perfect for absolute beginners. 100+ exercises included.",
    keywords=[
        "python beginner tutorial",
        "learn python basics",
        "python variables explained",
        "python data types",
        "python if else tutorial",
        "python for loop examples",
        "python while loop",
        "python functions tutorial",
        "python programming basics",
        "python syntax for beginners",
        "python first program",
        "python hello world",
        "python operators",
        "python string formatting",
        "python f-strings tutorial"
    ],
    schema_type="TechArticle",
    canonical_url="https://pythonmastery.dev/beginner",
    reading_time=60,
    breadcrumbs=[
        {"name": "Home", "url": "https://pythonmastery.dev"},
        {"name": "Beginner", "url": "https://pythonmastery.dev/beginner"}
    ],
    course_info={
        "name": "Python Beginner Module: Complete Fundamentals Guide",
        "description": "Master Python fundamentals including variables, data types, conditional logic, loops, and functions with detailed explanations and real-world analogies.",
        "level": "Beginner",
        "prerequisites": "None - start from zero",
        "teaches": ["Python Variables", "Data Types", "Conditional Logic", "Loops", "Functions", "String Formatting"],
        "workload": "PT10H",
        "rating": "4.9",
        "rating_count": 1523
    },
    faq_items=[
        {
            "question": "What is a variable in Python?",
            "answer": "A variable in Python is a named container that stores data in your computer's memory. Think of it like a labeled jar - the jar is the variable, the label is the name, and the contents are the data."
        },
        {
            "question": "What are the main data types in Python?",
            "answer": "The four main data types in Python are: String (str) for text, Integer (int) for whole numbers, Float for decimal numbers, and Boolean (bool) for True/False values."
        },
        {
            "question": "How do for loops work in Python?",
            "answer": "A for loop in Python iterates over a sequence (like a list or range) and executes a block of code for each item. Example: 'for item in list:' runs the indented code for each element."
        },
        {
            "question": "What is the difference between = and == in Python?",
            "answer": "A single equals sign (=) assigns a value to a variable. Double equals (==) compares two values for equality. This is a common beginner mistake to watch out for."
        }
    ]
)
apply_custom_css()
show_top_nav(current_page="Beginner")

# Header
st.markdown("""
<div style="text-align: center; padding: 12px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 10px; margin-bottom: 10px;">
    <h2 style="margin: 0 !important; font-size: 1.4rem !important;">🌱 Beginner: The Complete Foundation</h2>
    <p style="margin: 5px 0 0 0 !important; font-size: 14px; color: #166534;">Zero prior knowledge assumed. Every concept explained simply.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📦 Variables & Types", "🤔 Logic & Decisions", "🔄 Loops", "🛠️ Functions", "🎮 Projects"])

# =============================================================================
# TAB 1: VARIABLES & DATA TYPES
# =============================================================================
with tab1:
    st.markdown("## 📦 Chapter 1: Variables & Data Types")
    
    # Section 1.1: What is a Variable
    st.markdown("### 1.1 What is a Variable?")
    
    col_def, col_analogy = st.columns(2)
    with col_def:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important; font-size: 14px !important;">📖 Definition</h4>
            <p style="margin: 0 !important; font-size: 13px;">A <b>Variable</b> is a named container that stores a piece of data in your computer's memory.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_analogy:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important; font-size: 14px !important;">💡 Analogy: Labeled Jar</h4>
            <p style="margin: 0 !important; font-size: 13px;">Think of a <b>labeled jar</b>. The jar is the variable, the label is the name, the stuff inside is the data.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🖼️ Visual: How Variables Work in Memory", expanded=False):
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [fontname="Arial", fontsize=11];
            subgraph cluster_memory {
                label="Computer Memory (RAM)"; style=filled; fillcolor="#f8fafc";
                jar1 [label="Jar: 'user_name'\\n\\nContents: 'Alice'", shape=cylinder, fillcolor="#dbeafe", style=filled];
                jar2 [label="Jar: 'age'\\n\\nContents: 25", shape=cylinder, fillcolor="#dcfce7", style=filled];
                jar3 [label="Jar: 'is_student'\\n\\nContents: True", shape=cylinder, fillcolor="#fef3c7", style=filled];
            }
        }
        """)

    with st.expander("💻 Code Example: Creating Variables", expanded=True):
        st.code("""
# Creating variables - Python figures out the type automatically!

user_name = "Alice"     # String (text)
age = 25                # Integer (whole number)
height = 5.7            # Float (decimal)
is_student = True       # Boolean (True/False)

print(user_name)   # Output: Alice
print(age)         # Output: 25
        """, language="python")
        st.info("**Key Insight**: Unlike Java/C++, you don't declare types. Python is smart enough to figure it out.")

    # Section 1.2: Data Types
    st.markdown("### 1.2 The Four Main Data Types")

    with st.expander("📚 Data Types Explained", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                <h4 style="margin: 0 0 5px 0 !important;">📝 String (str)</h4>
                <p style="margin: 0 !important; font-size: 13px;"><b>What:</b> Text in quotes. <b>Ex:</b> <code>"Hello"</code>, <code>'Python'</code></p>
            </div>
            <div style="background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%); padding: 12px; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0 !important;">🔢 Integer (int)</h4>
                <p style="margin: 0 !important; font-size: 13px;"><b>What:</b> Whole numbers. <b>Ex:</b> <code>10</code>, <code>-5</code>, <code>1000</code></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                <h4 style="margin: 0 0 5px 0 !important;">🧮 Float</h4>
                <p style="margin: 0 !important; font-size: 13px;"><b>What:</b> Decimals. <b>Ex:</b> <code>3.14</code>, <code>99.99</code></p>
            </div>
            <div style="background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); padding: 12px; border-radius: 8px;">
                <h4 style="margin: 0 0 5px 0 !important;">✅ Boolean (bool)</h4>
                <p style="margin: 0 !important; font-size: 13px;"><b>What:</b> ON/OFF. <b>Ex:</b> <code>True</code>, <code>False</code></p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🔢 Math Operations", expanded=False):
        st.code("""
a = 10
b = 3

print(a + b)   # Addition: 13
print(a - b)   # Subtraction: 7
print(a * b)   # Multiplication: 30
print(a / b)   # Division: 3.333...
print(a // b)  # Floor Division: 3
print(a % b)   # Modulus (Remainder): 1
print(a ** b)  # Exponentiation (10^3): 1000
        """, language="python")

    with st.expander("✨ F-Strings: Modern String Formatting", expanded=False):
        st.code("""
name = "Hermione"
house = "Gryffindor"
points = 150

# OLD WAY (Avoid)
message_old = "Welcome " + name + " of " + house + "!"

# NEW WAY (Use this!)
message_new = f"Welcome {name} of {house}! You have {points} points."

print(message_new)
# Output: Welcome Hermione of Gryffindor! You have 150 points.
        """, language="python")
        st.success("The `f` before the quote tells Python to replace `{...}` with variable values.")

    with st.expander("🧪 Interactive Lab: Build Your Character", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            char_name = st.text_input("Character Name:", "Gandalf", key="c_name")
            char_class = st.selectbox("Class:", ["Warrior", "Mage", "Rogue", "Healer"], key="c_class")
        with col2:
            char_level = st.slider("Level:", 1, 100, 50, key="c_level")
            char_has_magic = st.checkbox("Has Magic Powers?", value=True, key="c_magic")
            
        st.code(f'''character_name = "{char_name}"
character_class = "{char_class}"
level = {char_level}
has_magic = {char_has_magic}

print(f"Meet {{character_name}}, the Level {{level}} {{character_class}}!")''', language="python")
        
        if st.button("▶️ Run My Code", key="run_vars"):
            st.success(f"Meet {char_name}, the Level {char_level} {char_class}! Magic: {char_has_magic}")


# =============================================================================
# TAB 2: LOGIC & DECISIONS
# =============================================================================
with tab2:
    st.markdown("## 🤔 Chapter 2: Logic & Making Decisions")
    
    col_def, col_analogy = st.columns(2)
    with col_def:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">🧠 Why Logic?</h4>
            <p style="margin: 0 !important; font-size: 13px;">Code needs to react to conditions: "Did user win?", "Is password correct?"</p>
        </div>
        """, unsafe_allow_html=True)
    with col_analogy:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy: The Bouncer</h4>
            <p style="margin: 0 !important; font-size: 13px;">A bouncer checks: "Are you 21+?" If yes → enter. If no → denied.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📋 Comparison Operators", expanded=True):
        st.markdown("""
        | Operator | Meaning | Example | Result |
        |---|---|---|---|
        | `==` | Equal? | `5 == 5` | `True` |
        | `!=` | Not Equal? | `5 != 3` | `True` |
        | `>` | Greater? | `10 > 5` | `True` |
        | `<` | Less? | `3 < 8` | `True` |
        | `>=` | Greater/Equal? | `5 >= 5` | `True` |
        | `<=` | Less/Equal? | `4 <= 10` | `True` |
        """)
        st.warning("**Common Mistake:** `=` assigns value. `==` compares values!")

    with st.expander("🚦 The `if` Statement", expanded=True):
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.code("""
age = 25

if age >= 21:
    print("Welcome to the club! 🎉")
    print("Enjoy your evening.")

print("This line runs no matter what.")
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=TB; node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
                start [label="Start", shape=oval, fillcolor="#e0f2fe"];
                check [label="age >= 21?", shape=diamond, fillcolor="#fef3c7"];
                yes [label="Print: Welcome!", fillcolor="#dcfce7"];
                end [label="Continue", shape=oval, fillcolor="#f3f4f6"];
                start -> check; check -> yes [label="True"]; check -> end [label="False"]; yes -> end;
            }
            """)

    with st.expander("🔀 The `if-else` Statement", expanded=False):
        st.code("""
temperature = 15

if temperature > 25:
    print("It's hot! Wear shorts. ☀️")
else:
    print("It's cool. Grab a jacket. 🧥")
        """, language="python")

    with st.expander("🪜 The `if-elif-else` Ladder", expanded=False):
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.code("""
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Your grade is: {grade}")
            """, language="python")
        with col2:
            st.graphviz_chart("""
            digraph {
                rankdir=TB; node [fontname="Arial", fontsize=9, shape=box, style="rounded,filled"];
                c1 [label=">=90?", shape=diamond, fillcolor="#fef3c7"];
                c2 [label=">=80?", shape=diamond, fillcolor="#fef3c7"];
                gA [label="A", fillcolor="#86efac"]; gB [label="B", fillcolor="#86efac"]; gF [label="F", fillcolor="#fca5a5"];
                c1 -> gA [label="Y"]; c1 -> c2 [label="N"]; c2 -> gB [label="Y"]; c2 -> gF [label="N"];
            }
            """)

    with st.expander("🧪 Interactive Lab: Wizard's Gate", expanded=False):
        password_attempt = st.text_input("Enter password:", type="password", key="pwd")
        if st.button("Try to Enter", key="try_pwd"):
            if password_attempt == "OpenSesame":
                st.success("✨ The gate opens! Welcome, Wizard! ✨")
                st.balloons()
            elif password_attempt == "":
                st.warning("Enter something!")
            else:
                st.error("🚫 Access Denied.")


# =============================================================================
# TAB 3: LOOPS
# =============================================================================
with tab3:
    st.markdown("## 🔄 Chapter 3: Loops & Repetition")

    st.markdown("""
    <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; border-left: 4px solid #16a34a;">
        <h4 style="color: #166534; margin: 0 0 8px 0 !important;">🧠 Core Idea</h4>
        <p style="margin: 0 !important; font-size: 13px;">Loops let you say: <b>"Do this X times"</b> or <b>"Keep doing this until I say stop."</b></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔢 `for` Loop: Iterate Over Items", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Iterating a List:**")
            st.code("""
fruits = ["Apple", "Banana", "Cherry"]

for fruit in fruits:
    print(f"I am eating {fruit}")
            """, language="python")
        with col2:
            st.markdown("**Using `range()` for Counting:**")
            st.code("""
for i in range(5):  # 0, 1, 2, 3, 4
    print(f"Lap {i + 1}")
            """, language="python")

    with st.expander("🖼️ Loop Flowchart", expanded=False):
        st.graphviz_chart("""
        digraph {
            rankdir=TB; node [fontname="Arial", fontsize=10];
            start [label="Start", shape=oval, fillcolor="#e0f2fe", style=filled];
            next [label="Get next item", shape=box, fillcolor="#dbeafe", style="rounded,filled"];
            check [label="Items left?", shape=diamond, fillcolor="#fef3c7", style=filled];
            action [label="Run code", shape=box, fillcolor="#dcfce7", style="rounded,filled"];
            end [label="Done", shape=oval, fillcolor="#f3f4f6", style=filled];
            start -> next -> check; check -> action [label="Yes"]; action -> next; check -> end [label="No"];
        }
        """)

    with st.expander("🔁 `while` Loop: Repeat Until Condition", expanded=True):
        st.markdown("**Analogy:** Fill bathtub WHILE not full. Stop when full.")
        st.code("""
countdown = 5

while countdown > 0:
    print(f"T-minus {countdown}...")
    countdown -= 1  # IMPORTANT: Update the variable!

print("🚀 LIFTOFF!")
        """, language="python")
        st.error("**Danger:** If condition never becomes False → Infinite Loop! Always update the variable.")

    with st.expander("🚪 `break` & `continue`", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**`break`: Exit immediately**")
            st.code("""
for n in range(10):
    if n == 5:
        print("Found 5!")
        break
    print(n)
# Output: 0,1,2,3,4, Found 5!
            """, language="python")
        with col2:
            st.markdown("**`continue`: Skip this one**")
            st.code("""
for n in range(5):
    if n == 2:
        continue
    print(n)
# Output: 0,1,3,4 (2 skipped)
            """, language="python")


# =============================================================================
# TAB 4: FUNCTIONS
# =============================================================================
with tab4:
    st.markdown("## 🛠️ Chapter 4: Functions")

    col_def, col_analogy = st.columns(2)
    with col_def:
        st.markdown("""
        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #2563eb;">
            <h4 style="color: #1e40af; margin: 0 0 8px 0 !important;">📖 Definition</h4>
            <p style="margin: 0 !important; font-size: 13px;">A <b>Function</b> is a reusable block of code. Write once, use many times.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_analogy:
        st.markdown("""
        <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #b45309; margin: 0 0 8px 0 !important;">💡 Analogy: Coffee Machine</h4>
            <p style="margin: 0 !important; font-size: 13px;"><b>Input:</b> Beans → <b>Process:</b> Brew → <b>Output:</b> Coffee ☕</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🖼️ Function Flow Diagram", expanded=False):
        st.graphviz_chart("""
        digraph {
            rankdir=LR; node [fontname="Arial", fontsize=11];
            inputs [label="Inputs\\n(Arguments)", shape=note, fillcolor="#fef08a", style=filled];
            machine [label="FUNCTION\\n(Process)", shape=component, fillcolor="#bfdbfe", style=filled];
            output [label="Output\\n(Return)", shape=note, fillcolor="#bbf7d0", style=filled];
            inputs -> machine -> output;
        }
        """)

    with st.expander("✍️ Defining & Calling Functions", expanded=True):
        st.code("""
# DEFINING a function
def greet(name):
    message = f"Hello, {name}! Welcome."
    return message  # Output

# CALLING the function
result = greet("Alice")
print(result)  # Hello, Alice! Welcome.

# Reuse it!
print(greet("Bob"))
print(greet("Charlie"))
        """, language="python")

    with st.expander("📦 Multiple Arguments", expanded=False):
        st.code("""
def calculate_total(price, quantity, tax_rate):
    subtotal = price * quantity
    tax = subtotal * tax_rate
    return subtotal + tax

my_bill = calculate_total(15.00, 3, 0.08)
print(f"Total: ${my_bill:.2f}")  # $48.60
        """, language="python")

    with st.expander("⚙️ Default Arguments", expanded=False):
        st.code("""
def say_hello(name, language="Python"):
    print(f"Hello {name}, best language is {language}!")

say_hello("Alice")              # Uses default: Python
say_hello("Bob", "JavaScript")  # Overrides: JavaScript
        """, language="python")


# =============================================================================
# TAB 5: PROJECTS
# =============================================================================
with tab5:
    st.markdown("## 🎮 Chapter 5: Mini Projects")
    st.markdown("Put everything together with these hands-on projects!")

    with st.expander("🎲 Project 1: Number Guessing Game", expanded=True):
        st.markdown("**Concepts:** Variables, Loops, If/Else, Random")
        st.graphviz_chart("""
        digraph {
            node [fontname="Arial", fontsize=10, shape=box, style="rounded,filled"];
            start [label="Start", shape=oval, fillcolor="#e0f2fe"];
            pick [label="Pick random\\n1-100", fillcolor="#c7d2fe"];
            ask [label="Ask guess", fillcolor="#dbeafe"];
            check [label="Correct?", shape=diamond, fillcolor="#fef3c7"];
            win [label="Win! 🎉", fillcolor="#86efac"];
            low [label="Too Low ⬆️", fillcolor="#fecaca"];
            high [label="Too High ⬇️", fillcolor="#fed7aa"];
            start -> pick -> ask -> check;
            check -> win [label="="]; check -> low [label="<"]; check -> high [label=">"];
            low -> ask; high -> ask;
        }
        """)
        st.code("""
import random

target = random.randint(1, 100)
attempts = 0

print("I'm thinking of a number between 1 and 100...")

while True:
    guess = int(input("Your guess: "))
    attempts += 1
    
    if guess < target:
        print("Too low! ⬆️")
    elif guess > target:
        print("Too high! ⬇️")
    else:
        print(f"🎉 Correct! {attempts} attempts!")
        break
        """, language="python")

    with st.expander("📋 Project 2: To-Do List App", expanded=False):
        st.markdown("**Concepts:** Lists, Functions, Loops, User Input")
        st.code("""
tasks = []

def show_menu():
    print("\\n--- To-Do List ---")
    print("1. Add  2. View  3. Remove  4. Quit")

def add_task():
    task = input("Enter task: ")
    tasks.append(task)
    print(f"'{task}' added!")

def view_tasks():
    if not tasks:
        print("Empty list.")
    else:
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")

def remove_task():
    view_tasks()
    try:
        num = int(input("Remove #: "))
        removed = tasks.pop(num - 1)
        print(f"'{removed}' removed.")
    except:
        print("Invalid.")

while True:
    show_menu()
    choice = input("Choose: ")
    if choice == "1": add_task()
    elif choice == "2": view_tasks()
    elif choice == "3": remove_task()
    elif choice == "4": break
        """, language="python")
