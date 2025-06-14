import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Page config
st.set_page_config(page_title="🛠️ SDLC Assistant", layout="wide", page_icon="🛠️")

# Custom CSS for modern UI
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .card {
            background-color: #ffffff;
            padding: 25px;
            margin: 20px 0;
            border-left: 6px solid #2ecc71;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .navbar {
            display: flex;
            justify-content: center;
            gap: 20px;
            padding: 15px 0;
            background: linear-gradient(to right, #2ecc71, #27ae60);
            border-radius: 10px;
            margin-bottom: 25px;
        }
        .nav-button {
            background-color: #ffffff;
            color: #2ecc71;
            border: none;
            width: 50px;
            height: 50px;
            font-size: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .nav-button:hover {
            background-color: #eafaf1;
            transform: scale(1.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        label {
            font-weight: bold;
            color: #34495e;
        }
        input, select, textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
            width: 100%;
            font-size: 14px;
        }
        button {
            background-color: #2ecc71;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 14px;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover {
            background-color: #27ae60;
        }
        .chat-bubble-user, .chat-bubble-bot {
            padding: 10px 15px;
            border-radius: 12px;
            max-width: 70%;
            margin: 6px 0;
            font-size: 14px;
        }
        .chat-bubble-user {
            background-color: #d6eaff;
            align-self: flex-end;
        }
        .chat-bubble-bot {
            background-color: #e6f0ff;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tasks" not in st.session_state:
    st.session_state.tasks = []

# Load Watsonx credentials from secrets
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]

    model_map = {
        "requirements": "ibm/granite-3-2-13b-instruct",
        "design": "ibm/granite-3-2-8b-instruct",
        "development": "ibm/granite-3-2-code-instruct",
        "testing": "ibm/granite-3-2-3b-instruct",
        "deployment": "ibm/granite-3-2-8b-instruct",
        "maintenance": "ibm/granite-3-2-13b-instruct",
        "chat": "ibm/granite-3-2-3b-instruct"
    }

    def get_llm(model_name):
        return WatsonxLLM(
            model_id=model_map[model_name],
            url=credentials.get("url"),
            apikey=credentials.get("apikey"),
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.TEMPERATURE: 0,
                GenParams.MIN_NEW_TOKENS: 5,
                GenParams.MAX_NEW_TOKENS: 250,
                GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
            },
        )

except KeyError:
    st.warning("⚠️ Watsonx credentials missing.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

# Navigation Bar with Circular Buttons
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    if st.button("🏠", key="btn_home"):
        st.session_state.current_section = "home"
with col2:
    if st.button("📝", key="btn_requirements"):
        st.session_state.current_section = "requirements"
with col3:
    if st.button("🎨", key="btn_design"):
        st.session_state.current_section = "design"
with col4:
    if st.button("🧱", key="btn_development"):
        st.session_state.current_section = "development"
with col5:
    if st.button("🧪", key="btn_testing"):
        st.session_state.current_section = "testing"
with col6:
    if st.button("🚀", key="btn_deployment"):
        st.session_state.current_section = "deployment"
with col7:
    if st.button("⚙️", key="btn_maintenance"):
        st.session_state.current_section = "maintenance"
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align:center;">🛠️ SDLC Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px;">Manage your software development lifecycle with AI.</p>', unsafe_allow_html=True)

# Section Renderer
def render_section(title, content):
    st.markdown(f'<div class="card">{title}</div>', unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)

# ------------------------------ HOME PAGE ------------------------------
if st.session_state.current_section == "home":
    render_section("<h2>🛠️ Welcome to Your SDLC Assistant</h2>", """
        This application helps you manage software development projects comprehensively — from planning to deployment.
        ### 🧠 Highlights:
        - 📝 Requirement gathering & prioritization  
        - 🧩 Task management (Kanban, Scrum)  
        - 🔍 Test case creation & tracking  
        - 🚀 Deployment automation  
        - 🤖 AI Code generation  
        Get started by exploring any of the tools above!
    """)

# ------------------------------ REQUIREMENTS ------------------------------
elif st.session_state.current_section == "requirements":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Requirements Gathering</h2>', unsafe_allow_html=True)
    req = st.text_area("Enter requirement:")
    if st.button("Add Requirement"):
        st.session_state.tasks.append(req)
        st.success("Requirement added!")
    priority = st.radio("Prioritization Method", ["MoSCoW", "Kano Model"])
    if st.button("Analyze Requirements"):
        llm = get_llm("requirements")
        res = llm.invoke(f"Use {priority} to prioritize:\n{req}")
        st.markdown(f"🤖 **AI Analysis:**\n{res}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ DESIGN ------------------------------
elif st.session_state.current_section == "design":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🎨 UI/UX & Architecture Design</h2>', unsafe_allow_html=True)
    design_prompt = st.text_input("Describe UI or architecture idea:")
    if st.button("Generate Design Idea"):
        llm = get_llm("design")
        response = llm.invoke(design_prompt)
        st.code(response)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ DEVELOPMENT ------------------------------
elif st.session_state.current_section == "development":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🧱 AI Code Generator</h2>', unsafe_allow_html=True)
    prompt = st.text_area("Describe what code you need:")
    if st.button("Generate Code"):
        llm = get_llm("development")
        code = llm.invoke(prompt)
        st.code(code)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ TESTING ------------------------------
elif st.session_state.current_section == "testing":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🧪 Automated Test Case Generation</h2>', unsafe_allow_html=True)
    function_desc = st.text_input("Function description:")
    if st.button("Generate Test Cases"):
        llm = get_llm("testing")
        test_cases = llm.invoke(f"Write test cases for: {function_desc}")
        st.markdown(f"🔍 **Generated Tests:**\n{test_cases}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ DEPLOYMENT ------------------------------
elif st.session_state.current_section == "deployment":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🚀 Deployment Automation</h2>', unsafe_allow_html=True)
    env = st.selectbox("Environment", ["Development", "Staging", "Production"])
    if st.button("Deploy"):
        st.success(f"Deploying to {env}...")
        advice = llm.invoke(f"How to deploy to {env}?")
        st.markdown(f"🤖 **Deployment Tips:**\n{advice}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ MAINTENANCE ------------------------------
elif st.session_state.current_section == "maintenance":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>⚙️ Issue Resolution & Updates</h2>', unsafe_allow_html=True)
    issue = st.text_area("Report an issue:")
    if st.button("Submit Issue"):
        st.session_state.tasks.append(issue)
        st.success("Issue logged.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 SDLC Assistant | Built with ❤️ using Streamlit & Watsonx")

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
