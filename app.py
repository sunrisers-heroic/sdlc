import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Page config
st.set_page_config(page_title="SDLC Assistant", layout="wide", page_icon="🛠️")

# Custom CSS for animated UI and green/blue theme
st.markdown("""
    <style>
        body {
            background-color: #f0fff4;
            font-family: Arial, sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            transition: all 0.3s ease-in-out;
        }
        .card {
            background-color: #ffffff;
            padding: 15px 20px;
            border-left: 5px solid #2ecc71;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        .section-title {
            color: #2ecc71;
        }
        .chat-bubble-user {
            background-color: #d6eaff;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            align-self: flex-end;
            margin: 5px 0;
        }
        .chat-bubble-bot {
            background-color: #e6f0ff;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            align-self: flex-start;
            margin: 5px 0;
        }
        .navbar {
            display: flex;
            justify-content: center;
            gap: 15px;
            padding: 10px 0;
            background: linear-gradient(to right, #2ecc71, #27ae60);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .nav-button {
            background-color: #ffffff;
            color: #2ecc71;
            border: none;
            padding: 10px;
            font-size: 16px;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .nav-button:hover {
            background-color: #eafaf1;
        }
        .fade-enter {
            opacity: 0;
            transform: translateY(10px);
        }
        .fade-enter-active {
            opacity: 1;
            transform: translateY(0);
            transition: all 0.3s ease;
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
if "documents" not in st.session_state:
    st.session_state.documents = []

# Load Watsonx credentials from secrets
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]

    model_map = {
        "symptoms": "ibm/granite-3-2-8b-instruct",
        "chat": "ibm/granite-3-2-3b-instruct",
        "documentation": "ibm/granite-3-2-13b-instruct",
        "code": "ibm/granite-3-2-code-instruct"
    }

    def get_llm(model_name):
        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.TEMPERATURE: 0,
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.MAX_NEW_TOKENS: 250,
            GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
        }
        return WatsonxLLM(
            model_id=model_map[model_name],
            url=credentials.get("url"),
            apikey=credentials.get("apikey"),
            project_id=project_id,
            params=params
        )

except KeyError:
    st.warning("⚠️ Watsonx credentials missing.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

# Top Navigation Buttons
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    if st.button("🏠", key="btn_home"):
        st.session_state.current_section = "home"
with col2:
    if st.button("🔐", key="btn_login"):
        st.session_state.current_section = "login"
with col3:
    if st.button("🧾", key="btn_profile"):
        st.session_state.current_section = "profile"
with col4:
    if st.button("🧠", key="btn_symptoms"):
        st.session_state.current_section = "requirements"
with col5:
    if st.button("🤖", key="btn_chat"):
        st.session_state.current_section = "chat"
with col6:
    if st.button("🪄", key="btn_code"):
        st.session_state.current_section = "development"
with col7:
    if st.button("⚙️", key="btn_settings"):
        st.session_state.current_section = "settings"
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align:center; color:#2ecc71;">🛠️ SDLC Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px;">Manage software development lifecycle phases with AI.</p>', unsafe_allow_html=True)

# Function to show/hide sections with animation
def render_section(title, content):
    st.markdown(f'<div class="card fade-enter-active">{title}</div>', unsafe_allow_html=True)
    st.markdown(content, unsafe_allow_html=True)

# ------------------------------ HOME PAGE ------------------------------
if st.session_state.current_section == "home":
    render_section(
        "<h2>🛠️ Welcome to Your SDLC Assistant</h2>",
        """
        This application helps you manage software development projects comprehensively — from requirements to deployment.
        ### 🧠 Highlights:
        - 📝 Requirement gathering & prioritization  
        - 🧩 Task management (Kanban, Scrum)  
        - 🔍 Test case creation & tracking  
        - 🚀 Deployment automation  
        - 🤖 AI Code generation  
        Get started by exploring any of the tools above!
        """
    )

# ------------------------------ LOGIN PAGE ------------------------------
elif st.session_state.current_section == "login":
    render_section("<h2>🔐 Login</h2>", """
        <form>
            <label>Username:</label><br>
            <input type="text" placeholder="Enter username"><br><br>
            <label>Password:</label><br>
            <input type="password" placeholder="Enter password"><br><br>
            <button>Login</button>
        </form>
    """)

# ------------------------------ USER PROFILE ------------------------------
elif st.session_state.current_section == "profile":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🧾 User Profile & Dashboard</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        role = st.selectbox("Role", ["Developer", "Tester", "Project Manager", "Designer"])
    with col2:
        team = st.text_input("Team / Department")
        experience = st.number_input("Years of Experience", min_value=0)
    if st.button("Save Profile"):
        st.session_state.profile = {"name": name, "role": role, "team": team, "experience": experience}
        st.success("Profile saved!")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ REQUIREMENTS GATHERING ------------------------------
elif st.session_state.current_section == "requirements":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>📝 Requirements Gathering</h2>', unsafe_allow_html=True)
    requirement = st.text_area("Enter a new requirement:")
    if st.button("Add Requirement"):
        st.session_state.requirements.append(requirement)
        st.success("Requirement added!")
    st.markdown("### Prioritization Tools")
    priority = st.radio("Prioritize using:", ["MoSCoW", "Kano Model"])
    if st.button("Analyze Requirements"):
        llm = get_llm("documentation")
        prompt = f"Use {priority} to prioritize these requirements:\n{requirement}"
        response = llm.invoke(prompt)
        st.markdown(f"🔍 **AI Analysis:**\n{response}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🤖 AI Chatbot</h2>', unsafe_allow_html=True)
    user_input = st.text_input("Ask anything about SDLC...")
    if st.button("Send") and user_input:
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Thinking..."):
            try:
                llm = get_llm("chat")
                ai_response = llm.invoke(user_input)
                st.session_state.messages.append(("assistant", ai_response))
            except Exception as e:
                st.session_state.messages.append(("assistant", f"Error: {str(e)}"))
    # Display chat history
    for role, msg in st.session_state.messages:
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
        st.markdown(f'<div class="{bubble_class}"><b>{role}:</b> {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ CODE GENERATION ------------------------------
elif st.session_state.current_section == "development":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>🪄 AI Code Generator</h2>', unsafe_allow_html=True)
    prompt = st.text_area("Describe what code you need:")
    if st.button("Generate Code"):
        with st.spinner("Generating..."):
            llm = get_llm("code")
            code = llm.invoke(prompt)
            st.code(code)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ SETTINGS ------------------------------
elif st.session_state.current_section == "settings":
    st.markdown('<div class="card fade-enter-active">', unsafe_allow_html=True)
    st.markdown('<h2>⚙️ Settings & Preferences</h2>', unsafe_allow_html=True)
    language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
    theme = st.selectbox("Theme", ["Light", "Dark"])
    font_size = st.slider("Font Size", 12, 24)
    if st.button("Save Preferences"):
        st.success("Preferences updated!")
        st.markdown(f"🤖 **AI Tip:** A good font size for readability is usually between 14-16px.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 SDLC Assistant | Built with ❤️ using Streamlit & Watsonx")

# Debug Mode
with st.expander("🔧 Debug Mode"):
    st.write("Session State:", st.session_state)
