import os
import uuid
import base64
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Import the cached compiled agent & stateful entrypoint
from agent import ask_medical_agent_stateful, get_compiled_medical_agent


@st.cache_resource
def get_cached_agent():
    """
    Caches the compiled LangGraph agent in Streamlit RAM resource memory.
    Prevents rebuilding the LLM, tools, and StateGraph on every user message.
    """
    return get_compiled_medical_agent()


def get_base64_image(image_path: str) -> str:
    """
    Encodes local image to base64 string for inline HTML rendering.
    """
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""


def main():
    st.set_page_config(
        page_title="MediBot - Agentic AI Medical Assistant", 
        page_icon="assets/bot_logo.png" if os.path.exists("assets/bot_logo.png") else "🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS Styling for Modern Medical Interface (White Title & Larger Robot Logo)
    st.markdown("""
    <style>
        .main-header-container {
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 0.5rem;
            padding-top: 0.5rem;
        }
        .main-title-white {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFFFFF !important;
            margin: 0;
            padding: 0;
            line-height: 1.2;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #CBD5E0;
            margin-bottom: 1.5rem;
        }
        .event-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .badge-rag { background-color: #EBF8FF; color: #2B6CB0; border: 1px solid #BEE3F8; }
        .badge-calc { background-color: #EDF2F7; color: #2D3748; border: 1px solid #E2E8F0; }
        .badge-symptom { background-color: #E6FFFA; color: #234E52; border: 1px solid #B2F5EA; }
        .badge-safety { background-color: #FEFCBF; color: #744210; border: 1px solid #FAF089; }
        .badge-emergency { background-color: #FED7D7; color: #9B2C2C; border: 1px solid #FEB2B2; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize Streamlit agent cache
    get_cached_agent()

    # Load Robot Logo Image
    logo_b64 = get_base64_image("assets/bot_logo.png")

    # Sidebar Information Panel
    with st.sidebar:
        if logo_b64:
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" width="130" style="border-radius: 12px; margin-bottom: 12px;">', unsafe_allow_html=True)
        else:
            st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=90)
            
        st.markdown("### MediBot Agentic System")
        st.caption("Production Multi-Tool LangGraph Agent with RAG, Safety Triage & Evidence Validation.")
        
        st.divider()
        st.markdown("**Core Capabilities:**")
        st.markdown("• 🔍 **RAG Knowledge Search**: Indexed handbooks (304 pages, 582 chunks)")
        st.markdown("• 🚨 **Safety Triage**: Multi-category red-flag detection")
        st.markdown("• 📋 **Symptom Guide**: 15+ structured clinical guides")
        st.markdown("• 📊 **Medical Calculator**: Deterministic BMI, BMR, IBW")
        
        st.divider()
        st.markdown("**LLM Engine:** `llama-3.3-70b-versatile` (Groq API)")
        st.markdown("**State Checkpointer:** `MemorySaver`")
        
        if st.button("🗑️ Reset Chat Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
            st.rerun()

    # Main Header with Larger Inline Robot Logo Image & White Title
    if logo_b64:
        st.markdown(f"""
        <div class="main-header-container">
            <img src="data:image/png;base64,{logo_b64}" width="110" style="border-radius: 12px; vertical-align: middle;">
            <h1 class="main-title-white">MediBot - Agentic AI Medical Assistant</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-title-white">🏥 MediBot - Agentic AI Medical Assistant</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="sub-title">Powered by LangGraph Production StateGraph with Safety Triage, Evidence Validator & Single-Pass RAG Reuse.</div>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Persistent thread ID per session for stateful memory
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message.get('safety_status') == "EMERGENCY_RED_FLAG":
                st.error("🚨 **CRITICAL SAFETY ALERT**: Emergency Red-Flag Symptoms Intercepted. Immediate Emergency Care Recommended.")
                
            if message.get('events'):
                event_html = "".join([f'<span class="event-badge {e["class"]}">{e["text"]}</span>' for e in message['events']])
                st.markdown(event_html, unsafe_allow_html=True)
                
            st.markdown(message['content'])
            
            if message.get('retrieved_documents'):
                with st.expander("📚 View Grounded Source References (Enriched Metadata)"):
                    for i, doc in enumerate(message['retrieved_documents'], 1):
                        page = doc.get('page', 'N/A')
                        source = doc.get('source', 'Medical Reference')
                        topic = doc.get('topic', 'Clinical Reference')
                        section = doc.get('section', 'General Section')
                        content = doc.get('content', '')
                        
                        st.markdown(f"**Reference {i}** | `{source}` (Page {page})")
                        st.caption(f"📌 **Topic**: {topic} | 🔖 **Section**: {section}")
                        st.info(content)

    prompt = st.chat_input("Ask a medical question, follow-up, calculate BMI/BMR, or describe symptoms...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Invoke Stateful LangGraph Agent using persistent thread_id
            agent_response = ask_medical_agent_stateful(user_query=prompt, thread_id=st.session_state.thread_id)
            result = agent_response["answer"]
            tool_calls = agent_response["tool_calls"]
            retrieved_documents = agent_response.get("retrieved_documents", [])
            safety_status = agent_response.get("safety_status", "SAFE_ROUTINE")

            # Format Concise Execution Event Badges (No CoT Exposure)
            events = []
            if safety_status == "EMERGENCY_RED_FLAG":
                events.append({"text": "🚨 Emergency Triage", "class": "badge-emergency"})
                
            if tool_calls:
                for call in tool_calls:
                    t_name = call.get("name", "")
                    if t_name == "medical_calculator":
                        events.append({"text": "⚡ Executed: Medical Calculator", "class": "badge-calc"})
                    elif t_name == "symptom_information":
                        events.append({"text": "⚡ Executed: Symptom Guide", "class": "badge-symptom"})
                    elif t_name == "medical_knowledge_search":
                        events.append({"text": "⚡ Executed: RAG Knowledge Search", "class": "badge-rag"})

            with st.chat_message('assistant'):
                # Render Prominent Safety Banner if Emergency
                if safety_status == "EMERGENCY_RED_FLAG":
                    st.error("🚨 **CRITICAL SAFETY ALERT**: Emergency Red-Flag Symptoms Intercepted. Immediate Emergency Care Recommended.")

                # Render Concise Action Badges
                if events:
                    event_html = "".join([f'<span class="event-badge {e["class"]}">{e["text"]}</span>' for e in events])
                    st.markdown(event_html, unsafe_allow_html=True)

                st.markdown(result)
                
                # Expandable Evidence Cards
                if retrieved_documents:
                    with st.expander("📚 View Grounded Source References (Enriched Metadata)"):
                        for i, doc in enumerate(retrieved_documents, 1):
                            page = doc.get('page', 'N/A')
                            source = doc.get('source', 'Medical Reference')
                            topic = doc.get('topic', 'Clinical Reference')
                            section = doc.get('section', 'General Section')
                            content = doc.get('content', '')
                            
                            st.markdown(f"**Reference {i}** | `{source}` (Page {page})")
                            st.caption(f"📌 **Topic**: {topic} | 🔖 **Section**: {section}")
                            st.info(content)

            st.session_state.messages.append({
                'role': 'assistant',
                'content': result,
                'events': events,
                'safety_status': safety_status,
                'retrieved_documents': retrieved_documents
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()