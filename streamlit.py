import streamlit as st
from main import run_agent  # uses your existing RAG pipeline

st.set_page_config(page_title="RAG but Caffeinated ☕", page_icon="☕")

st.title("Quantum-caffeinated chaos")
st.write(
    "Ask a question. "
    "The agent will retrieve from your local documents and generate an answer."
)

# Text input
question = st.text_input("Your question:", placeholder="What on your mind ?")

if "last_state" not in st.session_state:
    st.session_state.last_state = None

# Button to run the agent
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            state = run_agent(question)
            st.session_state.last_state = state

# Display results if available
if st.session_state.last_state is not None:
    state = st.session_state.last_state

    st.subheader("🧠 Answer")
    st.write(state.get("answer", "(no answer)"))

    eval_result = state.get("evaluation", {})
    ok = eval_result.get("ok")
    feedback = eval_result.get("feedback", "")

    st.subheader("✅ Reflection")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**OK:** `{ok}`")
    with col2:
        st.markdown(f"**Feedback:** {feedback}")
