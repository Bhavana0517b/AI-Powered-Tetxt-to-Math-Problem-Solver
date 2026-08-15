import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Math Problem Solver",
    page_icon="🧮"
)

st.title("🧮 Text To Math Problem Solver Using Llama-3")
st.write("Solve mathematical and logical reasoning problems with detailed explanations.")

# ------------------------------
# API Key
# ------------------------------
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

if not groq_api_key:
    st.info("Please enter your Groq API Key.")
    st.stop()

# ------------------------------
# LLM
# ------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.2
)

# ------------------------------
# Prompt
# ------------------------------
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert mathematics teacher.

Your task is to solve the user's problem exactly as a human tutor would.

Instructions:

- Read the complete question carefully.
- Identify all the given information.
- Explain your thinking clearly.
- Show EVERY mathematical step.
- Never skip calculations.
- Explain why each step is performed.
- Use numbered steps.
- If arithmetic is involved, show the calculations.
- If logical reasoning is involved, explain the logic.
- At the end write:

Final Answer:

inside a separate section.

Question:
{question}
"""
)

# ------------------------------
# Chain
# ------------------------------
chain = prompt | llm | StrOutputParser()

# ------------------------------
# Chat History
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------
# User Input
# ------------------------------
question = st.text_area(
    "Enter your question",
    height=150,
    value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack contains 25 blueberries. How many total pieces of fruit do I have at the end?"
)

# ------------------------------
# Generate Answer
# ------------------------------
if st.button("Solve Problem"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        response = chain.invoke(
            {
                "question": question
            }
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)