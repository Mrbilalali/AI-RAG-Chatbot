import os
import streamlit as st
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Emporio Solution Support", page_icon="💬", layout="centered")
st.title("💬 Emporio Support Assistant")

# Sidebar for API Key and PDF Upload 
with st.sidebar:
    st.header("Configuration")    
    api_key_input = st.text_input("Enter your OpenAI API Key:", type="password", key="api_key_input")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf", key="pdf_uploader")

# Session State Initialization
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = ""

# Update API key from input
if api_key_input:
    st.session_state.openai_api_key = api_key_input

# Main Application Logic 
if not st.session_state.openai_api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

if uploaded_file:
    # Process the file only if its new
    if uploaded_file.name != st.session_state.processed_file_name:
        with st.spinner("Processing your document... This may take a moment."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    tmp_path = tmpfile.name

                # Load and split the document
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                split_docs = text_splitter.split_documents(docs)

                # Create embeddings and vector store
                embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
                vectorstore = Chroma.from_documents(split_docs, embeddings)

                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file_name = uploaded_file.name
                
                # Clean up temp file
                os.remove(tmp_path)
                st.success("Document processed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
                st.stop()
else:
    st.info("Please upload a PDF document to start the chat.")
    st.stop()

# Proceed only if vectorstore is ready
if st.session_state.vectorstore is None:
    st.warning("Vector store is not initialized. Please ensure the document was processed correctly.")
    st.stop()

# LLM and Memory Setup
llm = ChatOpenAI(openai_api_key=st.session_state.openai_api_key, model="gpt-4o-mini")

def build_memory():
    """Recreate conversation memory from chat history."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    for role, msg in st.session_state.chat_history:
        if role == "user":
            memory.chat_memory.add_user_message(msg)
        elif role == "assistant":
            memory.chat_memory.add_ai_message(msg)
    return memory

prompt = PromptTemplate(
    template="""
You are Emporio Solutions intelligent customer support assistant.

You can use two sources to answer:
1️⃣ Chat history — for conversational continuity  
2️⃣ Context — for factual answers from the uploaded document  

Rules:
- Never repeat greetings like "Hello!" or "Welcome" after the first turn.
- If the user asks about Emporio's services, summarize them politely using the provided context.
- If the user asks about **price**, **cost**, **charges**, **budget**, **quotation**, or anything related to money (e.g., "how much", "give me a price", "tell me the cost"), respond with exactly this:
  "Our pricing depends on your project requirements. For a detailed quotation, please email us at support@emporiosolution.com — our sales team will assist you promptly."
- Do not ask again for project details if the user has already said things like “decide yourself,” “it’s your choice,” or “just give me a price.” In those cases, directly refer to the email.
- If no relevant info is found in the context, use your general business and software knowledge to provide a professional answer.
- Always keep your tone friendly, confident, and business-professional.
- Be conversational, but concise and precise — avoid overexplaining unless asked for details.

Chat history:
{chat_history}

Relevant document context:
{context}

User question: {question}

Now craft your answer below:
""",
    input_variables=["chat_history", "context", "question"]
)

# Chat Interface-
user_query = st.chat_input("Ask your question about the document...")

if user_query:
    with st.spinner("Thinking..."):
        # Build memory for the current turn
        memory = build_memory()

        # Search for relevant context
        docs_with_scores = st.session_state.vectorstore.similarity_search_with_score(user_query, k=3)
        filtered_docs = [doc for doc, score in docs_with_scores if score < 0.4] # Adjusted threshold for better relevance
        context = "\n\n".join([doc.page_content for doc in filtered_docs])

        # Fallback if context is empty
        if not context:
            context = "No relevant context was found in the document for your query."

        # Run the LLM chain
        qa_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, output_key="answer")
        result = qa_chain.invoke({"question": user_query, "context": context})
        answer = result["answer"]

        # Update chat history
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", answer))

# Chat Display
for role, text in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"**You:** {text}")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(text)

