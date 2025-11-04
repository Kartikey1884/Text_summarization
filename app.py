import re
import validators
import streamlit as st

# ✅ Modern LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader

# ---------------- Streamlit setup ----------------
st.set_page_config(
    page_title="LangChain: Summarize YouTube or Website",
    layout="wide",
    page_icon="📜"
)
st.title("📜 LangChain: Summarize YouTube or Website")
st.subheader("Summarize any YouTube video or website content")

# ---------------- Sidebar: Groq API key ----------------
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API key", type="password")
    st.markdown(
        "[Get your API key here](https://console.groq.com) if you don’t have one."
    )

# ---------------- Main input ----------------
generic_url = st.text_input("Paste a YouTube or website URL", label_visibility="collapsed")

# ---------------- Initialize LLM ----------------
llm = None
if groq_api_key.strip():
    try:
        llm = ChatGroq(
            api_key=groq_api_key.strip(),
            model="llama-3.3-70b-versatile"
        )
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")

# ---------------- Prompt Template ----------------
prompt_template = """You are an expert summarizer.
Summarize the following content in around 300 words. Focus on key points and clarity.

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ---------------- Summarize Button ----------------
if st.button("🚀 Summarize the content"):

    if not groq_api_key.strip():
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    if not generic_url.strip():
        st.error("Please enter a valid URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("The input is not a valid URL.")
        st.stop()

    if llm is None:
        st.error("LLM initialization failed. Please check your API key.")
        st.stop()

    try:
        with st.spinner("🔍 Fetching content..."):

            # Decide loader type
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                loader = YoutubeLoader.from_youtube_url(
                    generic_url,
                    add_video_info=True,
                    language=["en"]
                )
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/58.0.3029.110 Safari/537.3"
                        )
                    }
                )

            docs = loader.load()

            if not docs:
                st.error("No readable content found at the URL.")
                st.stop()

            # Trim and clean content
            text_to_summarize = docs[0].page_content[:1500]
            text_to_summarize = re.sub(r"[\x00-\x1f]+", " ", text_to_summarize)
            text_to_summarize = re.sub(r"\s+", " ", text_to_summarize)

            # Create Document object (optional, just for structure)
            trimmed_doc = Document(page_content=text_to_summarize, metadata=docs[0].metadata)

            # Preview content
            st.subheader("🔎 Preview of extracted content (first 300 characters):")
            st.code(text_to_summarize[:300])

        # ---------------- Run summarization ----------------
        with st.spinner("🧠 Generating summary using Groq LLM..."):
            summary_prompt = prompt.format(text=text_to_summarize)
            response = llm.invoke(summary_prompt)
            summary = response.content if hasattr(response, "content") else response

        st.success("✅ Summary generated successfully:")
        st.write(summary)

    except Exception as e:
        st.exception(f"Error occurred: {e}")
