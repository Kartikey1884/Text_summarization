import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document

from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader
import re

# ---------------- Streamlit setup ----------------
st.set_page_config(
    page_title="LangChain: Summarize YouTube or Website",
    layout="wide",
    page_icon="📜"
)
st.title("LangChain: Summarize YouTube or Website")
st.subheader("Summarize URL")

# ---------------- Sidebar: Groq API key ----------------
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API key", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# ---------------- Initialize LLM only if key is provided ----------------
llm = None
if groq_api_key.strip():
    try:
        llm = ChatGroq(api_key=groq_api_key.strip(), model="llama-3.3-70b-versatile")
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}")

# ---------------- Prompt template ----------------
prompt_template = """
Provide a summary of the following content in 300 words:
Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ---------------- Summarize button ----------------
if st.button("Summarize the content from YouTube or Website"):

    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Missing API key or URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please provide a valid URL (YouTube or website).")
        st.stop()

    if llm is None:
        st.error("LLM not initialized. Check your API key.")
        st.stop()

    try:
        with st.spinner("Fetching content..."):

            # Load content from YouTube or website
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
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
                st.error("No content found at the URL.")
                st.stop()

            # Trim content to avoid HTTP 400
            max_len = 2000
            text_to_summarize = docs[0].page_content[:max_len]

            # Remove unsupported control characters
            text_to_summarize = re.sub(r"[\x00-\x1f]+", " ", text_to_summarize)

            # Create new Document with trimmed content
            trimmed_doc = Document(page_content=text_to_summarize, metadata=docs[0].metadata)

            st.subheader("Preview of content to summarize (first 500 chars)")
            st.code(text_to_summarize[:300])

            # Run summarization chain
            with st.spinner("Generating summary..."):
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run([trimmed_doc])

            st.success("Summary generated:")
            st.write(output_summary)

    except Exception as e:
        st.exception(f"Error occurred: {e}")
