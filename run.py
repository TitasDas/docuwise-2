import openai
import streamlit as st
import os
from openai import OpenAI
from evaluation_metrics import calculate_faithfulness
from document_processor import get_index_for_pdf

st.title("Turn PDFs into an Instant, Smart Searchable Knowledge Base")

if not os.environ.get("OPENAI_API_KEY"):
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("Please enter your OpenAI API key to use the chatbot.")
        st.stop()
openai.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

@st.cache_resource
def create_vectordb(files, filenames):
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
        )
    return vectordb
pdf_files = st.file_uploader("Upload PDF file", type="pdf", accept_multiple_files=True)
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

prompt_template = """
You are an Assistant designed to provide concise answers based on various contexts provided.

Keep your responses brief and relevant.

Your evidence is drawn from the context of a PDF extract along with its metadata.

Pay particular attention to the 'filename' and 'page' metadata when formulating your responses.

Always include the filename and page number at the end of the sentence when referencing specific content.

Also at the end of the answer show all the chunks that were used to generate the answer sentences. 

If the text is not relevant to the question, reply with "Not applicable."

The PDF content is as follows:
{pdf_extract}
"""

prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
question = st.chat_input("Ask anything")
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        st.error("You need to provide a PDF")
        st.stop()
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "/n ".join([result.page_content for result in search_results])
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        botmsg = st.empty()
    response = []
    result = ""
    for chunk in client.chat.completions.create(
        model="gpt-4o", messages=prompt, stream=True
    ):
        text = chunk.choices[0].delta.content
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)
            
    faithfulness_score = calculate_faithfulness(result, pdf_extract)
    st.write(f"Faithfulness score: {faithfulness_score:.2f}")
    
    prompt.append({"role": "assistant", "content": result})
    st.session_state["prompt"] = prompt
    
    prompt.append({"role": "assistant", "content": result})
    st.session_state["prompt"] = prompt
