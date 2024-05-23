import streamlit as st
import os
from evaluation_metrics import calculate_faithfulness
from document_processor import get_index_for_pdf
from transformers import pipeline

# Set the title for the Streamlit app
st.title("Turn PDFs into an Instant, Smart Searchable Knowledge Base")

# Set up the Hugging Face pipeline for question answering
question_answering_pipeline = pipeline("question-answering", model="EleutherAI/gpt-neo-2.7B")

# Cached function to create a vectordb for the provided PDF files
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames
        )
    return vectordb

# Upload PDF files using Streamlit's file uploader
pdf_files = st.file_uploader("Upload PDF file", type="pdf", accept_multiple_files=True)

# If PDF files are uploaded, create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    st.session_state["vectordb"] = create_vectordb(pdf_files, pdf_file_names)

# Define the template for the chatbot prompt
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

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        st.error("You need to provide a PDF")
        st.stop()

    # Search the vectordb for similar content to the user's question
    search_results = vectordb.similarity_search(question, k=3)
    pdf_extract = "/n ".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract
    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Use the Hugging Face pipeline for question answering
    context = pdf_extract
    response = question_answering_pipeline(question=question, context=context)
    result = response['answer']

    # Display the response
    botmsg.write(result)

    faithfulness_score = calculate_faithfulness(result, [doc.page_content for doc in search_results])
    st.write(f"Faithfulness score: {faithfulness_score:.2f}")

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
