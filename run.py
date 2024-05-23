import openai
import streamlit as st
import os
from openai import OpenAI
from evaluation_metrics import calculate_faithfulness, calculate_context_relevancy, calculate_information_coverage
from document_processor import get_index_for_pdf


# Set the title for the Streamlit app
st.title("Turn PDFs into an Instant, Smart Searchable Knowledge Base")

# Prompt for OpenAI API key if not already set in environment variables
if not os.environ.get("OPENAI_API_KEY"):
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("Please enter your OpenAI API key to use the chatbot.")
        st.stop()

openai.api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# Cached function to create a vectordb for the provided PDF files
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai.api_key
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
    # search_results
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

    # Call ChatGPT with streaming and display the response as it comes
    response = []
    result = ""
    for chunk in client.chat.completions.create(
        model="gpt-4-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].delta.content
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    faithfulness_score = calculate_faithfulness(result, [doc.page_content for doc in search_results])
    context_relevancy_score = calculate_context_relevancy(question, result, [doc.page_content for doc in search_results])
    information_coverage_score = calculate_information_coverage(result, [doc.page_content for doc in search_results])

    st.write(f"Faithfulness score: {faithfulness_score:.2f}")
    st.write(f"Context relevancy score: {context_relevancy_score:.2f}")
    st.write(f"Information coverage score: {information_coverage_score:.2f}")

    # Add the assistant's response to the prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
    prompt.append({"role": "assistant", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
