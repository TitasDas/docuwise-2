# DocuWise 2.0

> **DocuWise 2.0** is the evolution of [**DocuWise 1.0**](https://github.com/TitasDas/DocuWise) — originally an Android app that used **Apache OpenNLP** for linguistic parsing and question understanding.  
> The 2.0 version extends that foundation into a modern RAG framework enabling multi-document retrieval, semantic reasoning, and faithfulness evaluation.

Welcome to [**DocuWise 2.0**](https://docuwise.streamlit.app/) , an experimental **Retrieval-Augmented Generation (RAG)** application built to explore **security methodologies**, **evaluation metrics**, and **faithfulness scoring** within RAG pipelines.  

This project serves as both a prototype and a research sandbox for improving the reliability and interpretability of LLM-powered document question-answering systems.

---

## ⚠️ Important Disclaimer

> **Warning:**  
> This project is **not representative of a production-grade RAG system**.  
> It was intentionally built as a **lightweight prototype** for experimentation and learning not for deployment in live, user-facing environments.  
>  
> DocuWise 2.0 prioritizes **exploration over scalability**, **observability**, and **security hardening**.  
> A production-ready RAG system would require:
> - Persistent vector storage (e.g. Pinecone, Qdrant, Weaviate, or FAISS with proper indexing)  
> - Secure API key management and backend isolation  
> - Evaluation pipelines with logging, caching, and consistency checks  
> - Fine-tuned prompt templates and controlled retrieval workflows  
> - Data governance, privacy controls, and a well-defined user/data flow architecture — along with several other production-grade optimizations. 
>  
> Think of this app as a **research tool and experimental baseline**, not as a deployment blueprint.

---

## Getting Started

Follow these steps to begin extracting knowledge from your PDF documents:

### 1. Configure API Key
Enter your `OPENAI_API_KEY` in the app’s configuration field.  
This key enables the model to process text, generate responses, and perform semantic retrieval.  

> *Tip:* You can create or manage your API key at [OpenAI’s API Dashboard](https://platform.openai.com/account/api-keys).

### 2. Upload Documents
Use the **Browse Files** button to upload one or more PDF files.  
The system supports most standard PDF formats and automatically indexes document content for retrieval.

### 3. Ask Questions
Once your PDFs are uploaded, type a question in the input field.  
DocuWise 2.0 retrieves relevant sections, analyzes the context, and generates an answer grounded in your uploaded content.

---

## Key Features

- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with LLM reasoning for contextual answers.  
- **Faithfulness Scoring:** Evaluates how closely each answer aligns with the source material.  
- **Privacy-Conscious Design:** Uploaded documents are processed in real-time and not persistently stored.  
- **Security Evaluation Mode:** Built to test and iterate on security methodologies for RAG systems.  
- **Custom Evaluation Metrics:** Experiment with metrics beyond traditional accuracy and relevance.  

---

## Important Considerations

In some cases, the **faithfulness score** may appear as zero, even when the answer originates from the uploaded document.  
This occurs when the evaluation metric fails to detect sufficient lexical overlap a common challenge when paraphrasing or summarizing.

---

## Upcoming features

### Security of LLM based systems
- Inhibit prompt injection attacks by design - 

---

## Contribution & Feedback

DocuWise 2.0 is open for exploration, collaboration, and ideas. 
If you’d like to contribute whether it’s improving metrics, refining RAG workflows, or experimenting with new evaluation paradigms feel free to open an [issue](https://github.com/TitasDas/DocuWise2.0/issues) or start a discussion.

> Always happy to extend the boundaries of RAG systems every contribution helps make retrieval-based AI more secure, interpretable, and useful.

---
