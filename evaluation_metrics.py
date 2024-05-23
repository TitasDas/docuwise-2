from typing import List
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def calculate_faithfulness(answer: str, source_documents: List[str]) -> float:
    source_sentences = [sent.strip() for doc in source_documents for sent in doc.split('.')]
    answer_sentences = [sent.strip() for sent in answer.split('.')]
    
    faithful_count = sum(any(ans_sent in src_sent for src_sent in source_sentences) for ans_sent in answer_sentences)
    faithful_score = faithful_count / max(len(answer_sentences), 1)  # Avoid division by zero

    return faithful_score


def calculate_context_relevancy(question: str, answer: str, source_documents: List[str]) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embeddings for the question, answer, and source documents
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_documents, convert_to_tensor=True)

    # Relevancy scores
    question_relevancy = util.pytorch_cos_sim(question_embedding, answer_embedding).item()
    context_relevancy = util.pytorch_cos_sim(answer_embedding, source_embeddings).mean().item()

    relevancy_score = (question_relevancy + context_relevancy) / 2

    return relevancy_score


def calculate_information_coverage(answer: str, source_documents: List[str]) -> float:
    # Combine source documents into a single text
    source_text = ' '.join(source_documents)

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([source_text, answer])
    dense = vectors.todense()
    denselist = dense.tolist()
    source_vector = denselist[0]
    answer_vector = denselist[1]

    # Coverage score (cosine similarity)
    coverage_score = np.dot(source_vector, answer_vector) / (np.linalg.norm(source_vector) * np.linalg.norm(answer_vector))

    return coverage_score
