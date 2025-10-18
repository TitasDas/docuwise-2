# Custom faithfulness score
def calculate_faithfulness(answer, source_documents):
    source_sentences = [sent.strip() for doc in source_documents for sent in doc.split('.')]
    answer_sentences = answer.split('.')

    faithful_score = sum(any(ans_sent in src_sent for src_sent in source_sentences)
                         for ans_sent in answer_sentences) / len(answer_sentences)

    return faithful_score
