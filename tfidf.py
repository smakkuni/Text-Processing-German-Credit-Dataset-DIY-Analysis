import math
import re
import os

def clean(text):
    # Remove website links
    text = re.sub(r'https?://\S+', '', text)
    # Remove non-word, non-whitespace characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text, stopwords):
    words = text.split()
    return ' '.join(w for w in words if w not in stopwords)

def stem(word):
    if word.endswith('ing') and len(word) > 3:
        return word[:-3]
    if word.endswith('ly') and len(word) > 2:
        return word[:-2]
    if word.endswith('ment') and len(word) > 4:
        return word[:-4]
    return word

def stemming(text):
    words = text.split()
    return ' '.join(stem(w) for w in words)

def preprocess(text, stopwords):
    text = clean(text)
    text = remove_stopwords(text, stopwords)
    text = stemming(text)
    return text

def compute_tf(words):
    total = len(words)
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return {w: freq[w] / total for w in freq}

def compute_idf(all_words_per_doc, total_docs):
    doc_freq = {}
    for words in all_words_per_doc:
        for w in set(words):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    return {w: math.log(total_docs / doc_freq[w]) + 1 for w in doc_freq}

def main():
    # Load document list
    with open('tfidf_docs.txt', 'r') as f:
        doc_files = [line.strip() for line in f if line.strip()]

    # Load stopwords
    with open('stopwords.txt', 'r') as f:
        stopwords = set(line.strip() for line in f if line.strip())

    # Preprocess each document
    preprocessed_texts = []
    for doc in doc_files:
        with open(doc, 'r') as f:
            text = f.read()
        processed = preprocess(text, stopwords)
        preprocessed_texts.append(processed)

        out_name = 'preproc_' + os.path.basename(doc)
        with open(out_name, 'w') as f:
            f.write(processed)

    # Compute TF-IDF
    all_words = [text.split() for text in preprocessed_texts]
    total_docs = len(doc_files)
    idf = compute_idf(all_words, total_docs)

    for i, doc in enumerate(doc_files):
        words = all_words[i]
        tf = compute_tf(words)
        tfidf = {}
        for w in tf:
            score = round(tf[w] * idf[w], 2)
            tfidf[w] = score

        # Sort: descending score, then alphabetical
        sorted_words = sorted(tfidf.items(), key=lambda x: (-x[1], x[0]))
        top5 = sorted_words[:5]

        out_name = 'tfidf_' + os.path.basename(doc)
        with open(out_name, 'w') as f:
            f.write(str(top5) + '\n')

if __name__ == '__main__':
    main()
