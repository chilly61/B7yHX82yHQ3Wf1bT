import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api
import re
from sentence_transformers import SentenceTransformer

df = pd.read_csv("ProjectC.csv")
role_keywords = [
    "full-stack software engineer",
    "engineering manager",
    "aspiring human resources"
]

# This is the second run with the starring marks.
# df = pd.read_csv("ProjectC_mean.csv")
# df['starring'] = pd.to_numeric(df['starring'], errors='coerce')
# star_job_titles = df.loc[df['starring'] == 1, 'job_title'].tolist()
# role_keywords = [
#     "full-stack software engineer",
#     "engineering manager",
#     "aspiring human resources"
# ]
# role_keywords += star_job_titles

# =========================
# TF-IDF pipeline
# =========================
# Step 1: Combine job titles and keywords for TF-IDF vectorization
corpus = df['job_title'].tolist() + role_keywords

# Step 2: TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit(corpus)
job_vectors = vectorizer.transform(df['job_title'])
keyword_vectors = vectorizer.transform(role_keywords)
tfidf_vocab = vectorizer.vocabulary_
idf = vectorizer.idf_
idf_dict = {
    word: idf[idx]
    for word, idx in tfidf_vocab.items()
}

# Step 3: Compute cosine similarity between each job_title and all keywords
similarity_matrix = cosine_similarity(job_vectors, keyword_vectors)

# Step 4: For each candidate, take the max similarity as fit_score
# df['fit_score_raw'] = similarity_matrix.max(axis=1) #similarity option, max similarity
df['fit_score_raw'] = similarity_matrix.mean(axis=1)
# Step 5: Normalize fit_score to 0-1
df['fit_score'] = (df['fit_score_raw'] - df['fit_score_raw'].min()) / \
                  (df['fit_score_raw'].max() - df['fit_score_raw'].min())

# Optional: show which keyword contributes most
most_relevant_keyword_idx = similarity_matrix.argmax(axis=1)
df['top_keyword'] = [role_keywords[i] for i in most_relevant_keyword_idx]

print(df[['id', 'job_title', 'fit_score', 'top_keyword']])

# 按 fit_score 降序排序
df_sorted = df.sort_values(by='fit_score', ascending=False)

df_sorted[['id', 'job_title', 'fit_score', 'fit_score_raw', 'top_keyword',
           'location', 'connection']].to_csv("ProjectC_new_mean.csv", index=False, float_format="%.3f")


# =========================
# Word2Vec pipeline
# =========================
# Load Word2Vec
w2v = api.load("word2vec-google-news-300")

# Text → Word2Vec vector


def text_to_vector(text, model, idf_dict, dim=300):
    words = re.findall(r"[a-zA-Z]+", text.lower())

    weighted_vectors = []
    weights = []

    for w in words:
        if w in model and w in idf_dict:
            weight = idf_dict[w]
            weighted_vectors.append(weight * model[w])
            weights.append(weight)

    if not weighted_vectors:
        return np.zeros(dim)

    return np.sum(weighted_vectors, axis=0) / np.sum(weights)

# Vectorize job titles and keywords


job_vectors = np.vstack([
    text_to_vector(title, w2v, idf_dict)
    for title in df['job_title']
])

keyword_vectors = np.vstack([
    text_to_vector(k, w2v, idf_dict)
    for k in role_keywords
])

# Cosine similarity
similarity_matrix = cosine_similarity(job_vectors, keyword_vectors)

# mean similarity
df['w2v_fit_raw'] = similarity_matrix.mean(axis=1)

# standardization 0–1
df['w2v_fit'] = (
    (df['w2v_fit_raw'] - df['w2v_fit_raw'].min()) /
    (df['w2v_fit_raw'].max() - df['w2v_fit_raw'].min())
)

# the most similar keyword
top_idx = similarity_matrix.argmax(axis=1)
df['w2v_top_keyword'] = [role_keywords[i] for i in top_idx]

# Sort & export
df_sorted = df.sort_values(by='w2v_fit', ascending=False)

df_sorted.to_csv(
    "ProjectC_word2vec_only.csv",
    index=False,
    float_format="%.3f"
)

print(df_sorted[
    ['id', 'job_title', 'w2v_fit', 'w2v_top_keyword']
].head(10))


# =========================
# Glove pipeline
# =========================

glove = api.load("glove-wiki-gigaword-300")


job_vectors = np.vstack([
    text_to_vector(title, glove, idf_dict)
    for title in df['job_title']
])

keyword_vectors = np.vstack([
    text_to_vector(k, glove, idf_dict)
    for k in role_keywords
])


# Cosine similarity
similarity_matrix = cosine_similarity(job_vectors, keyword_vectors)

# mean similarity
df['glove_fit_raw'] = similarity_matrix.mean(axis=1)

# standardization 0–1
df['glove_fit'] = (
    (df['glove_fit_raw'] - df['glove_fit_raw'].min()) /
    (df['glove_fit_raw'].max() - df['glove_fit_raw'].min())
)

# the most similar keyword
top_idx = similarity_matrix.argmax(axis=1)
df['glove_top_keyword'] = [role_keywords[i] for i in top_idx]


# Sort & export
df_sorted = df.sort_values(by='glove_fit', ascending=False)

df_sorted.to_csv(
    "ProjectC_glove_only.csv",
    index=False,
    float_format="%.3f"
)

print(df_sorted[
    ['id', 'job_title', 'glove_fit', 'glove_top_keyword']
].head(10))


# =========================
# Load fastText model
# =========================

ft = api.load("fasttext-wiki-news-subwords-300")


job_vectors = np.vstack([
    text_to_vector(title, ft, idf_dict)
    for title in df['job_title']
])

keyword_vectors = np.vstack([
    text_to_vector(k, ft, idf_dict)
    for k in role_keywords
])

similarity_matrix = cosine_similarity(job_vectors, keyword_vectors)

df['fasttext_fit_raw'] = similarity_matrix.mean(axis=1)

df['fasttext_fit'] = (
    (df['fasttext_fit_raw'] - df['fasttext_fit_raw'].min()) /
    (df['fasttext_fit_raw'].max() - df['fasttext_fit_raw'].min())
)

top_idx = similarity_matrix.argmax(axis=1)
df['fasttext_top_keyword'] = [role_keywords[i] for i in top_idx]

df_sorted = df.sort_values(by='fasttext_fit', ascending=False)

df_sorted.to_csv(
    "ProjectC_fasttext_only.csv",
    index=False,
    float_format="%.3f"
)

print(df_sorted[
    ['id', 'job_title', 'fasttext_fit', 'fasttext_top_keyword']
].head(10))
# My understanding:
# Location, Connection
# LDA Topic Model, Word2Vec


# Recommend: Do it!
# 1. Word2Vec Model(google research), TF-IDF
# 2. Explore Glove Model? (from standford)
# 3. Fast text? (facebook ai research)
# 聚合

# Next Next:
#     contextualized embeddings.


# =========================
# Load Sentence-BERT
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")
# 可替换为：
# model = SentenceTransformer("all-mpnet-base-v2")

# =========================
# Encode texts
# =========================
job_titles = df['job_title'].tolist() + role_keywords
job_embeddings = model.encode(
    job_titles,
    convert_to_numpy=True,
    normalize_embeddings=True
)

keyword_embeddings = model.encode(
    role_keywords,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# Cosine similarity
# =========================
similarity_matrix = cosine_similarity(job_embeddings, keyword_embeddings)

# mean similarity（与你现在的 mean 逻辑一致）
df['sbert_fit_raw'] = similarity_matrix.mean(axis=1)

# 0–1 normalization
df['sbert_fit'] = (
    (df['sbert_fit_raw'] - df['sbert_fit_raw'].min()) /
    (df['sbert_fit_raw'].max() - df['sbert_fit_raw'].min())
)

# Most similar keyword
top_idx = similarity_matrix.argmax(axis=1)
df['sbert_top_keyword'] = [role_keywords[i] for i in top_idx]

# =========================
# Sort & export
# =========================
df_sorted = df.sort_values(by='sbert_fit', ascending=False)

df_sorted.to_csv(
    "ProjectC_sbert_only.csv",
    index=False,
    float_format="%.3f"
)

print(df_sorted[
    ['id', 'job_title', 'sbert_fit', 'sbert_top_keyword']
].head(10))

# textual data, exposed to the technics (already) TF-IDF, Statical Embedding (change NLP), Chatgpt(Transform embedding) - 2017-google
# LSTM (step by step), Transfomer (attention mechanism) - more compute, revolutional faster

# Moving forward:
# LLM models: explore
# learn LLMs by the six steps (fundamentals), and build the LLM to do the ranking.
# in all projects (probably) - LLMs.