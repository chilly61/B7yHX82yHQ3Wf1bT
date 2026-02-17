# TalentFit: AI-Powered Candidate Ranking and Matching System

## Project Overview

TalentFit is an intelligent talent matching system based on Natural Language Processing (NLP), designed to help recruiters find candidates who best match job requirements more efficiently. Imagine the traditional recruitment process: recruiters need to sift through hundreds of resumes one by one, manually comparing each candidate's qualifications against the job requirements—not only is this time-consuming and exhausting, but excellent candidates can easily be missed due to human fatigue.

TalentFit exists to solve this pain point. The system can automatically analyze candidate job descriptions, calculate match scores based on predefined job keywords, and present candidates sorted from highest to lowest match. Even more importantly, when a recruiter marks a candidate as an ideal fit (by clicking the star icon), the system dynamically adjusts its understanding and continuously optimizes subsequent search results—this is a truly "learning" intelligent system.

This project follows a progressive development strategy, starting from traditional TF-IDF baseline matching methods, gradually introducing word embedding techniques (Word2Vec, GloVe, fastText), then upgrading to context-aware Transformer-based models (Sentence-BERT), and finally integrating Large Language Models (LLM) including Qwen and Llama with advanced techniques like LoRA, QLoRA, and RAG to build an industry-leading talent matching solution.

---

## Core Features

### 1. Intelligent Candidate Ranking

The system can automatically assess how well each candidate matches a specific job, generating a match score from 0 to 1. This score isn't just simple keyword counting—it's based on deep semantic understanding. The system recognizes that "HR Specialist" and "HR Professional" essentially describe the same type of position, even when they share no common vocabulary.

### 2. Dynamic Re-ranking

When a recruiter stars a candidate as an ideal match, the system extracts that candidate's features and adjusts ranking weights for all candidates. This means every star operation helps the system better understand "what makes an ideal candidate for this particular role."

### 3. Multi-level Matching Strategies

The system supports flexible switching between multiple matching algorithms, from fast and efficient TF-IDF to semantically stronger LLM methods. Users can find the optimal balance between precision and speed based on actual needs.

---

## Technical Architecture

### Algorithm Evolution

Our development process follows a progressive methodology, with each generation of technology building meaningful breakthroughs on the previous one.

### Generation 1: TF-IDF Baseline Matching

TF-IDF (Term Frequency-Inverse Document Frequency) is a classic algorithm in information retrieval that performs text similarity matching by calculating the importance of words in documents. The core idea is straightforward: if a word appears frequently in a particular resume but rarely across the entire candidate pool, then this word is particularly important for distinguishing that resume.

For each candidate's job description, we vectorize it and compute cosine similarity with the target job keywords, taking the maximum or average as the candidate's match score. This method is simple to implement and fast to compute, but has obvious limitations when handling synonyms and semantic variations.

### Generation 2: Distributed Word Embeddings

Distributed word embedding techniques map words into dense low-dimensional vector spaces, making semantically similar words closer in that space. Word2Vec, GloVe, and fastText are representative solutions in this category. Despite their different implementations, they all share a common theoretical foundation—the distributional hypothesis: words that appear in similar contexts tend to have similar meanings.

#### 2.1 Word2Vec

Word2Vec formulates representation learning as a prediction problem. In the Skip-gram variant, the optimization objective can be expressed as:

$$
\max \sum_{(w,c)\in D} \log P(c \mid w)
$$

where $w$ is the target word, $c$ is a context word sampled within a fixed window, and the conditional probability is parameterized using a softmax over the dot product of word vectors. In practice, to avoid expensive softmax computations, negative sampling is typically used as an approximation—pulling observed word-context pairs closer while pushing randomly sampled negative samples apart.

The theoretical implication is that Word2Vec learns embeddings shaped entirely by local context prediction. Any global corpus-level statistical regularities emerge implicitly as a side effect of repeatedly solving this local prediction task. As a result, Word2Vec excels at capturing fine-grained semantic similarity but remains fundamentally dependent on word-level tokens and cannot represent unseen vocabulary.

#### 2.2 GloVe

GloVe (Global Vectors for Word Representation) approaches the same problem from an explicitly count-based and global perspective. Rather than predicting context words, it begins by constructing a word-word co-occurrence matrix $X$, where each entry $X_{ij}$ represents how often word $j$ appears in the context of word $i$. The model then learns word vectors by minimizing a weighted least-squares objective of the form:

$$
J = \sum_{i,j} f(X_{ij}) \left( \mathbf{w}_i^\top \mathbf{\tilde{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

Here, the dot product between word vectors is explicitly constrained to approximate the logarithm of co-occurrence counts, and the weighting function $f$ controls the influence of rare and frequent pairs.

The theoretical distinction is crucial: while Word2Vec relies on a neural prediction task to implicitly encode statistics, GloVe directly factorizes a global statistical structure. This makes GloVe embeddings particularly stable and well-aligned with global semantic relationships, but also ties the model to a fixed vocabulary and a precomputed co-occurrence matrix. From a conceptual standpoint, GloVe can be understood as a bridge between classical matrix factorization methods (such as LSA) and neural embeddings.

#### 2.3 fastText

fastText extends the Word2Vec framework by changing the fundamental unit of representation. Instead of treating words as atomic symbols, fastText represents each word as a sum of vectors corresponding to its character-level n-grams. Formally, the embedding of a word $w$ is defined as:

$$
\mathbf{v}_w = \sum_{g \in G(w)} \mathbf{z}_g
$$

where $G(w)$ denotes the set of character n-grams contained in the word and $\mathbf{z}_g$ is the embedding of an individual n-gram. These word representations are then trained using the same Skip-gram objective as Word2Vec.

The theoretical innovation of fastText is that it factorizes morphology into the embedding space itself. Because words are composed from subword units, the model can generate meaningful vectors for rare words, misspellings, or entirely unseen tokens, as long as their character n-grams were observed during training. This shifts the model from a purely lexical representation to a partially compositional one, at the cost of introducing additional noise when unrelated words share similar character patterns.

#### Unified Perspective

From a unifying perspective, Word2Vec can be seen as optimizing a local predictive objective, GloVe as performing a global log-co-occurrence factorization, and fastText as introducing subword-level parameter sharing into a predictive framework. All three models ultimately learn static embeddings, meaning each word is assigned a single vector regardless of context—this reflects their shared limitation and motivates the move toward contextualized representations in more advanced models.

### Generation 3: Transformer Context-Aware Models

Sentence-BERT (SBERT) represents a significant leap forward in NLP technology. Unlike static word embeddings, Transformer models can generate different representations based on the complete context in which words appear. This means "manager" in "Sales Manager" and "Product Manager" will receive different vector representations, because the model understands that context changes meaning.

SBERT was trained on large amounts of sentence pair data using contrastive learning, specifically optimized for computing sentence-level semantic similarity. We use the all-MiniLM-L6-v2 model to generate embeddings for candidate job descriptions, then compute match with target positions through cosine similarity.

This approach demonstrated significant advantages in capturing phrase-level semantic relationships, able to recognize close connections between "Talent Operations" and "Human Resources" even when they share no vocabulary.

### Generation 4: Large Language Model Integration

Our final tech stack integrates Large Language Models, including the Qwen and Llama model families, achieving deep understanding of job requirements and candidate characteristics.

#### 4.1 Prompt Engineering

We designed structured prompt templates containing job descriptions, evaluation criteria, and output format requirements. Through carefully worded prompts, LLMs can directly score candidate fit.

#### 4.2 LoRA Efficient Fine-tuning

LoRA (Low-Rank Adaptation) achieves efficient fine-tuning by introducing trainable low-rank matrices into frozen pre-trained models. This approach reduces trainable parameters by over 99% while maintaining performance close to full fine-tuning. The key insight is that adapting to a new task doesn't require modifying all model parameters—just introducing a small number of task-specific low-rank perturbations.

#### 4.3 QLoRA Quantized Fine-tuning

QLoRA combines 4-bit quantization with adapter fine-tuning. This makes training large language models on consumer-grade GPUs possible, compressing model precision from 16-bit to 4-bit, significantly reducing computational resource requirements while incurring only negligible performance loss.

#### 4.4 RAG Retrieval-Augmented Generation

Retrieval-Augmented Generation combines pre-trained language model knowledge with external knowledge bases. We constructed a knowledge base focused on the HR domain, containing skills, experience, and qualities that make excellent HR professionals. When evaluating candidates, the system first retrieves relevant knowledge, then augments prompts with this context to generate more accurate and informed scores.

#### 4.5 Pairwise Comparison Optimization

To improve ranking consistency, we adopted pairwise comparison methods: instead of evaluating each candidate's match in isolation, we have the model directly compare the relative merits of two candidates. Based on the Bradley-Terry model, we derive consistent ranking order from pairwise decisions—this approach is more stable than absolute scoring and aligns better with human judgment.

---

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Apziva-Project-C

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Main Script Description

| Script | Function Description |
|--------|---------------------|
| `EDA.py` | Exploratory Data Analysis, analyzing candidate distribution features |
| `TF-IDF.py` | TF-IDF-based basic similarity matching |
| `Prompt_only.ipynb` | LLM zero-shot prompt engineering experiments |
| `Lora.ipynb` | LoRA fine-tuning experiments |
| `QLoRA.ipynb` | QLoRA quantized fine-tuning experiments |
| `RAG.ipynb` | Retrieval-Augmented Generation implementation |
| `RAG_Score.ipynb` | Pairwise ranking score optimization |

### Usage Example

```python
from candidate_ranking import CandidateRanker

# Initialize ranker (using SBERT model)
ranker = CandidateRanker(model='sbert')

# Define target position
target_keywords = "Aspiring human resources"

# Rank candidates
ranked_candidates = ranker.rank(candidates_df, target_keywords)

# Print top 10 candidates
print(ranked_candidates.head(10))
```

---

## Performance Evaluation

### Model Comparison

| Method | Coverage | Semantic Quality | Inference Speed |
|--------|-----------|------------------|-----------------|
| TF-IDF | High | Medium | Fast |
| Word2Vec | Medium | Medium-High | Medium |
| GloVe | Medium | Medium | Medium |
| fastText | High | High | Medium |
| Sentence-BERT | High | Very High | Slow |
| Prompt-Only LLM | High | Very High | Very Slow |
| QLoRA | High | Very High | Medium |
| RAG + Pairwise | Highest | Highest | Medium |

Each generation of technology brought meaningful quality improvements. LLM with RAG and pairwise evaluation achieves the highest level of ranking quality while maintaining acceptable computational efficiency through LoRA/QLoRA techniques.

### Key Findings

Static embedding methods perform poorly when handling out-of-vocabulary words. FastText outperforms Word2Vec and GloVe in this regard through subword information. SBERT provides excellent semantic understanding at the sentence level. LLM combined with RAG achieves the deepest understanding of HR domain knowledge, with pairwise ranking ensuring consistency in relative ordering.

---

## Challenges and Solutions

### Challenge 1: Limited Compute Resources

Training large language models requires significant GPU memory. Our solution was utilizing Google Colab's T4 GPUs for accelerated training while applying QLoRA's 4-bit quantization technology to reduce memory requirements. Additionally, we optimized resource usage by reducing batch sizes and sequence lengths.

### Challenge 2: Out-of-Vocabulary Words

Static embedding methods struggle when encountering rare or new words. Our coping strategies included FastText's subword information handling, the inherent text understanding ability of LLMs, and RAG's retrieval capabilities combined with domain-specific vocabulary.

### Challenge 3: Ranking Consistency

Different similarity scores may conflict. Solutions included adopting pairwise comparison methods, introducing the Bradley-Terry probabilistic model for consistency calibration, and using ensemble voting mechanisms when necessary.

---

## Future Improvements

Multi-modal fusion will be an important next direction, integrating more candidate features like skills and experience for more comprehensive matching evaluation. Deploying real-time inference endpoints will enable the system to support production-level high-concurrency requests. Expanding the HR knowledge base to cover a wider range of job types will enhance system versatility and practicality. Introducing active learning mechanisms will enable continuous improvement from user feedback, reducing reliance on labeled data. Applying interpretability techniques like SHAP will help understand the logic behind LLM decisions, enhancing system trustworthiness and transparency.

---

## Project Structure

```
Apziva-Project-C/
├── EDA.py                    # Exploratory Data Analysis
├── TF-IDF.py                 # TF-IDF Baseline Matching
├── Prompt_only.ipynb         # LLM Prompt Engineering
├── Lora.ipynb               # LoRA Fine-tuning
├── QLoRA.ipynb              # QLoRA Quantized Fine-tuning
├── RAG.ipynb                # Retrieval-Augmented Generation
├── RAG_Score.ipynb          # Pairwise Ranking Scoring
├── README.md                # Project Documentation
├── requirements.txt         # Dependency List
└── images/                  # Documentation Images
```

---

## Tech Stack

- **Data Processing**: pandas, numpy, scikit-learn
- **Traditional NLP**: TF-IDF, NLTK, spaCy
- **Word Embeddings**: gensim (Word2Vec, GloVe, fastText)
- **Transformer Models**: sentence-transformers, transformers
- **LLM Fine-tuning**: peft (LoRA, QLoRA)
- **Vector Retrieval**: ChromaDB (RAG Knowledge Base)
- **Experiment Environment**: Google Colab (GPU Acceleration)

---

## Contributors

Thanks to the Apziva team for their support and guidance.

---

## License

This project is for internal use only.

---

## References

1. Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR*.

2. Pennington, J., et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.

3. Bojanowski, P., et al. (2017). Enriching Word Vectors with Subword Information. *TACL*.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.

5. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

6. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv*.

7. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive Tasks. *arXiv*.
