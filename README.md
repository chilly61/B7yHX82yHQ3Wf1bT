# pro-C
This is the Apziva Project C - Potential Talents

**Algorithm Analysis of NLP**
1. Distributional Word Embeddings: Common Theoretical Foundation
All three models—**Word2Vec**, **GloVe**, and **fastText**—are based on the distributional hypothesis, which states that words appearing in similar contexts tend to have similar meanings. Each model learns dense vector representations of words by exploiting statistical regularities in large text corpora, such that semantic similarity between words can be approximated by geometric proximity in the embedding space.

Despite this shared foundation, the three models differ significantly in how contextual information is modeled, what linguistic units are learned, and how they handle rare or unseen words.

2. Comparison

   2.1 Word2Vec
       Word2Vec formulates representation learning as a predictive problem. Instead of explicitly constructing a global co-occurrence matrix, it defines an objective that maximizes the likelihood of observing context words given a target word (or vice versa). In the Skip-gram formulation, the optimization objective can be written as
   
   2.2 Glove

   2.3 FastText
