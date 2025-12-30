# pro-C
This is the Apziva Project C - Potential Talents

**Algorithm Analysis of NLP**
1. Distributional Word Embeddings: Common Theoretical Foundation
All three models—**Word2Vec**, **GloVe**, and **fastText**—are based on the distributional hypothesis, which states that words appearing in similar contexts tend to have similar meanings. Each model learns dense vector representations of words by exploiting statistical regularities in large text corpora, such that semantic similarity between words can be approximated by geometric proximity in the embedding space.

Despite this shared foundation, the three models differ significantly in how contextual information is modeled, what linguistic units are learned, and how they handle rare or unseen words.

2. Comparison

   2.1 Word2Vec

      Word2Vec formulates representation learning as a predictive problem. Instead of explicitly constructing a global co-occurrence matrix, it defines an objective that maximizes the likelihood of observing context words given a target word (or vice versa). In the Skip-gram formulation, the optimization objective can be written as
   
   $$
\max \sum_{(w,c)\in D} \log P(c \mid w)
$$

      where 𝑤 is a target word, 𝑐 is a context word sampled within a fixed window, and the conditional probability is parameterized using a softmax over the dot product of word vectors. In practice, this objective is approximated using negative sampling, which replaces the expensive softmax with a contrastive loss that pushes observed word–context pairs closer together while pushing randomly sampled pairs apart.

      The theoretical implication is that Word2Vec learns embeddings that are shaped entirely by local context prediction. Any global corpus-level statistics emerge implicitly as a side effect of repeatedly solving this local prediction task. As a result, Word2Vec excels at capturing fine-grained semantic similarity but remains fundamentally dependent on word-level tokens and cannot represent unseen vocabulary.

   2.2 Glove

      GloVe approaches the same problem from an explicitly count-based and global perspective. Rather than predicting context words, it begins by constructing a word–word co-occurrence matrix 
𝑋, where each entry 𝑋𝑖𝑗 represents how often word 𝑗 appears in the context of word 𝑖. The model then learns word vectors by minimizing a weighted least-squares objective of the form

      $$ 
      J = \sum_{i,j} f(X_{ij})
      \left(
      \mathbf{w}_i^\top \mathbf{\tilde{w}}_j + b_i + \tilde{b}_j - \log X_{ij}
      \right)^2
      $$

     Here, the dot product between word vectors is explicitly constrained to approximate the logarithm of co-occurrence counts, and the weighting function 𝑓 controls the influence of rare and frequent pairs.

     The theoretical distinction is crucial: while Word2Vec relies on a neural prediction task to implicitly encode statistics, GloVe directly factorizes a global statistical structure. This makes GloVe embeddings particularly stable and well-aligned with global semantic relationships, but also ties the model to a fixed vocabulary and a precomputed co-occurrence matrix. From a conceptual standpoint, GloVe can be understood as a bridge between classical matrix factorization methods (such as LSA) and neural embeddings.

     2.3 FastText
         fastText extends the Word2Vec framework by changing the fundamental unit of representation. Instead of treating words as atomic symbols, fastText represents each word as a sum of vectors corresponding to its character-level n-grams. Formally, the embedding of a word 𝑤 is defined as:
         
      $$
      \mathbf{v}_w = \sum_{g \in G(w)} \mathbf{z}_g
      $$
   
      where 𝐺(𝑤) denotes the set of character n-grams contained in the word and 𝑧𝑔​ is the embedding of an individual n-gram. These word representations are then trained using the same Skip-gram objective as Word2Vec.
   
      The theoretical innovation of fastText is that it factorizes morphology into the embedding space itself. Because words are composed from subword units, the model can generate meaningful vectors for rare words, misspellings, or entirely unseen tokens, as long as their character n-grams were observed during training. This shifts the model from a purely lexical representation to a partially compositional one, at the cost of introducing additional noise when unrelated words share similar character patterns.

From a unifying perspective, Word2Vec can be seen as optimizing a local predictive objective, GloVe as performing a global log-co-occurrence factorization, and fastText as introducing subword-level parameter sharing into a predictive framework. All three models ultimately learn static embeddings, meaning each word is assigned a single vector regardless of context, which reflects their shared limitation and motivates the move toward contextualized representations in more advanced models.
