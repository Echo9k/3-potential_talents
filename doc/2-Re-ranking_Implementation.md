### Re-ranking Implementation

The re-ranking process is essential to dynamically adjust candidate rankings based on user feedback (e.g., starring an ideal candidate). Here’s how to implement it step-by-step:

---

### 1. **Objective**
The goal of re-ranking is to adjust the candidate list dynamically so that candidates similar to a starred candidate (manually marked as ideal) are prioritized higher. This process ensures that subsequent rankings align better with user preferences.

---

### 2. **Approach**
We’ll use a combination of **distance-based similarity** and **ranking loss** techniques to re-rank the candidates.

---

#### Step 1: **Initial Candidate Ranking**
- Rank candidates using the original model (e.g., fit scores computed via a meta-model or ensemble of similarity scores). 
- Sort candidates by the predicted fit scores in descending order.

#### Step 2: **Feedback (Starring a Candidate)**
- Allow the user to select a candidate (e.g., the 7th candidate) as the reference point for re-ranking.
- This starred candidate becomes the **reference vector** for similarity comparisons.

#### Step 3: **Similarity Calculation**
For each candidate in the list, compute a similarity score relative to the starred candidate. Use any or a combination of the following metrics:
1. **Cosine Similarity**:
   - Use the feature embeddings (e.g., TF-IDF, BERT, or hybrid embeddings) for the job title, location, and other numerical features (like `connections`) to calculate the cosine similarity between the starred candidate and others.

   $$
   \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
   $$

2. **Euclidean Distance**:
   - For normalized numerical features (e.g., `TF-IDF_fit_score`, `BERT_model_fit_score`), compute the Euclidean distance between the starred candidate and others.

   $$
   \text{Euclidean Distance}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
   $$

3. **Weighted Similarity**:
   - Combine multiple similarity scores using a weighted average. For example:
     $$
     \text{Weighted Similarity} = w_1 \cdot \text{Cosine Similarity} + w_2 \cdot (1 - \text{Euclidean Distance}) + \dots
     $$

#### Step 4: **Re-rank Candidates**
1. **Update Fit Scores**:
   - Adjust the original fit scores by incorporating similarity to the starred candidate. For example:
     $$
     \text{Adjusted Fit Score} = \alpha \cdot \text{Original Fit Score} + \beta \cdot \text{Similarity Score}
     $$
     where:
     - $\alpha, \beta$: Weights balancing original fit and similarity scores (tune based on feedback).

2. **Sort the List**:
   - Re-sort the candidate list based on the adjusted fit scores in descending order.

---

### 3. **Iterative Re-ranking**
Repeat the process whenever a new candidate is starred:
1. Calculate similarity scores between the newly starred candidate and the rest.
2. Update adjusted fit scores using the new similarity scores.
3. Re-rank the candidates again.

---

### 4. **Implementation Example in Python**
Here’s a simplified implementation using cosine similarity for re-ranking:

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example data
data = {
    "id": [1, 2, 3, 4],
    "job_title_embedding": [
        [0.2, 0.8, 0.5], 
        [0.1, 0.7, 0.6], 
        [0.9, 0.4, 0.3], 
        [0.3, 0.8, 0.7]
    ],
    "original_fit_score": [0.8, 0.6, 0.9, 0.5],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Assume the user stars the 3rd candidate (index 2)
starred_index = 2
starred_embedding = np.array(df.loc[starred_index, "job_title_embedding"]).reshape(1, -1)

# Calculate cosine similarity
df["similarity_score"] = cosine_similarity(df["job_title_embedding"].tolist(), starred_embedding).flatten()

# Re-rank based on adjusted fit score
alpha = 0.7  # Weight for original fit score
beta = 0.3   # Weight for similarity score
df["adjusted_fit_score"] = alpha * df["original_fit_score"] + beta * df["similarity_score"]

# Sort by adjusted fit score
df = df.sort_values(by="adjusted_fit_score", ascending=False).reset_index(drop=True)

# Display the updated rankings
print(df[["id", "original_fit_score", "similarity_score", "adjusted_fit_score"]])
```

---

### 5. **Key Considerations**
- **Tuning Parameters ($\alpha$, $\beta$)**:
  - Balance the influence of the original fit score and the similarity score. Start with equal weights and adjust based on feedback.
- **Similarity Features**:
  - Experiment with which features to use for similarity (e.g., embeddings for job titles, numerical features like connections, or combined embeddings).
- **Bias Mitigation**:
  - Regularly audit the rankings to ensure that certain groups or candidates are not being unfairly penalized during re-ranking.

---

### 6. **Evaluation Metrics**
To assess the quality of re-ranking:
1. **Rank Correlation**:
   - Measure how well the adjusted rankings align with user preferences (e.g., Kendall’s Tau or Spearman’s rank correlation).
2. **User Feedback**:
   - Collect qualitative feedback from users on whether the re-ranked list improves relevance.
3. **Top-N Accuracy**:
   - Track how often the top N candidates include the final selection.

This re-ranking approach ensures flexibility and continuous improvement as the system incorporates user feedback.