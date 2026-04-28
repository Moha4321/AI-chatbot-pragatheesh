# Lesson 2: Words as Math (Vector Embeddings)

LLMs don't read English; they do math. To make our chatbot detect "Junk Food Cravings," we have to turn sentences into numbers. We call these **Vectors**.

### 1. What is a Vector?
Imagine a 2D graph. 
*   X-axis is "Sweetness"
*   Y-axis is "Salty/Savory"

The word "Cake" might be at coordinate `[0.9, 0.1]`.
The word "Fries" might be at `[0.1, 0.9]`.
The word "Burger" might be at `[0.2, 0.9]`.

Notice how "Fries" and "Burger" are very close together? In AI, we don't use 2D graphs. We use **384-dimensional graphs**. A sentence becomes a list of 384 numbers: `[0.12, -0.44, 0.81, ...]`.

### 2. Cosine Similarity (How we detect cravings)
If we have a vector for the user's message, and a vector for "I want junk food", how do we know if they mean the same thing? We measure the angle between the two arrows on the graph.
*   **Small angle (arrows point same way):** Cosine is close to `1.0`. They mean the same thing!
*   **Large angle (arrows point away):** Cosine is close to `0.0` or `-1.0`. They mean different things.

**The Math Formula (You will code this in Python!):**
`Cosine_Similarity(A, B) = (A · B) / (||A|| * ||B||)`
*(Don't worry, it's just multiplying lists of numbers and adding them up).*

### 3. The "Negation" Problem (Why we need advanced math)
If a user says, *"I want a burger,"* the AI flags it as junk food. 
But what if they say, *"I HATE burgers!"*? 
Because the word "burger" is in the sentence, a basic AI gets confused and still thinks it's a craving. 

To fix this, we will use a **Negation Correction Formula**:
`Final_Score = sim(User, JunkFood) - sim(User, Negation)`
If the user hates burgers, their sentence will be very similar to our Negation Vector, pulling the final score down. This is the math that makes your project a legendary research paper!