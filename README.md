### Project Overview: Jump-Dog Heuristic Enhancement

This project focuses on extending the difficulty levels of the board game **Jump-Dog**, a strategic game inspired by the mechanics of Checkers. The AI player, which currently uses the Mini-Max algorithm with varying depths for difficulty levels, will be augmented by introducing a heuristic model to enhance performance. The goal is to support deeper calculations (up to 8-10 levels) without sacrificing efficiency.

---

### Relation to the Main Project

This enhancement serves as an advanced component of the Jump-Dog game. It complements the existing Mini-Max algorithm by improving move ordering, enabling the algorithm to explore deeper game trees within the same computational constraints. By introducing a heuristic model based on Artificial Neural Networks (ANNs), this project aligns closely with the game's core functionality while elevating its strategic depth.

---

### Motivation

The primary motivation for this enhancement stems from the desire to create a more challenging AI player capable of competing with seasoned players. The current Mini-Max implementation struggles with higher depth levels due to performance bottlenecks. By integrating a heuristic model, the AI's move prioritization will become more efficient, paving the way for deeper and more intelligent gameplay.

---

### Path

1. **Problem Statement:** Design and implement a heuristic model to reorder valid moves for the Mini-Max algorithm. This will use Artificial Neural Networks to predict the board state's desirability for the AI player.

2. **Input Data:** 
   - **Representation:** A 5x5 matrix with cell values as:
     - `1` for black pieces (player 1),
     - `-1` for white pieces (player 2),
     - `0` for empty cells.

3. **Output Data:**
   - A scalar integer between `-100` and `100`, representing the desirability of the current board state:
     - `-100`: Very bad situation for player 1,
     - `100`: Very good situation for player 1,
     - `0`: Balanced state between players.

4. **Neural Network Design:** Evaluate various neural network architectures (e.g., one-layer, multi-layer, RNN, CNN, Transformer, LSTM) to determine the best fit for predicting the board state efficiently. The network will be trained using the collected dataset of board states and their corresponding desirability scores.

5. **Implementation:** Integrate the trained heuristic model into the Mini-Max algorithm to reorder moves based on predicted board state desirability, thereby optimizing its decision-making process.

6. **Resources:**
   - Training dataset details and preprocessing methods are documented [here](./training_data.md).
   - Exploration of neural network architectures is detailed [here](./neural_network_choices.md).

---

### Additional Resources

- [Comparing Various Neural Network Archetecture](./docs/Choosing_between_Neural_Networks.md)
- [Choosing Model Parameters](./docs/Choosing_Models_Parameters.md)
- [Choosing Training Parameters](./docs/Choosing_Trainings_Parameters.md)
- [Code Explanations](./docs/Code_Explanations.md)

