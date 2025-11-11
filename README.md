# CSCN8020 – Assignment 3: Deep Q-Learning on Pong

**Student Name:** Kapil Bhardwaj  
**Student ID:** 9064347  
**Course:** CSCN8020 – Reinforcement Learning  


---

## 🧠 Project Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to play **PongDeterministic-v4** from the Atari environment using **Reinforcement Learning**.

The DQN agent uses:
- Convolutional Neural Network (CNN) to approximate Q-values from raw image frames  
- Experience replay for stable learning  
- Target network to reduce Q-value oscillations  
- ε-greedy exploration for balancing exploration and exploitation  

A fallback to **CartPole-v1** is included for CPU testing if Atari ROMs are unavailable.

---

## ⚙️ Implementation Summary

### Components
| Component | Description |
|------------|-------------|
| `assignment3_utils.py` | Provided utility file for frame preprocessing |
| `FrameStack` | Maintains last 4 frames to capture ball motion |
| `ReplayBuffer` | Stores transitions for experience replay |
| `DQN` | CNN model with 3 conv layers + 2 FC layers |
| `train_dqn()` | Core training loop handling ε-decay, replay, target updates |
| `plot_results()` | Generates reward and average reward graphs |

### Architecture
```

Input: 4 stacked grayscale frames (84 × 80)
Conv1: 32 filters, 8×8 kernel, stride 4, ReLU
Conv2: 64 filters, 4×4 kernel, stride 2, ReLU
Conv3: 64 filters, 3×3 kernel, stride 1, ReLU
FC1:   512 units, ReLU
Output: #actions (6 for Pong)

````

---

## 🧩 Experimental Setup

Four experiments were run by varying **mini-batch size** and **target network update frequency**:

| Experiment | Batch Size | Target Update Interval |
|-------------|-------------|-------------------------|
| Exp 1 | 8 | 10 episodes |
| Exp 2 | 16 | 10 episodes |
| Exp 3 | 8 | 3 episodes |
| Exp 4 | 16 | 3 episodes |

Each run collected:
- Score per episode  
- Average cumulative reward (last 5 episodes)

Plots were generated to visualize learning trends.

---

## 🧪 Requirements

Install required dependencies:
```bash
pip install torch torchvision matplotlib numpy gymnasium[atari,accept-rom-license]
pip install autorom
python -m autorom
````

If `PongDeterministic-v4` fails to load, the script automatically switches to `CartPole-v1` for testing.

---

## 🚀 How to Run

1. Place all files in the same folder:

   ```
   CSCN8020_Assignment3/
   ├── CSCN8020_Assignment3_DQN.ipynb
   ├── assignment3_utils.py
   ├── CSCN8020_Assignment3_Report.pdf
   └── README.md
   ```

2. Open **Jupyter Notebook**:

   ```bash
   jupyter notebook CSCN8020_Assignment3_DQN.ipynb
   ```

3. Run the cells sequentially.

4. To execute full experiments:

   ```python
   experiments = [
       {'batch_size': 8, 'target_update_episodes': 10},
       {'batch_size': 16, 'target_update_episodes': 10},
       {'batch_size': 8, 'target_update_episodes': 3},
       {'batch_size': 16, 'target_update_episodes': 3},
   ]
   ```

5. View results:

   * **Reward per Episode**
   * **Average Reward (last 5 episodes)**

---

## 📊 Results Summary

* Smaller batches (8) → faster but noisier learning
* Larger batches (16) → smoother updates, slower adaptation
* Frequent target updates (3 ep) → unstable Q-values
* Less frequent updates (10 ep) → more stable convergence

**Best configuration:**
`Batch size = 16`, `Target update = 10 episodes`

---

## 🏁 Conclusion

This implementation successfully demonstrates Deep Q-Learning on Pong (or CartPole fallback).
It covers all major RL components: state preprocessing, replay buffer, CNN-based Q-value approximation, and target network updates.
The experiments confirm how hyperparameters like batch size and update frequency affect learning performance.

---

## 📚 References

* Mnih, V. et al. (2015). *Playing Atari with Deep Reinforcement Learning*. DeepMind.
* OpenAI Gym Documentation – [https://www.gymlibrary.dev](https://www.gymlibrary.dev)
* CSCN8020 Course Materials and Lecture Notes

---

### ✅ Files Included

| File                              | Description                           |
| --------------------------------- | ------------------------------------- |
| `CSCN8020_Assignment3_DQN.ipynb`  | Jupyter notebook implementing DQN     |
| `assignment3_utils.py`            | Frame preprocessing utility           |
| `CSCN8020_Assignment3_Report.pdf` | Written report with results and plots |
| `README.md`                       | Project documentation                 |

---

**Author:** Kapil Bhardwaj
**Conestoga College – 2025**

```
