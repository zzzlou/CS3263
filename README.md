# LATTE: Learning Attendance Timetables through Tailored Experiences

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Personalized AI-based timetable generation and attendance prediction system.**

---

## Overview

**LATTE** is an AI framework designed to generate university timetables that align with individual student profiles, considering not just feasibility (no clashes) but also **behavioural preferences** such as avoiding early classes, preserving free days, and managing academic workload.

Unlike traditional timetabling systems, LATTE models **day-to-day attendance behaviour** using Reinforcement Learning from Human Feedback (RLHF) and outputs **realistic, persona-aligned attendance trajectories**.

🔗 **Project Report**: [Project Report](docs/LATTE_Report.pdf)

🔗 **Paper Abstract**: See below

---

## Motivation

- Traditional timetable generators ignore **real-world behaviours** like fatigue or GPA trade-offs.
- Students often cannot **explicitly articulate** their true preferences.
- LATTE closes this gap by **learning from behavioural simulations**, producing **personalized and realistic** attendance plans.

---

## Key Features

- 📚 **Constraint Satisfaction** (AC-3) to eliminate infeasible timetables
- 🔎 **Heuristic-guided Beam Search** with Simulated Annealing and Random Restarts
- 🎯 **Persona Modeling** (e.g., Lazy, Hardworking) via interpretable heuristics
- 🤖 **RLHF**: Training a reward model from **pairwise GPT-4o simulated preferences**
- 🏫 **MDP-based Attendance Simulation** using OpenAI Gym
- 🚀 **Q-learning** to derive personalized attendance strategies

---

## Methodology

1. **Timetable Search**:
   - Filter using **AC-3 Constraint Satisfaction**.
   - Rank feasible schedules with **custom heuristics** (free days, early mornings, etc.).
2. **Attendance Decision Modeling**:
   - Model daily attendance as a **Markov Decision Process (MDP)**.
   - States include GPA level, fatigue, day of the week, start time, and class importance.
3. **Preference Learning**:
   - Generate simulated pairwise preference data using **GPT-4o**.
   - Train a **neural reward model** via **binary cross-entropy loss**.
4. **Policy Optimization**:
   - Optimize attendance strategies with **Q-learning**, using the learned reward model for feedback.

---

## Installation

```
git clone https://github.com/zzzlou/CS3263.git
cd CS3263
pip install -r requirements.txt
```

## Results

- **Reward Model Accuracy**: ~92% preference agreement

- **Policy Comparison**:
  - Random Policy: Reward 4.9
  - Rule-based Lazy Policy: Reward 7.1
  - **Q-learning Policy (ours): Reward 10.2**

- **Visualization**: Realistic, persona-aligned attendance trajectories (Lazy profile prefers afternoon classes, avoids Fridays)
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [OpenAI](https://openai.com/) for GPT-4o assistance in preference simulation.
- Inspired by research on RLHF and behaviour-aware educational planning.

---
