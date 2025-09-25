# MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents

> **TL;DR.** We introduce MobileRL, an online agentic reinforcement learning framework that turns general-purpose vision-language models into strong mobile GUI agents. By combining a staged reasoning warm-up with difficulty-adaptive online RL, MobileRL achieves state-of-the-art success rates on AndroidWorld and AndroidLab. 


## Open-Source Roadmap
- [x] **Evaluation framework** 
- [ ] **MobileRL-9B checkpoint** —  **Will be open-sourced soon upon legal approval.**


## Quick Start Guide

This guide will help you get started quickly with our evaluation framework.  
Please follow the steps in the order provided.

---

### Step 1: Hardware Requirements

The Android Emulator requires **KVM (Kernel-based Virtual Machine)** support on the host machine.  
You can verify if your system supports KVM by running:

```bash
apt-get install cpu-checker
kvm-ok
```

---

### Step 2: Download AVD Images

We provide packaged test environments for **AndroidWorld** and **AndroidLab** as Docker images to simplify setup and ensure reproducibility.
Before proceeding, pull the required Docker images:

```shell
docker pull xuyifan0731/mobilerl-androidlab-eval
docker pull xuyifan0731/mobilerl-androidworld-eval
```

---

### Step 3: Usage Modes

We support two modes of usage:

* **Local Testing** – Recommended for quick debugging and making modifications.
* **Docker-based Deployment with AgentRL** – Provides a consistent, containerized environment for convenient deployment.

For detailed usage instructions, please refer to [inference/README.md](inference/README.md).



## Method

![Framework overview](assets/androidrl-main.png)

Mobile GUI agents must follow complex instructions, reason over cluttered screens, and act under sparse, delayed rewards—all while task difficulty is heavy-tailed and environment sampling is expensive.  
**MobileRL** addresses these challenges with a two-stage recipe:

1. **Reasoning Warm-up:**  
   - **reasoning-free sft** on large expert data.  
   - **reasoning sft** to inject and polish rationale-driven planning and transparency.

2. **Online Agentic RL (Difficulty–Adaptive GRPO, AdaGRPO):**  
   - **Adaptive Positive Replay (AdaPR):** store high-quality trajectories and re-use them efficiently.  
   - **Failure Curriculum Filtering (FCF):** prune low-quality rollouts and focus learning on actionable tasks.  
   - **Shortest-Path Reward Adjustment (SPA):** reward shaping that stabilizes credit assignment for long-horizon interactions.
