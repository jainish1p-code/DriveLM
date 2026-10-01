# CMPE 249 Project Proposal

**Course:** CMPE 249 – Intelligent Autonomous Systems  
**Project Title:** Language-Grounded Driving Reasoning and Discrete Decision Making via Graph VQA on DriveLM  
**Author:** Jainish Patel  
**Target System:** Open-Source Vision-Language-Action (VLA) Pipeline for Autonomous Driving  

---

## 1. Executive Summary & AI Novelty Audit

### 1.1 Project Objective
This project implements a modular Vision-Language-Action (VLA) pipeline that fine-tunes **Qwen2-VL-7B** via Low-Rank Adaptation (LoRA) to execute multi-stage Graph Visual Question Answering (GVQA) on the `DriveLM-nuScenes` dataset. Rather than forcing the vision-language backbone to output unconstrained spatial coordinates directly, the model discretizes high-level tactical choices into a 10-class canonical action space. These discrete commands directly condition a classical, rule-based Frenét-frame motion planner to output smooth, kinematically valid continuous trajectories $[(x_1, y_1), \dots, (x_N, y_N)]$.

### 1.2 AI Novelty & Feasibility Audit
* **Core Novelty (Tactical Action Abstraction):** Pure vision-language models frequently suffer from numerical drift, poor spatial regression, and unsafe trajectory generation when forced to output continuous coordinate tokens directly. **Our novel contribution lies in abstracting driving decisions into a structured 10-class tactical meta-action taxonomy ($\mathcal{A}$) that acts as a semantically grounded interface.** This decouples slow, high-level multi-modal reasoning from fast, deterministic, physics-constrained motion planning.

---

## 2. Technical Architecture & Workflow  
```
+-----------------------------------------------------------------+
|     6x Surround-View Cameras (nuScenes) + Navigation Goal       |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|              Qwen2-VL-7B Vision Encoder (Dynamic ViT)           |
+--------------------------------+--------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------+
|         Graph VQA Sequential Reasoning Engine (LoRA)            |
|  (1. Perception --> 2. Prediction --> 3. Planning --> Action)   |
+--------------------------------+--------------------------------+
                                 | (Discrete Meta-Action Output)
                                 v
+-----------------------------------------------------------------+
|                Classical Frenet Motion Planner                  |
|  (Generates [x,y] coordinates conditioned on target velocity)   |
+-----------------------------------------------------------------+
```

### 2.1 Graph VQA Task Formulation
Following the `DriveLM` benchmark paradigm, the model processes multi-view camera inputs $I_{1:6}$ and answers an interconnected chain of visual reasoning questions across four sequential stages:
1. **Perception ($\mathcal{P}_1$):** Identifies key dynamic actors and roadside hazards (e.g., *"Pedestrian standing near the crosswalk on the right lane"*).
2. **Prediction ($\mathcal{P}_2$):** Estimates actor future trajectories and intentions (e.g., *"Pedestrian is stepping into the ego-vehicle lane"*).
3. **Planning ($\mathcal{P}_3$):** Generates high-level behavioral rules (e.g., *"Yield right-of-way to pedestrian, bring ego-vehicle to a full stop"*).
4. **Behavior / Meta-Action ($\mathcal{B}$):** Selects a discrete tactical command from an expanded 10-class canonical action space:
   $$\mathcal{A} \in \begin{Bmatrix} \text{STOP}, & \text{YIELD}, & \text{MAINTAIN\_SPEED}, & \text{ACCELERATE}, & \text{DECELERATE}, \\ \text{LANE\_CHANGE\_LEFT}, & \text{LANE\_CHANGE\_RIGHT}, & \text{SLIGHT\_LEFT\_NUDGE}, & \text{SLIGHT\_RIGHT\_NUDGE}, & \text{TURNING\_MANEUVER} \end{Bmatrix}$$

### 2.2 Trajectory Interface
The VLM's discrete decision $\mathcal{B}$ sets the target profile velocity $v_{\text{target}}$ and lateral offset $d_{\text{target}}$ for a rule-based Frenét motion planner, which outputs smooth, kinematically valid Cartesian trajectories $(x_t, y_t)$ over a $3.0\text{-second}$ horizon.

---

## 3. Literature & SOTA Survey

| Paper / Framework | Venue / Year | Core Methodology & Technical Relevance |
| :--- | :--- | :--- |
| **DriveLM** (Sima et al.) | ECCV 2024 | Introduces Graph VQA connecting perception, prediction, and planning reasoning chains. Primary benchmark dataset. |
| **DriveVLM** (Tian et al.) | ECCV 2024 | Proposes a hybrid architecture decoupling slow VLM CoT reasoning from fast classical trajectory generation. |
| **SteerVLA** (Mao et al.) | arXiv 2026 | Demonstrates VLA policy steering via language-grounded discrete meta-actions in complex urban scenarios. |
| **StyleVLA** (Zhang et al.) | arXiv 2026 | Evaluates discrete vs. continuous action spaces in VLMs, establishing kinematic advantages of discrete token interfaces. |
| **LinguDrive** (Li et al.) | CVPR 2024 | Utilizes LLMs for high-level decision making to condition low-level continuous execution controllers. |
| **UniAD** (Hu et al.) | CVPR 2023 | SOTA end-to-end perception-to-planning framework on nuScenes used as a classic baseline reference. |
| **Qwen2-VL** (Wang et al.) | arXiv 2024 | Open-source VLM backbone with dynamic visual resolution used as our foundation reasoning engine. |
| **LoRA** (Hu et al.) | ICLR 2022 | Parameter-efficient fine-tuning protocol ($r=16, \alpha=32$) enabling VLM adaptation on single-GPU hardware. |

---

## 4. Experimental Plan & Evaluation Protocol

### 4.1 Dataset Split & Protocols
* **Dataset:** `DriveLM-nuScenes` ($32,000+$ keyframe QA pairs across $1,000$ driving scenes).
* **Partitioning Strategy:** Scene-level split ($70\%$ train, $15\%$ validation, $15\%$ test) to eliminate temporal data leakage between adjacent frames.

### 4.2 Quantitative Metrics
1. **Reasoning Quality (NLP Metrics):** BLEU-4, ROUGE-L, and CIDEr scores on intermediate perception, prediction, and planning reasoning paths.
2. **Decision Accuracy (Classification Metrics):** Accuracy, Precision, Recall, and F1-Score across predicted 10-class discrete meta-actions ($\mathcal{A}$) relative to ground-truth human driver commands.
3. **Planning Validation (Robotics Metrics):** Average Displacement Error (ADE) and Final Displacement Error (FDE) of the Frenét planner when conditioned on predicted meta-actions vs. ground-truth actions.

---

## 5. Compute Plan & Resource Allocation

* **Primary Compute Resource:** NSF ACCESS Allocation at Pittsburgh Supercomputing Center utilizing **PSC Bridges-2 GPU** (99 GPU Hours remaining, active through Aug 20, 2027)[cite: 1].
* **Storage Allocation:** **PSC Ocean** distributed storage (100 GB active allocation) for staging the `DriveLM-nuScenes` dataset and LoRA checkpoints[cite: 1].
* **Local Fallback Option:** $1\times$ NVIDIA RTX 3090 / 4090 ($24\text{GB}$ VRAM) workstation using DeepSpeed ZeRO-Offload and Gradient Checkpointing.
* **Software Environment:** Containerized PyTorch 2.x environment with Hugging Face `transformers`, TRL, PEFT, and `OpenDriveLab/DriveLM` official evaluation toolkit.
