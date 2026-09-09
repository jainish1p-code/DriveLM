# DriveLM
# Fine-Tuning Open-Source VLMs for Chain-of-Thought Driving Logic and Waypoint Planning
 
**Team:** Jainish Patel
**Track:** Track 2: Vision-Language-Action (VLA) & End-to-End Driving

---

## Abstract
Autonomous vehicle safety relies heavily on interpreting complex visual scenes and executing transparent decision-making. Standard end-to-end neural networks generate trajectories as "black boxes," making edge-case failures difficult to audit. This project implements parameter-efficient fine-tuning (LoRA) on open-source Vision-Language Models (Qwen2-VL-7B and LLaVA-NeXT) using the `DriveLM-nuScenes` benchmark dataset. By outputting step-by-step Chain-of-Thought (CoT) driving logic (perception → object interaction → behavior reasoning → trajectory waypoints), this system aligns high-level visual reasoning with physical coordinate control. We evaluate zero-shot versus fine-tuned reasoning quality (Graph VQA accuracy / BLEU-4) and trajectory accuracy (Average Displacement Error / ADE) on a single consumer GPU.

---

## Literature & SOTA Survey Results

| Paper / Model | Venue / Year | Key Methodology & Contribution | Relevance to Project |
| :--- | :--- | :--- | :--- |
| **DriveVLM** | ECCV 2024 | Proposes visual Chain-of-Thought reasoning (scene description → critical object interaction → intent prediction → meta-actions) coupled with a classical motion planner. | Serves as the primary baseline for CoT driving logic. |
| **DriveLM** | ECCV 2024 | Introduces Graph Visual Question Answering (GVQA) connecting perception, prediction, and planning QA pairs in a graph-structured sequence. | Source of our multi-modal dataset (`DriveLM-nuScenes`). |
| **LMDrive** | CVPR 2024 | Closed-loop end-to-end driving framework utilizing multi-view visual sensors and LLMs for instruction-following driving in CARLA. | Informs prompt formatting and waypoint tokenization setup. |
| **Qwen2-VL** | arXiv 2024 | Uses Naive Dynamic Resolution and 3D Multi-Head Rotary Position Embeddings (M-RoPE) to model complex multi-image visual inputs. | Selected as our primary open-source VLM backbone. |
| **CarLLaVA** | arXiv 2024 | Adapts LLaVA-NeXT to predict trajectories directly from visual tokens, proving general open VLMs can match specialized planners when fine-tuned. | Baseline model architecture for fine-tuning comparisons. |

---

## AI Novelty & Feasibility Audit

### AI Critique Summary
> *"The project avoids the 'red ocean' of pre-training foundation models or world models from scratch, which requires multi-GPU clusters. Instead, it targets a high-value physical AI niche: adapting general-purpose open-source VLMs for explainable autonomous vehicle control via parameter-efficient fine-tuning (LoRA). Combining textual reasoning metrics (BLEU-4/ROUGE) with numerical trajectory error (ADE/FDE) provides the academic rigor required for autonomous systems research within realistic single-GPU compute constraints."*

### Red Ocean & Risk Mitigation Strategy
* **Risk 1: Simulator Setup Overhead (e.g., CARLA Crashes)**
  * *Mitigation:* Conduct offline evaluations on pre-collected multi-view visual frames from `DriveLM-nuScenes`, avoiding weeks spent debugging physics engine synchronization issues.
* **Risk 2: Heavy GPU Memory Footprint during VLM Fine-Tuning**
  * *Mitigation:* Freeze the main vision encoder and language backbones. Inject LoRA adapters (Rank=16, Alpha=32) into cross-attention layers (`q_proj`, `v_proj`) to train <1% of model parameters on a single 24 GB VRAM GPU.
* **Risk 3: Lack of Explainability in Waypoint Generation**
  * *Mitigation:* Force the model to output step-by-step Graph VQA reasoning chains prior to generating final numerical waypoint coordinates.

---

## Technical Roadmap & Milestones

- [ ] **Month 1:** Set up `DriveLM-nuScenes` visual frames; evaluate zero-shot baselines for Qwen2-VL and LLaVA-NeXT on 50 test cases.
- [ ] **Month 2:** Implement LoRA fine-tuning and joint cross-entropy + smooth $L_1$ loss function; train adapter modules.
- [ ] **Month 3:** Profile latency under FP16/INT8 quantization; construct interactive Gradio Web UI demo for final presentation.
