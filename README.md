# AIRL-Assignment

# Vision Transformer on CIFAR-10 (PyTorch)
This report summarizes experiments on training a Vision Transformer (ViT) model from scratch on the CIFAR-10 dataset. All accuracies reported are **best accuracies achieved on the full test set (10k examples)**.

---

## General Settings

- Number of dimensions for **MLP heads** (`mlp_dim`) was always set to **2 × feature_dims**.  
- Unless stated otherwise, other hyperparameters and optimizer settings remained constant.

---

## Baseline Configuration

- Depth: `6`  
- Feature dimensions: `256`  
- Learning rate: `3e-4`  
- Epochs: `100`  
- Scheduler: `20`-epoch linear warmup with cosine decay  
- Optimizer: **AdamW** with weight decay = `0.3` (other hyperparameters kept default)  

**Augmentation:**  
- RandomFlip, RandomCrop  
- **Accuracy:** `72.43%`  
- Observation: Clear signs of **overfitting**.

---

## Heavy Augmentation

- Augmentations: **MixUp** and **CutMix**  
- Model config: same as baseline  

**Results:**  
- Train accuracy: ~`57%`  
- Test accuracy: `67.51%`  

**Analysis:**  
- Heavy regularization made the model struggle on training data.  
- However, since test data has no augmentations, the gap shows improved **generalization** despite lower train accuracy.  

---

## Scaling Depth

Keeping other parameters same, depth was varied:

| Depth | Accuracy |
|-------|-----------|
| 8     | 68.14%    |
| 10    | 69.27%    |
| 12    | 69.45%    |

**Insights:**  
- Accuracy increases slightly with depth but shows **diminishing returns**.  
- Likely limited by choice of learning rate and insufficient training.  

---

## Choosing Depth = 8

- **Why 8?** Faster iteration compared to depth 12, while maintaining competitive accuracy.  

### Longer Training with Depth = 8

- Epochs: `200`  
- Warmup: `30` epochs with cosine decay  
- Tested different learning rates:  

| Learning Rate | Accuracy |
|---------------|-----------|
| 3e-4          | 72.73%    |
| 6e-4          | 78.58%    |
| 1e-3          | 77.44%    |

**Decision:**  
- Chose **6e-4** as optimal learning rate.  
- `1e-3` was unstable, causing noisy training curves.  

---

## Overlapping Patches (Shifted Patch Tokenization)

- Applied **Shifted Patch Tokenization (SPT)** for overlapping patches.  
- Other hyperparameters same as before.  

**Result:** `79.12%` accuracy.  

---

## Feature Dimension Scaling (after adopting SPT)

Since SPT improved accuracy, it was kept for further experiments. Feature dimensions were varied:

| Feature Dimensions | Accuracy |
|---------------------|-----------|
| 384                 | 82.0%     |
| 512                 | 86.0%     |
| 768                 | 87.0%     |

**Decision:**  
- Adopted **768 feature dimensions** as the best-performing configuration.  

---

## Key Takeaways

1. **Baseline ViT (depth=6, dim=256)** achieved `72.43%` but overfitted.  
2. **Heavy augmentation** improved generalization but reduced raw accuracy.  
3. **Scaling depth** helped slightly, but tradeoff with training efficiency.  
4. **Depth=8 + 200 epochs + lr=6e-4** gave the best balance, achieving ~`78.58%`.  
5. **Shifted Patch Tokenization (SPT)** boosted accuracy further to `79.12%`.  
6. **Feature dimension scaling with SPT** led to a maximum of `87%` at `768` dimensions.  

---

# Text-Driven Image Segmentation with SAM 2

The repository also includes a notebook demonstrating **text-driven segmentation** using **Segment Anything Model 2 (SAM 2.1)**.

---

## Workflow

1. **GroundingDINO** is first used to detect bounding boxes around the required object using the input text prompt.  
2. These bounding boxes are passed to **SAM 2.1**, which then generates the segmentation masks.  
3. **Rendering:** Custom functions (using OpenCV) overlay the predicted masks directly on the original image for visualization.  

---

## Video Segmentation

- For videos, a **HuggingFace Sam2VideoProccessor** is used.  
- This processor leverages **SAM 2's efficient memory module for videos**, enabling consistent mask tracking across frames.  

---

## Limitations

- Video processing length and resolution are **limited by available VRAM**.  
- Longer or higher-resolution videos require more GPU memory and may not run on constrained hardware.  

---
