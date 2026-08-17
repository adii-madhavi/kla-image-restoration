# Research_and_Resources.md
# KLA Research, External Data, Pretrained Models and Citation Discipline

## 0. Purpose

This file controls external research and resource use.

The KLA guidance explicitly permits public external image datasets and publicly available pretrained model weights when the license permits competition use. Every such resource must be disclosed.

## 1. KLA-shared references

1. Kumar, T. et al. (2024), `Image Data Augmentation Approaches: A Comprehensive Survey and Future Directions`, IEEE Access, Volume 12.
2. Zhai, L. et al. (2023), `A Comprehensive Review of Deep Learning-Based Real-World Image Restoration`, IEEE Access, 11, 21049-21067.
3. Terven, J. et al. (2025), `A Comprehensive Survey of Loss Functions and Metrics in Deep Learning`, Artificial Intelligence Review, 58, 195.
4. Monga, V. et al. (2021), `Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing`, IEEE Signal Processing Magazine, 38(2), 18-44.

## 2. Research questions

### Data

* Which augmentation improves OOD generalization?
* Which synthetic degradation pipeline best matches the official training distribution?
* Does external data pretraining help or hurt?

### Model

* CNN versus transformer
* compact versus high capacity
* spatial versus spatial plus frequency
* generalized restoration versus degradation-specific priors

### Loss

* L1/L2 baseline
* structural loss
* frequency-domain loss
* perceptual loss
* composite objective

### Systems

* batch size
* AMP
* `torch.compile`
* loader optimization
* host/device transfers
* output writing

## 3. External dataset policy

Allowed uses include:

* pretraining
* domain adaptation
* learned restoration priors

Every dataset must be recorded with:

```text
Name
URL
License
Version/access date
Paper
Dataset card
Role
Preprocessing
```

## 4. Pretrained model policy

Potential public ecosystems discussed by KLA include:

* Hugging Face
* PyTorch
* Torch Hub
* timm
* TensorFlow Model Zoo

Record:

```text
Model name
Source URL
License
Version
Paper
Model card
Input assumptions
Fine-tuning method
```

## 5. Novelty discipline

Do not call a method novel merely because it combines known components.

Before making a novelty claim:

1. search the literature
2. search official repositories
3. inspect related implementations
4. document what is actually different

Distinguish a novel research contribution from an effective engineering combination.

## 6. Citation discipline

Every external factual claim used in the presentation or README should be traceable to an authoritative paper, official repository, dataset card or model card.

Do not submit unverified AI-generated references.

## 7. Drift Sense boundary

The Applied Materials Drift Sense material is not a KLA solution source.

Its transferable value is methodological:

* understand image formation
* use controlled synthetic experiments
* study noise
* think about difficult cases
* measure failure
* care about throughput

Do not import its task requirements such as localization, coordinate output, 100x/10x matching, closest-to-center selection or its separate scoring rubric.

## 8. Research log

Maintain:

```text
question
source
finding
experiment
decision
```

This is especially important for architecture, loss and external resource decisions.
