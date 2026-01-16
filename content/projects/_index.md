---
title: "Projects"
---

A curated list of open-source packages I develop and maintain.

**Packages**
- [Model2Vec](#model2vec)
- [SemHash](#semhash)
- [Pyversity](#pyversity)
- [Vicinity](#vicinity)
- [Tokenlearn](#tokenlearn)
- [Model2Vec-rs](#model2vec-rs)

---

## Model2Vec

State-of-the-art static embedding models distilled from sentence transformers, designed for extremely fast CPU inference. Model2Vec produces tiny models (as small as 4MB) that deliver high-quality text embeddings, processing thousands of texts per second.

- GitHub: https://github.com/MinishLab/model2vec  
- Docs: https://minish.ai/packages/model2vec/introduction

---

## SemHash

A semantic deduplication and filtering library for text datasets based on embedding similarity. SemHash can be used to remove near-duplicates, detect overlap between splits, or clean large corpora before training.

- GitHub: https://github.com/MinishLab/semhash  
- Docs: https://minish.ai/packages/semhash/introduction

---

## Pyversity

A fast, lightweight library for diversifying retrieval results using classical diversification strategies. Pyversity provides a unified API for methods such as MMR, MSD, DPP, COVER, and SSD, with NumPy as its only dependency.

- GitHub: https://github.com/Pringled/pyversity  
- Docs: https://github.com/Pringled/pyversity

---

## Vicinity

A unified nearest-neighbor search interface that provides a consistent API over multiple vector search backends. Vicinity is designed to make it easy to switch or compare ANN implementations without changing application code.

- GitHub: https://github.com/MinishLab/vicinity  
- Docs: https://minish.ai/packages/vicinity/introduction

---

## Tokenlearn

A pre-training method and tooling for learning compact static embeddings used in distillation pipelines. Tokenlearn focuses on efficiently learning token-level representations that transfer well to downstream static models.

- GitHub: https://github.com/MinishLab/tokenlearn  
- Docs: https://minish.ai/packages/overview#tokenlearn

---

## Model2Vec-rs

A Rust implementation of Model2Vec for performance-critical and native Rust use cases. This package mirrors the Python version while emphasizing speed, memory efficiency, and Rust ecosystem integration.

- GitHub: https://github.com/MinishLab/model2vec-rs  
- Docs: https://minish.ai/packages/overview#model2vec-rs

---