<h1 align="center">HarmProfile: Characterizing Harmful Distributions in Frontier LLMs</h1>

<p align="center">
  <b>Zhouyuan Ma<sup>1</sup>, Yutao Wu<sup>2</sup>, Hanxun Huang<sup>3</sup>, Xiang Zheng<sup>4</sup>, Xiao Liu<sup>2</sup>, Yixin Cao<sup>1</sup>, Zuxuan Wu<sup>1</sup>, Xingjun Ma<sup>1†</sup>, Yu-Gang Jiang<sup>1†</sup></b>
</p>

<p align="center">
  <sup>1</sup>Fudan University, <sup>2</sup>Deakin University, <sup>3</sup>The University of Melbourne, <sup>4</sup>City University of Hong Kong
</p>

<p align="center">
  <a href="https://fresh-ma.github.io/HarmProfile/"><img src="https://img.shields.io/badge/Website-HarmProfile-blue" alt="Website"></a>
  <a href="https://arxiv.org/abs/2608.14577"><img src="https://img.shields.io/badge/ArXiv-2608.14577-b31b1b" alt="arXiv"></a> 
</p>

<p align="center">
  🌐 <a href="https://fresh-ma.github.io/HarmProfile/">Website</a> |
  📄 <a href="https://arxiv.org/abs/2608.14577">Paper</a>
</p>

## Abstract

Frontier large language models (LLMs) safety evaluation has largely treated harmful generation as an attack outcome rather than an object of analysis. Consequently, little is known about the outputs themselves, partly because large-scale, high-quality collections of frontier-model misbehavior are difficult to obtain. To address this gap, we introduce **HarmProfile**, a content-centric benchmark dataset that collects model misbehavior across diverse harm categories and model families, and defines the resulting content distribution as a model-level risk profile. The premise is that, just as linguistic behavior can be characterized from an utterance corpus, model risk can be characterized from the content, severity, and variation of its safety failures. HarmProfile contains over 80,000 validated artifacts from 23 frontier LLMs across 13 model families, organized into 15 harm categories and 57 subcategories. Using this corpus, we find that frontier LLMs reliably produce harmful content at scale, yet exhibit distinct risk profiles; both harmfulness and diversity grow with model capability, suggesting that frontier LLMs may appear safe yet harbor increasingly dangerous knowledge beneath the alignment surface.

> **Content Warning:** This paper contains examples of harmful content.
