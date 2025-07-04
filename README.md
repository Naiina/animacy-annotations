# The Cross-linguistic Role of Animacy in Grammar Structures

This repository contains the code for the paper **"The Cross-linguistic Role of Animacy in Grammar Structures"** (ACL Main 2025).  
## Abstract 
Animacy is a semantic feature of nominals and follows a hierarchy: personal pronouns > human > animate > inanimate. In several languages, animacy imposes hard constraints on grammar. While it has been argued that these constraints may emerge from universal soft tendencies, it has been difficult to provide empirical evidence for this conjecture due to the lack of data annotated with animacy classes. In this work, we first propose a method to reliably classify animacy classes of nominals in 11 languages from 5 families, leveraging multilingual large language models (LLMs) and word sense disambiguation datasets. Then, through this newly acquired data, we verify that animacy displays consistent cross-linguistic tendencies in terms of preferred morphosyntactic constructions, although not always in line with received wisdom: animacy in nouns correlates with the alignment role of agent, early positions in a clause, and syntactic pivot (e.g., for relativisation), but not necessarily with grammatical subjecthood.
Furthermore, the behaviour of personal pronouns in the hierarchy is idiosyncratic as they are rarely plural and relativised, contrary to high-animacy nouns.

## Models and Datasets on Hugging Face

- [**Multilingual BERT**, task-specific SFT](https://huggingface.co/lingvenvist/mbert-animacy)
- [**Aya Expanse 8B**, LoRA adapter](https://huggingface.co/lingvenvist/mbert-animacy)
- [**Animacy-annotated datasets** for all languages](https://huggingface.co/lingvenvist), derived from [XL-WSD](https://sapienzanlp.github.io/xl-wsd) (Pasini et al., 2021)
