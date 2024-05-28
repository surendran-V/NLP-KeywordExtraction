# Keyword Extraction from Scientific Articles

This project implements a keyword extraction system for scientific articles, leveraging Natural Language Processing (NLP) techniques and graph-based methods. The system identifies significant keywords from abstracts to facilitate efficient information retrieval and analysis.

## Project Structure

- `keywordExtractor.py`: Contains the `KeywordExtractor` class, responsible for processing the text, constructing the co-occurrence graph, and extracting keywords.
- `lexicons.py`: Defines lexicons, including stopwords and multi-word expressions (MWEs) used in tokenization and preprocessing.
- `nlp_utils.py`: Provides utility functions for text preprocessing, tokenization, lemmatization, and co-occurrence calculations.

## Files Description

### `keywordExtractor.py`

This script defines the `KeywordExtractor` class, which includes:

- **Initialization**: Processes the input text, tokenizes it, and initializes a co-occurrence graph.
- **Graph Initialization**: Constructs a graph with tokens as nodes and co-occurrence relations as edges.
- **Embedding Weights**: Reweights the graph using word embeddings, enhancing the co-occurrence relationships.
- **Node Ordering**: Orders nodes based on various graph centrality measures to identify significant keywords.
- **Graph Visualization**: Visualizes the graph to illustrate the relationships between keywords.

### `lexicons.py`

This script defines:

- **Stopwords**: A list of common stopwords to be ignored during keyword extraction.
- **Multi-Word Expressions (MWEs)**: Common MWEs that should be treated as single tokens during processing.

### `nlp_utils.py`

This script provides utility functions for:

- **Text Pruning**: Preprocesses the input text, including tokenization, MWE handling, and stopword removal.
- **Co-Occurrence Calculation**: Computes co-occurrence representations for tokens within a sliding window.
- **Word Embeddings**: Retrieves word embeddings for tokens and checks their presence in the model.

## Installation

To run this project, ensure you have the following dependencies installed:

- `nltk`
- `networkx`
- `matplotlib`
- `sklearn`
- `gensim`

You can install these dependencies using pip:

```bash
pip install nltk networkx matplotlib sklearn gensim
```
## Acknowledgments
This project utilizes concepts from graph theory and NLP to extract meaningful keywords from scientific texts. Special thanks to the developers of the nltk, networkx, and gensim libraries.
