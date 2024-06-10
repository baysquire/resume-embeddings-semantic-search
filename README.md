## README.md

# Resume Embeddings and Semantic Search

This repository provides an end-to-end pipeline for generating synthetic resume data, embedding the resumes using a Large Language Model (LLM), and performing semantic search to find resumes similar to a given query. The pipeline includes data generation, embedding with Sentence Transformers, dimensionality reduction with PCA, and visualization using Matplotlib.

## Table of Contents

- [Introduction](#introduction)
- [Setup and Installation](#setup-and-installation)
- [Data Generation](#data-generation)
- [Embedding and Visualization](#embedding-and-visualization)
- [Semantic Search](#semantic-search)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the use of AI and machine learning techniques to process, embed, and search text data. The pipeline includes:

1. **Data Generation**: Using the `Faker` library to generate synthetic resume data.
2. **Embedding**: Utilizing the `SentenceTransformer` model to generate embeddings for resumes.
3. **Dimensionality Reduction**: Applying PCA to reduce the dimensionality of the embeddings for visualization.
4. **Semantic Search**: Performing similarity search to find resumes that best match a given query.

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Installation

1. Clone the repository:

```sh
git clone https://github.com/baysquire/resume-embeddings-semantic-search.git
cd resume-embeddings-semantic-search
```

2. Create a virtual environment and activate it:

```sh
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install the required packages:

```sh
pip install -r requirements.txt
```

### Required Libraries

The following Python libraries are required for this project:

- numpy
- pandas
- faker
- sentence-transformers
- scikit-learn
- matplotlib

## Data Generation

The `data_generation.py` script uses the `Faker` library to create synthetic resume data. This data is saved as a CSV file.

## Embedding and Visualization

The `embedding_visualization.py` script loads the synthetic data, generates embeddings using a Sentence Transformer model, and applies PCA for visualization.

## Semantic Search

The `semantic_search.py` script demonstrates how to perform semantic search to find the most relevant resumes based on a query.

## Usage

1. Generate synthetic data by running `data_generation.py`.
2. Generate embeddings and visualize them using `embedding_visualization.py`.
3. Perform semantic search using `semantic_search.py`.

## Examples

### Generate Synthetic Data

Run the `data_generation.py` script to create a dataset of synthetic resumes.

### Generate Embeddings and Visualize

Run the `embedding_visualization.py` script to create embeddings from the resumes and visualize them using PCA.

### Perform Semantic Search

Run the `semantic_search.py` script to find the resumes that are most similar to a given query.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss improvements or suggest features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.