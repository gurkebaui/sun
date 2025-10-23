# SUN — Simple Network for a Minimal Language Model

SUN is a small proof-of-concept that explores building a minimal, interpretable language model by composing a network of information sources and influence factors. The aim is educational: to experiment with how small, inspectable components can be combined to produce language-like behavior, not to compete with large production models.

## Concept

The project models a language system as a directed, weighted graph of:
- information nodes: corpora, token statistics, lexical resources, heuristics;
- influence edges: weights and transformation rules that modulate signals between nodes;
- aggregation and decoding components that combine signals and produce tokens.

By keeping components small and explicit, SUN makes it easier to reason about which inputs contribute to a decision, and to test simple alternatives to black-box models.

## Features

- Lightweight Python implementation for constructing information-influence networks.
- Configurable node and edge types (statistical, rule-based, heuristic).
- Simple token-level scoring and decoding (greedy / gated aggregation).
- Example datasets and notebooks for exploration.

## Repository layout

- src/ — core implementation (network, node classes, aggregation, decoders)
- data/ — tiny example datasets and token-frequency files
- examples/ — runnable scripts demonstrating basic usage
- notebooks/ — interactive experiments and visualizations
- tests/ — small unit tests (if present)
- README.md — this file

Adjust the structure to match the repository if files are organized differently.

## Installation

A small Python environment is sufficient.

1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate    # Windows

2. Install dependencies (this may not be up to date):

   pip install -r requirements.txt

If the requirements.txt is not up to date, install the basics for experimenting:

   pip install numpy networkx jupyter

## Quick start

Run an example script (adjust path to match repository):

   python examples/run_simple_network.py

Or start a notebook to explore interactively:

   jupyter notebook notebooks/demo.ipynb

Minimal conceptual steps:
1. Define nodes and edges (via config or code).
2. Load small token statistics or datasets into nodes.
3. Propagate and aggregate signals across the network to score tokens.
4. Decode tokens into short sequences using greedy or simple beam search.

## Data format

Nodes accept small, human-readable inputs such as JSON, CSV, or plain text token-frequency maps. Keep datasets tiny for fast iteration.

Example token-frequency JSON:

{  
  "the": 5000,
  "cat": 200,
  "sat": 150
}

## Limitations

- Educational toy project — not production-ready.
- Simplified components yield poor performance compared to modern LLMs.
- Limited evaluation tooling: add tests and metrics for experiments.

## Contributing

Contributions welcome. Good ways to help:
- Add examples and small datasets.
- Improve documentation and notebooks.
- Implement new node/edge types and aggregation strategies.
- Add tests and CI for reproducible experiments.

When contributing, please open a small, focused pull request describing changes.

## License

If you want to allow permissive reuse, add an MIT license file (MIT recommended). If no license exists, consider adding one before broad reuse.

## Contact

Owner: @gurkebaui
