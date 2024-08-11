# SmartSearch

[中文版 README](README CN.md)

SmartSearch is an advanced AI-powered search system that breaks down complex queries into manageable sub-questions, searches for information, and synthesizes comprehensive answers.

## Features

- Complex query decomposition
- Intelligent web searching
- Dynamic information synthesis
- Multi-agent collaboration

## How It Works

1. Query Decomposition: The Planner agent breaks down the main query into sub-questions.
2. Web Search: The Search agent performs web searches for each sub-question.
3. Information Analysis: The Reflection agent analyzes search results and determines if more information is needed.
4. Answer Synthesis: Once sufficient information is gathered, a final comprehensive answer is generated.

## Installation

1. Clone the repository:
git clone https://github.com/531619091/smart_search.git
2. Install the required packages:
pip install -r requirements.txt
3. Set up your OpenAI API key as an environment variable:
export OPENAI_API_KEY='your-api-key-here'

## Usage

Run the main script:
python smart_search.py

## Configuration

Adjust the `max_turn` parameter in the `MindSearch` class to control the maximum number of search iterations.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.