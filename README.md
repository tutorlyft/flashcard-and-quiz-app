# AI Study Tool

## Description
AI Study Tool is a Streamlit-based application that uses artificial intelligence to generate flashcards and quizzes from various input sources. It's designed to help students and learners create effective study materials quickly and easily.

## Features
- Generate flashcards from text, documents, or video transcripts
- Create multiple-choice quizzes based on input content
- Support for various input types (text, PDF, Word documents, images, YouTube videos)
- LaTeX support for mathematical expressions
- Interactive user interface for reviewing flashcards and taking quizzes

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-study-tool.git
   cd ai-study-tool
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Follow the on-screen instructions to:
   - Upload a document or input text
   - Choose between flashcard or quiz generation
   - Review your generated study materials

## Contributing
Contributions to the AI Study Tool are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- OpenAI for providing the GPT model used in this project
- Streamlit for the web application framework
