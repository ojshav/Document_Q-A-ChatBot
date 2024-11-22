# Document Q&A Bot

## Overview

The Document Q&A Bot is a Streamlit application that allows users to upload PDF documents and ask questions based on the content of those documents. The application utilizes various libraries from Langchain to process the documents and generate responses to user queries.

## Features

- Upload multiple PDF documents.
- Ask questions based on the uploaded documents.
- View chat history of user queries and bot responses.
- Clear session to reset the application state.

## Requirements

To run this application, you need the following:

- Python 3.7 or higher
- Streamlit
- Langchain libraries
- Ollama (for embeddings)
- FAISS (for vector store)
- Other dependencies as specified in the `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/document-qa-bot.git
   cd document-qa-bot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```


 

3. **Install Ollama**:
   Ensure that you have Ollama installed on your system. You can find installation instructions on the [Ollama website](https://ollama.com/).

4. **Download the LLM Model**:
   Make sure to have the Llama3.2 model available for use in your application.

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. **Upload PDF Documents**:
   - Use the file uploader to upload one or more PDF documents.

4. **Ask Questions**:
   - Type your question in the input box and click the "Send" button.
   - The bot will respond based on the content of the uploaded documents.

5. **View Chat History**:
   - The chat history will display all user queries and bot responses.

6. **Clear Session**:
   - Click the "Clear Session" button to reset the application state.

## Code Structure

- `app.py`: Main application file containing the Streamlit interface and logic for document processing and question answering.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Langchain](https://langchain.com/)
- [Ollama](https://ollama.com/)

![WhatsApp Image 2024-11-22 at 15 08 06_66a5c6d2](https://github.com/user-attachments/assets/d807face-5d9a-4d07-82e9-76b165023b43)

![WhatsApp Image 2024-11-22 at 15 08 21_c538a817](https://github.com/user-attachments/assets/fd880a05-74ea-465d-817d-a209d4a2e515)
![WhatsApp Image 2024-11-22 at 15 08 33_2076abe5](https://github.com/user-attachments/assets/1eec178c-7560-4fab-ae33-ae4a1ebc22d5)


