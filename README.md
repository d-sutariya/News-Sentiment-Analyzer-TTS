---
title: "News Sentiment Analyzer TTS"
emoji: "📰"
colorFrom: "blue"
colorTo: "green"
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
app_port: 7860
---

# News Sentiment Analyzer TTS

This project is a web-based application that extracts news articles about a given company, performs summarization, sentiment analysis, and comparative analysis, and generates a text-to-speech (TTS) output in Hindi. The final output is returned as a structured JSON response containing article details, comparative sentiment score, final sentiment analysis, and audio (or a fallback Hindi text summary).

## Project Setup

### Prerequisites

-   **Docker Desktop with WSL 2 Integration** (for local testing on Windows)
-   **Git** for version control
-   **Python 3.9+** (if running locally without Docker)
-   Configure environment variables for API keys via a `.env` file (for local testing) or through Hugging Face Space Secrets.

### Installation and Running Locally

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/yourusername/News-Sentiment-Analyzer-TTS.git](https://github.com/yourusername/News-Sentiment-Analyzer-TTS.git)
    cd News-Sentiment-Analyzer-TTS
    ```

2.  **Set Up Environment Variables:** Create a `.env` file with necessary keys:

    ```
    GEMINI_API_KEY=your_gemini_api_key
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
    TAVILY_API_KEY = your_tavily_api_token
    ```

3.  **Build and Run Using Docker:** Make sure your `Dockerfile` and `supervisord.conf` are in the repository root.

    ```bash
    docker build -t news-sentiment-analyzer-tts .
    ```

4.  **Run the container:**

    ```bash
    docker run -p 7860:7860 news-sentiment-analyzer-tts
    ```

5.  **Access the Application**:

    ```
    Frontend: http://localhost:7860
    Backend API (if needed): http://localhost:8080/analyze
    ```

### Model Details

-   **Summarization and Sentiment Analysis:**

    -   Utilizes custom AI agents defined in `utils.py` that leverage language models (e.g., Gemini models) to generate article summaries, sentiment labels, and comparative analysis.

-   **Text-to-Speech (TTS):**

    -   Uses a Hugging Face model (e.g., `facebook/mms-tts-hin`) to generate Hindi audio from text. If the TTS model fails, the Hindi text summary is used as a fallback.

## API Development

### Backend API

The backend API is built using Flask and exposes the following endpoint:

-   **`POST /analyze`**

    -   **Description:**
        -   Accepts a JSON payload with a company name and returns a structured JSON response containing:
            -   Company name
            -   List of articles with Title, Summary, Sentiment, and Topics
            -   Comparative Sentiment Score (including sentiment distribution, coverage differences, and topic overlap)
            -   Final Sentiment Analysis
            -   Audio (URL to generated audio or fallback Hindi text summary)
            -   Audio Message (indicates if audio was generated or fallback text is used)

### Accessing the API

You can test the API using Postman or curl:

-   **Postman:**

    -   Set up a `POST` request to `http://localhost:8080/analyze` with a JSON body:

    ```json
    {
      "company": "Tesla"
    }
    ```

-   **curl:**

    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"company": "Tesla"}' http://localhost:8080/analyze
    ```

### API Usage

-   **Third-Party APIs:**

    -   The application uses Hugging Face's Inference API for TTS and potentially other inference tasks.
    -   External libraries such as `requests` and `langchain` are used for API calls and model management.

-   **Integration Details:**

    -   Environment secrets (e.g., API keys) are injected at runtime via Hugging Face Space Secrets.
    -   The Docker container runs both the Flask backend and the Streamlit frontend using Supervisor (or a startup script) to manage multiple processes in a single container.

## Assumptions & Limitations

-   **Assumptions:**

    -   The provided company name is valid and relevant news articles can be found.
    -   The AI agents in `utils.py` generate structured outputs reliably.
    -   In case of TTS failure, a Hindi text summary is available as a fallback.

-   **Limitations:**

    -   The performance of summarization, sentiment analysis, and TTS depends on the underlying AI models.
    -   Rate limits on Hugging Face Inference API or other providers may affect performance.
    -   Running both frontend and backend in a single container simplifies deployment but might limit scalability compared to a multi-container architecture.