import os
import base64
import json
from collections import OrderedDict
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from langchain.schema import SystemMessage
from utils import (
    ArticleSummaryAgent,
    ComparativeAnalysisAgent,
    get_company_articles
)

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Expects a JSON payload: {"company": "Tesla"}
    Runs the ArticleSummaryAgent and ComparativeAnalysisAgent,
    then returns a response in the specified JSON format.
    
    Expected output format:
    {
      "Company": "Tesla",
      "Articles": [
         {
           "Title": "Tesla's New Model Breaks Sales Records",
           "Summary": "Tesla's latest EV sees record sales in Q3...",
           "Sentiment": "Positive",
           "Topics": ["Electric Vehicles", "Stock Market", "Innovation"]
         },
         ...
      ],
      "Comparative Sentiment Score": { ... },
      "Final Sentiment Analysis": "Tesla’s latest news coverage is mostly positive. Potential stock growth expected.",
      "Audio": "[URL to the audio file or Hindi summary text]",
      "Audio Message": "Audio generated successfully" 
                         OR "Audio generation failed; using Hindi summary text" 
    }
    """
    data = request.get_json()
    company = data.get("company", "").strip()
    if not company:
        return jsonify({"error": "Company name is required"}), 400

    # --- Run Article Summary Agent ---
    try:
        summary_agent = ArticleSummaryAgent([get_company_articles], company)
        initial_state = {"message": [SystemMessage(content="Start")]}
        summary_agent.graph.invoke(initial_state)
    except Exception as e:
        return jsonify({"error": f"Exception in ArticleSummaryAgent: {str(e)}"}), 500

    # Build the articles list from the summary agent's output
    articles = []
    for key, article in summary_agent.article_data.items():
        articles.append(OrderedDict([
            ("Title", article.Title),
            ("Summary", article.Summary),
            ("Sentiment", article.Sentiment),
            ("Topics", list(article.Topics))
        ]))

    # --- Run Comparative Analysis Agent ---
    try:
        comp_agent = ComparativeAnalysisAgent(summary_agent.article_data)
        comp_initial_state = {
            "message": [SystemMessage(content="Start")],
            "status_code": [200]
        }
        comp_output = comp_agent.graph.invoke(comp_initial_state)
        
        # If a 503 error occurs, retry once
        if comp_output.get("status_code", [500])[0] == 503:
            comp_agent = ComparativeAnalysisAgent(summary_agent.article_data)
            comp_output = comp_agent.graph.invoke(comp_initial_state)
            status = comp_output.get("status_code", [500])[-1]
            print("status is ", status)
            if status != 200:
                error_msg = (comp_output["message"][-1].content 
                             if comp_output.get("message") else "Hugging Face error on retry")
                return jsonify({"error": f"Hugging Face Error after retry: {error_msg}"}), 500

        elif comp_output.get("status_code", [500])[0] == 500:
            error_msg = (comp_output["message"][-1].content 
                         if comp_output.get("message") else "Unknown internal error")
            return jsonify({"error": f"Internal Server Error: {error_msg}"}), 500

    except Exception as e:
        return jsonify({"error": f"Exception in ComparativeAnalysisAgent: {str(e)}"}), 500

    # --- Prepare audio in static folder if available ---
    audio_filename = f"{company.lower().replace(' ', '_')}_hindi_audio.mp3"
    audio_dir_path = os.path.join("static", "audio")
    if not os.path.exists(audio_dir_path):
        os.makedirs(audio_dir_path, exist_ok=True)
    audio_path = os.path.join(audio_dir_path, audio_filename)
    
    if comp_agent.hindi_audio:
        # Save the audio file and generate a URL
        with open(audio_path, "wb") as audio_file:
            audio_file.write(comp_agent.hindi_audio)
        audio_url = f"{request.host_url}static/audio/{audio_filename}"
        audio_message = "Audio generated successfully"
    else:
        # If audio generation fails, use the Hindi text summary
        audio_url = comp_agent.hindi_summary or "No audio available"
        audio_message = "Audio generation failed; using Hindi summary text"
    
    # --- Build final ordered response ---
    final_response = OrderedDict([
        ("Company", company),
        ("Articles", articles),
        ("Comparative Sentiment Score", comp_agent.compartive_data.get("Comparative Sentiment Score", {})),
        ("Final Sentiment Analysis", comp_agent.final_analysis),
        ("Audio", audio_url),
        ("Audio Message", audio_message)
    ])

    response_json = json.dumps(final_response, ensure_ascii=False, indent=4)
    return Response(response_json, mimetype="application/json")


if __name__ == "__main__":
    # Bind to 0.0.0.0 so that the app is accessible externally (e.g., in Docker)
    app.run(host="0.0.0.0", port=5000, debug=True)
