import streamlit as st
import requests
import json

# --- App Title ---
st.set_page_config(page_title="Company News Sentiment Analyzer", page_icon="📰")

# Inject CSS to enforce a uniform font across the app
st.markdown(
    """
    <style>
    /* Import a custom font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* Apply the custom font to the whole page */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# The rest of your Streamlit code
st.title("Company News Sentiment Analyzer")

# --- Input from user ---
company_name = st.text_input("Enter Company Name:")

if st.button("Analyze"):
    if not company_name.strip():
        st.error("Please enter a company name.")
    else:
        with st.spinner("Analyzing... Please wait."):
            try:
                # Call Flask backend (adjust the URL as needed, here assuming local testing)
                response = requests.post("http://0.0.0.0:5000/analyze", json={"company": company_name.strip()})
                
                # If response code is not 200, show the error message
                if response.status_code != 200:
                    error = response.json().get("error", "Unknown error occurred.")
                    st.error(f"Error: {error}")
                else:
                    result = response.json()
                    st.success("Analysis Completed!")
                    st.header(f"Company: {result['Company']}")

                    # Show articles
                    st.subheader("Articles")
                    for idx, article in enumerate(result["Articles"], start=1):
                        st.markdown(f"**{idx}. {article['Title']}**")
                        st.write(f"Summary: {article['Summary']}")
                        st.write(f"Sentiment: {article['Sentiment']}")
                        st.write(f"Topics: {', '.join(article['Topics'])}")
                        st.markdown("---")

                    # Comparative Sentiment Score
                    st.subheader("Comparative Sentiment Score")
                    comparative = result["Comparative Sentiment Score"]
                    st.write("### Sentiment Distribution")
                    st.json(comparative.get("Sentiment Distribution", {}))

                    st.write("### Coverage Differences")
                    for diff in comparative.get("Coverage Differences", []):
                        st.write(f"**Comparison:** {diff['Comparison']}")
                        st.write(f"Impact: {diff['Impact']}")
                        st.markdown("---")

                    st.write("### Topic Overlap")
                    st.json(comparative.get("Topic Overlap", {}))

                    # Final Sentiment Analysis
                    st.subheader("Final Sentiment Analysis")
                    st.write(result["Final Sentiment Analysis"])

                    # Display Audio Message
                    if result.get("Audio Message"):
                        st.info(result["Audio Message"])

                    # Audio playback (if available)
                    if result.get("Audio",""):
                        st.subheader("Hindi Audio Summary")
                        st.audio(result["Audio"], format="audio/mp3")
                    
                    if result.get("Hindi Summary"):
                        st.subheader("Sentiment Analysis in HIndi")
                        st.write(result["Hindi Summary"])                      


            except Exception as e:
                st.error(f"An exception occurred: {str(e)}")
