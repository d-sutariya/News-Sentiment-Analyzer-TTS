import os
import operator
import ast
import random
import requests
import json 
import boto3
from typing import Sequence, Annotated, TypedDict, List, Literal, Set

from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage, ToolMessage
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
NUMBER_OF_ARTICLE_PER_SEARCH = 10
NUMBER_OF_UNIQUE_ARTICLES = 20

with open("/run/secrets/backend_secrets","r") as f:
    secrets = json.load(f)

@tool
def get_company_articles(search_queries: List[str]):
    """
    This tool takes company name as argument and search in the web using TavilySearch 
    And Scrap First 20 articles of the given company name using newspaper3k python library.
    
    Args:
        search_quiries (List(str)) : Different Unique Search quiries to get unique news articles 
                                        about company provided by user.
    Return:
        dict: A dictionary containing:
            - Article_i (key) : A dictionary containing:
                - content (key): content of the news article   
    example: 
        {
            'Article_1':{
                            'content':'content of the Article_1'
                        },
            'Article_2':{
                            'content':'content of the Article_2'
                        }
        }
    """
    logger.info("Inside the company article function")
    search_tool = TavilySearchResults(
        max_results=NUMBER_OF_ARTICLE_PER_SEARCH,
        search_depth="advanced",
        topic="news",
        tavily_api_key = secrets["TAVILY_API_KEY"]
    )
    news_articles = []
    for query in search_queries:
        try:
            articles = search_tool.invoke({"query": query})
            news_articles += articles
        except Exception as e:
            logger.error("Error during search for query %s: %s", query, e)
    # Shuffle the articles for uniqueness
    random.shuffle(news_articles)
    
    structured_articles = {}
    for i, article in enumerate(news_articles):
        try:
            # Extract the content from each article
            temp = {"content": article["content"]}
            structured_articles[f"Article_{i}"] = temp
        except KeyError as e:
            logger.error("Article missing 'content' field: %s", e)
    
    return structured_articles


class ArticleData(BaseModel):
    """Stores relevant data for an article."""
    Title: str = Field(description="Relevant title of the given article")
    Summary: str = Field(description="Relevant summary of the given article")
    Sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Relevant sentiment of the given article"
    )
    Topics: Set[str] = Field(description="Relevant topics of the given article")


# Agent state is defined as a TypedDict.
class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage], operator.add] = Field(
        description="State of the Agent and previous history"
    )


class ArticleSummaryAgent:
    def __init__(self, tools, company_name):
        self.tools = {tool.name: tool for tool in tools}
        # Initialize two Gemini LLMs with different models.
        self.gemini_flash_llm = ChatGoogleGenerativeAI(
            api_key=secrets["GEMINI_API_KEY"],
            model="models/gemini-1.5-flash"
        )
        self.gemini_flash_2_llm = ChatGoogleGenerativeAI(
            api_key=secrets["GEMINI_API_KEY"],
            model="models/gemini-2.0-flash-lite"
        )
        # Bind tools to the LLM that supports tool calls.
        self.gemini_flash_llm_with_tools = self.gemini_flash_llm.bind_tools(tools)
        self.company_name = company_name
        self.articles = {}       # Will hold raw article data from the scraping tool.
        self.article_data = {}   # Will hold summarized article data.

        # Build the state graph for article summarization.
        graph = StateGraph(AgentState)
        graph.add_node("query_generator_llm", self.generate_search_queries)
        graph.add_node("execute_scrapping_tool", self.execute_scrapping_tool)
        graph.add_node("generate_article_summary", self.generate_article_summary)
        graph.add_node("unique_article_filter", self.get_unique_articles)

        graph.add_edge(START, "query_generator_llm")
        graph.add_conditional_edges(
            "query_generator_llm",
            self.should_search,
            {True: "execute_scrapping_tool", False: END}
        )
        graph.add_edge("execute_scrapping_tool", "generate_article_summary")
        graph.add_edge("generate_article_summary", "unique_article_filter")
        graph.add_edge("unique_article_filter", END)

        try:
            self.graph = graph.compile()
        except Exception as e:
            logger.error("Error compiling state graph: %s", e)
            raise

    def generate_search_queries(self, state: AgentState):
        """
        Use the LLM to verify the company name and generate search queries.
        If the company exists, two unique search queries will be returned.
        """
        system_prompt = ("""
                    First Check whether user entered company can exist or not (based on name).
                    if exists strictly gives me two unique search queries that can leads to unique articles in List.
                    Don't include anything else in the answer as i am using Output parser to your answer
                    Strictly check whether the user entered text is company or not.
                    if the company exist strictly you should call appropriate tool for getting the news article
                    
                    For example:

                    company name: Tesla
                    Answer:["Tesla Finance news","Tesla technology news"]
                    company name: Reliance Foundation
                    Answer:["Reliance Foundation projects news","Reliance Foundation impact news"]
                    company name: Apollo Chain of hospitals
                    Answer:["Apollo Chain of hospitals latest news","Apollo Chain of hospitals finance news"]
                    company name: asfasjhfajsjkhfasjfksd
                    Answer:["invalid"]
                    company name: What are you doing?
                    Answer:["invalid"]
                    company name: What is the capital of france?
                    Answer:["invalid"]  
                    company name: Government of India
                    Answer:["invalid"]          
                """)
        logger.info("Inside the generate_search_queries function")
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.company_name)
        ]
        try:
            answer = self.gemini_flash_llm_with_tools.invoke(prompt)
        except Exception as e:
            logger.error("Error invoking LLM for search queries: %s", e)
            raise
        return {"message": [answer]}

    def should_search(self, state: AgentState):
        """
        Check the LLM output from the query generation node.
        If the output is 'invalid', then do not proceed with searching.
        """
        llm_output = state["message"][-1].content
        return llm_output != "invalid"

    def execute_scrapping_tool(self, state: AgentState):
        """
        Execute the article scraping tool using search queries generated by the LLM.
        """
        last_message = state["message"][-1]
        message = None
        if not hasattr(last_message, "tool_calls"):
            logger.warning("Last message doesn't contain tool_calls")
            message = SystemMessage(
                content="Last message doesn't contain tool calling but we are trying to call tool"
            )
        else:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in self.tools:
                    try:
                        logger.info("Invoking tool '%s' with args: %s", tool_name, tool_args)
                        self.articles = self.tools[tool_name](tool_args)
                        tool_message = "Tool has successfully scrapped the articles"
                    except Exception as e:
                        tool_message = str(e)
                        logger.error("Error invoking tool '%s': %s", tool_name, e)
                    message = ToolMessage(
                        content=tool_message, tool_call_id=tool_call["id"]
                    )
                else:
                    message = ToolMessage(
                        content="Tool not found", tool_call_id=tool_call["id"]
                    )
        return {"message": [message]}

    def generate_article_summary(self, state: AgentState):
        """
        Generate summaries for each article using the Gemini LLM.
        Only articles with a non-empty title in the summary are considered valid.
        """
        valid_articles = 0
        system_prompt = (
            f"""
            Give the user relevant detail about the below article
            if you feel this is not article or article hasn't based on {self.company_name} or this article hasn't any information even if it is based on {self.company_name} strictly put every field as empty string
            Example:
            
            User: Accenture, a global IT services giant and a key indicator for the Indian IT industry, reported "constrained" discretionary project spending and a lack of significant client budget increases. Accenture CEO Julie Spellman Sweet attributed some of the slowdown to the Trump administration's "Department of Government Efficiency" initiatives. "Federal represented approximately 8% of our global revenue and 16% of our Americas revenue in FY 2024. As you know, the new administration has a clear goal to run the federal government more efficiently. During this process, many new procurement actions have slowed, which is negatively impacting our sales and revenue," Sweet stated during a call with Wall Street analysts.
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason:This is is news article but This is not about {self.company_name} company

            User: Tesla's sales are plummeting due to an aging lineup, a controversial CEO, and the failure of its 4680 battery and Cybertruck.  The company's self-driving technology is also lagging, and its future looks uncertain.
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason:This is is news article but This is not about {self.company_name} company 

            User: green revolution, great increase in production of food grains (especially wheat and rice) that resulted in large part from the introduction into developing countries of new, high-yielding varieties, beginning in the mid-20th century. Its early dramatic successes were in Mexico and the Indian subcontinent. The new varieties require large amounts of chemical fertilizers and pesticides to produce their high yields, raising concerns about cost and potentially harmful environmental effects. Poor farmers, unable to afford the fertilizers and pesticides, have often reaped even lower yields with these grains than with the older strains, which were better adapted to local conditions and had some resistance to pests and diseases. See also Norman Borlaug.
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason: This is is news article but This is not about {self.company_name} company

            User: How to trade {self.company_name}, Tech Mahindra and Kotak Mahindra Bank on Monday
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason: This is about {self.company_name} company but this is not news article 

            User: {self.company_name}'s stock rose as much as 3.11 per cent during the day to Rs 273.95 per share, the biggest intraday gain since March 3 this year. The stock pared gains to trade 1.9 per cent higher at Rs 268.8 apiece, compared to a 0.59 per cent advance in Nifty 50 as of 11:02 AM. 
            Shares of the company extended gains to their third day while they have fallen 11 per cent this year, compared to a 2.7 per cent fall in the benchmark Nifty 50. The information technology major has a total market capitalisation of Rs 2.8 trillion, according to BSE data. 
            Answer:Title='{self.company_name} Stock Rises 1.9%', Summary="{self.company_name}'s stock rose by 1.9% today, extending gains for the third day.  This is the biggest intraday gain since March 3rd. The stock's performance outpaced the Nifty 50 index.  Despite today's gains, the stock is still down 11% this year.", Sentiment='Positive', Topics=['{self.company_name}', 'Stock Market', 'Stock Performance']
            Reason: This is news article and This is About {self.company_name}  
            
            User:abcd
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason: This is is not news article and This is not about {self.company_name}  

            User: get latest news about {self.company_name}
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason: This is about {self.company_name} company but this is not news article 

            User: Track All the news about {self.company_name}
            Answer:Title='', Summary='', Sentiment='Neutral', Topics=[]
            Reason: This is about {self.company_name} company but this is not news article 
            
            """
        )
        logger.info("Generating article summaries for %d articles", len(self.articles))
        for article_key in self.articles.keys():
            if valid_articles == NUMBER_OF_UNIQUE_ARTICLES:
                break
            try:
                result = self.gemini_flash_2_llm.with_structured_output(
                    schema=ArticleData
                ).invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=self.articles[article_key]["content"])
                ])
            except Exception as e:
                logger.error("Error generating summary for %s: %s", article_key, e)
                # Return early if a critical error occurs
                return {
                    "message": [SystemMessage(content="Summary of articles got generated")],
                }
            # Consider the article valid if the title is not empty
            if result.Title:
                self.article_data[article_key] = result
                valid_articles += 1
        logger.info("Valid articles after summarization: %s", list(self.article_data.keys()))
        return {"message": [SystemMessage(content="Summary of articles got generated")]}

    def get_unique_articles(self, state: AgentState):
        """
        Use the Gemini LLM to determine the most unique articles.
        The output should be a list of article keys parseable by ast.literal_eval.
        """
        system_prompt = (
            "Strictly give me a list of the article keys (i.e. Article_i) which are most unique. "
            "Output should be parseable with python's ast.literal_eval() and nothing else."
        )
        prompt_text = ""
        for key, value in self.article_data.items():
            prompt_text += f"{key} : {value}\n"
        try:
            result = self.gemini_flash_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_text)
            ])
        except Exception as e:
            logger.error("Error invoking LLM for unique article filtering: %s", e)
            raise

        logger.info("Unique article filter LLM output: %s", result.content)
        try:
            unique_article_list = ast.literal_eval(result.content)
        except Exception as e:
            logger.error("Error parsing unique article list: %s", e)
            raise

        new_article_data = {}
        logger.info("Previous article keys: %s", list(self.article_data.keys()))
        logger.info("Unique article keys determined by LLM: %s", unique_article_list)
        for article_key in unique_article_list:
            new_article_data[article_key] = self.article_data[article_key]
        self.article_data = new_article_data
        logger.info("Article keys after filtering duplicates: %s", list(self.article_data.keys()))
        return {"message": [AIMessage(content="Duplicate articles got removed")]}


class ComparativeAgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage], operator.add] = Field(
        description="State of the Agent and previous history"
    )
    status_code: Annotated[Sequence[int], operator.add] = Field(
        description="Status code of the request"
    )


class ComparativeData(BaseModel):
    comparison: str = Field(description="The differences between articles")
    impact: str = Field(description="Overall impact that can be concluded by the articles")
    final_sentiment_statment: str = Field(
        description="Overall final sentiment analysis of the news articles"
    )


class ComparativeAnalysisAgent:
    def __init__(self, article_data):
        self.article_data = article_data
        self.compartive_data = {}
        self.hindi_summary = ""
        self.hindi_audio = []
        self.final_analysis = ""

        self.gemini_flash_llm = ChatGoogleGenerativeAI(
            api_key=secrets["GEMINI_API_KEY"],
            model="models/gemini-1.5-flash"
        )
        self.gemini_flash_2_llm = ChatGoogleGenerativeAI(
            api_key=secrets["GEMINI_API_KEY"],
            model="models/gemini-2.0-flash-lite"
        )
        graph = StateGraph(ComparativeAgentState)
        graph.add_node("comparative_analyzer", self.generate_comparative_analysis)
        graph.add_node("hindi_translator", self.translate_to_hindi)
        graph.add_node("audio_generator", self.generate_audio)

        graph.add_edge(START, "comparative_analyzer")
        graph.add_edge("comparative_analyzer", "hindi_translator")
        graph.add_edge("hindi_translator", "audio_generator")
        graph.add_edge("audio_generator", END)

        try:
            self.graph = graph.compile()
        except Exception as e:
            logger.error("Error compiling ComparativeAnalysisAgent graph: %s", e)
            raise

    def generate_comparative_analysis(self, state: ComparativeAgentState):
        """
        Generate a comparative analysis between the articles.
        Combines sentiment distribution and topic overlaps, and outputs a structured result.
        """
        try:
            system_prompt = (
                """
                Give the User comparative summary between the articles and overall impact on the company mentioned in articles. 
                LIke news Article 1 (strictly write space between the word 'Article' and it's number) focused on this and news Article 2 focused on that etc...

                The below is the correct way to write article and it's number:
                Article 1 , Article 2 ,(strictly Write in this way)

                The below is the incorrect way to write article and it's number:
                Article_1 (strictly avoid this way) , Article-1 (strictly avoid this way)
                """
            )
            prompt_text = ""
            sentiment_distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
            common_topics = set()

            for i, (key, value) in enumerate(self.article_data.items()):
                prompt_text += f"{key} : {value}\n"
                sentiment_distribution[value.Sentiment] += 1
                if i == 0:
                    common_topics |= value.Topics
                else:
                    common_topics &= value.Topics
            topic_overlap = {"Common Topics": list(common_topics)}
            for key, value in self.article_data.items():
                topic_overlap[f"Unique Topics in {key}"] = list(value.Topics.difference(common_topics))

            result = self.gemini_flash_llm.with_structured_output(
                ComparativeData
            ).invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_text)
            ])

            self.compartive_data = {
                "Comparative Sentiment Score": {
                    "Sentiment Distribution": sentiment_distribution,
                    "Coverage Differences": [{
                        "Comparison": result.comparison,
                        "Impact": result.impact
                    }],
                    "Topic Overlap": topic_overlap,
                }
            }
            self.final_analysis =  result.final_sentiment_statment
            
        except Exception as e:
            logger.error("Error generating comparative analysis: %s", e)
            return {
                "message": [AIMessage(content="Internal Server Error")],
                "status_code": [500]
            }

        return {
            "message": [AIMessage(content="Comparative text data created successfully")],
            "status_code": [200]
        }

    def translate_to_hindi(self, state: ComparativeAgentState):
        """
        Translate the comparative analysis to Hindi using the Hugging Face Inference API.
        """
        if state["status_code"][-1] == 200:
            try:
                # API_URL = "https://router.huggingface.co/hf-inference/models/facebook/nllb-200-distilled-600M"
                # headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_KEY')}"}
                # logger.info(
                #     "Input to the translation model: %s",
                #     self.compartive_data["Comparative Sentiment Score"]["Coverage Differences"][0]["Comparison"]
                # )
                # payload = {
                #     "inputs": self.compartive_data["Comparative Sentiment Score"]["Coverage Differences"][0]["Comparison"],
                #     "parameters": {
                #         "src_lang": "eng_Latn",
                #         "tgt_lang": "hin_Deva"
                #     }
                # }
                # response = requests.post(API_URL, headers=headers, json=payload)
                try:
                    response = self.gemini_flash_llm.invoke([
                        SystemMessage(content="Give the hindi text of the below content. Strictly your response should be in hindi language. Strictly don't use any english word or number as i am using hindi parser"),
                        HumanMessage(content=self.compartive_data["Comparative Sentiment Score"]["Coverage Differences"][0]["Comparison"])
                    ])
                except Exception as e:
                    return {
                        "message": [AIMessage(content=f"Error while translating: {e}")],
                        "status_code": [503]
                    }
                else:
                    translation = response.content
                    logger.info("Translation successful: %s", translation)
                    self.hindi_summary = translation
            except Exception as e:
                logger.error("Exception in translation: %s", e)
                return {
                    "message": [AIMessage(content="Internal Server Error while summary generation")],
                    "status_code": [500]
                }
            return {
                "message": [AIMessage(content="Translated to Hindi")],
                "status_code": [200]
            }
        else:
            logger.warning("Skipping translation due to previous error status")
            return state

    def generate_audio(self, state: ComparativeAgentState):
        """
        Generate Hindi speech audio from the translated summary using the Hugging Face Inference API.
        """
        if state["status_code"][-1] == 200:
            try:
                # API_URL = "https://router.huggingface.co/hf-inference/models/facebook/mms-tts-hin"
                # headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_KEY')}"}
                # payload = {"inputs": self.hindi_summary}
                # response = requests.post(API_URL, headers=headers, json=payload)
                polly_client = boto3.client(
                    'polly',
                    region_name=secrets["AWS_REGION"],
                    aws_access_key_id=secrets["AWS_ACCESS_KEY"], 
                    aws_secret_access_key=secrets["AWS_SECRET_KEY"]
                )
                response = polly_client.synthesize_speech(
                    Engine = "standard",
                    LanguageCode = "hi-IN",
                    OutputFormat = "mp3",
                    Text = self.hindi_summary,
                    VoiceId = "Aditi"
                )
                status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status_code != 200:
                    logger.error("Audio generation error: %s", response.content)
                    logger.error("Response code is ",response.status_code)
                    return {
                        "message": [AIMessage(content=f"Error while generating audio: {response.content}")],
                        "status_code": [response.status_code]
                    }
                else:
                    self.hindi_audio = response["AudioStream"].read()
                    logger.info("Audio generation successful")
            except Exception as e:
                logger.error("Exception in audio generation: %s", e)
                return {
                    "message": [AIMessage(content="Internal Server Error while generating audio")],
                    "status_code": [500]
                }
            
            return {
                "message": [AIMessage(content="Speech generated successfully")],
                "status_code": [status_code]
            }
        else:
            logger.warning("Skipping audio generation due to previous error status")
            return state
