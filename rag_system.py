# rag_system.py

import google.generativeai as genai

from vectorstore import initialize_vectorstore
from config import GEMEINI_API_KEY

# Configure the API key for generative AI
genai.configure(api_key=GEMEINI_API_KEY)

def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"', "").replace("\n"," ")
    prompt = ("""
You are a helpful and informative bot that answers questions using text from the reference context included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and conversational tone. \
  If the context is irrelevant to the answer, you may ignore it.
                QUESTION: '{query}'
                CONTEXT: '{context}'
              
              ANSWER:
              """).format(query=query, context=context)
    return prompt

def get_relevant_context_from_db(query):
    vectorstore = initialize_vectorstore()
    context = ""
    search_results = vectorstore.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text
