import fitz 
import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(pdf_file):
   
    try:
        
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
            
        return text
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def get_llm_response(prompt):
   
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7, 
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error with Groq LLM: {e}")
        return None