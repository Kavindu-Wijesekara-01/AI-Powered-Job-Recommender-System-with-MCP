import fitz  # PyMuPDF library එක
import os
from dotenv import load_dotenv
from groq import Groq

# .env file එකේ තියෙන රහස් keys (API keys) load කරගන්නවා
load_dotenv()

# Groq client එක හදාගන්නවා
GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

def extract_text_from_pdf(pdf_file):
    """
    අප්ලෝඩ් කරන PDF Resume එකකින් text එක වෙන් කරලා ගන්න function එක.
    """
    try:
        # Streamlit එකෙන් එන file එක read කරන්න පුළුවන් විදිහට open කරනවා
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        
        # PDF එකේ තියෙන හැම page එකකින්ම text එක අරන් එකතු කරනවා
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
            
        return text
    
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def get_llm_response(prompt):
    """
    Groq LLM එකට prompt එකක් දීලා, ඒකෙන් එන උත්තරේ ගන්න function එක.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7, # 0.7 වගේ දුන්නම උත්තරේ ටිකක් නිර්මාණශීලී වෙනවා
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error with Groq LLM: {e}")
        return None