#  AI-Powered Job Recommender System

An advanced, containerized AI application that analyzes uploaded resumes (PDFs) to provide personalized career recommendations, identify skill gaps, and automatically fetch real-time job opportunities from LinkedIn. Built with a strong focus on AI integration and MLOps best practices.

## Features
* ** Smart Resume Parsing:** Extracts text from PDF resumes accurately.
* **AI Analysis:** Utilizes LLMs (Groq / Llama 3.1) to generate professional summaries, analyze industry-specific skill gaps, and suggest career roadmaps.
* ** Real-Time Job Scraping:** Integrates with Apify to scrape and present the latest matching job opportunities directly from LinkedIn.
* ** Dockerized Environment:** Fully containerized using Docker and Docker Compose for seamless deployment and reproducibility.
* ** Responsive UI:** Clean and interactive user interface built with Streamlit.

## Tech Stack & Tools
* **Language:** Python 3.10
* **AI/LLM:** Groq API (Llama-3.1-8b-instant)
* **Web Scraping:** Apify Client (LinkedIn Jobs Scraper)
* **Frontend:** Streamlit
* **PDF Processing:** PyMuPDF (fitz)
* **DevOps/MLOps:** Docker, Docker Compose

##  How to Run Locally

### Prerequisites
1. Install [Docker](https://www.docker.com/) and Docker Compose.
2. Get your API keys from [Groq](https://console.groq.com/) and [Apify](https://console.apify.com/).

### Installation
1. Clone this repository:
```bash
   git clone [https://github.com/your-username/job-recommender-system.git](https://github.com/your-username/job-recommender-system.git)
   cd job-recommender-system
