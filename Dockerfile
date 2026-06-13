# Python 3.10 පාවිච්චි කරනවා (Slim version එක size එක අඩුයි)
FROM python:3.10-slim

# Docker image එක ඇතුළේ වැඩ කරන folder එක හදනවා
WORKDIR /app

# මුලින්ම requirements.txt එක copy කරගන්නවා
COPY requirements.txt .

# Packages ටික install කරගන්නවා
RUN pip install --no-cache-dir -r requirements.txt

# ඊටපස්සේ project එකේ අනිත් files ඔක්කොම copy කරගන්නවා (app.py සහ src folder එක)
COPY . .

# Streamlit run වෙන්නේ 8501 port එකේ නිසා ඒක open කරනවා
EXPOSE 8501

# App එක Run කරන්න ඕන command එක
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]