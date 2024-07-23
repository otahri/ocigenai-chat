FROM python:3.12

WORKDIR /app

COPY chat_models.py requirements.txt /app/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "chat_models.py", "--server.port=8501"]
