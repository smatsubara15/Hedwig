FROM python:latest
EXPOSE 8080
WORKDIR /Hedwig/98_UI_Design
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "Streamlit_App_v1.py", "--server.port=8080", "--server.address=0.0.0.0"]