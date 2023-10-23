FROM python:3.8
EXPOSE 8080
WORKDIR /Hedwig/98_UI_Design/Dockerfile
COPY . ./
# Copy a folder from a different directory to the container
COPY Hedwig/05_RRM/vectorstores/* ./

RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "Streamlit_App_v1.py", "--server.port=8080", "--server.address=0.0.0.0"]