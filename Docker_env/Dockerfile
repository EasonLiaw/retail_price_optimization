FROM python:3.10
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip install -r requirements.txt
RUN mv _tree.py ../usr/local/lib/python3.10/site-packages/shap/explainers
CMD streamlit run pipeline_api.py --server.port $PORT