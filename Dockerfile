FROM python
WORKDIR /regressione_lineare
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "main.py"]