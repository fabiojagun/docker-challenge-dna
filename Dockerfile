FROM python:3.8-slim-buster

RUN mkdir /app
WORKDIR /app

RUN pip install jupyter \
                pandas==1.1.5 \
                scikit-learn==0.23.2

COPY . .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]