FROM python:3.10.14-bullseye

WORKDIR /flasy-ml

COPY requirements.txt /flasy-ml/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /flasy-ml

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
