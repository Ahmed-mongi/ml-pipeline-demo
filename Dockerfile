FROM python:3.10-slim

ARG RUN_ID

RUN pip install --no-cache-dir mlflow scikit-learn

WORKDIR /app

RUN echo "Downloading model for Run ID: ${RUN_ID}" && \
    echo "Model download simulated successfully"

COPY . .

CMD ["python", "-c", "print('Model server running')"]
```

4. Click **Commit changes** → **Commit changes**
