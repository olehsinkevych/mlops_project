FROM python:3.12-slim

WORKDIR /mlops_project

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements/requirements.txt /requirements/requirements.txt
RUN pip install --upgrade pip
RUN pip install pip-tools
RUN pip install --no-cache-dir -r /requirements/requirements.txt

COPY . .
LABEL authors="spirit"

# to complete
CMD ["uvicorn", "mlops-project.main:app", "--host", "0.0.0.0", "--port", "8000"]