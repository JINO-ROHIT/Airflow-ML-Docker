FROM apache/airflow:2.7.3

USER airflow

# Copy requirements.txt file into the container
COPY requirements.txt .

# Install packages listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

USER airflow