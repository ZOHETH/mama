FROM apache/airflow:2.2.2
USER root
RUN mkdir /data \
  && chown -R airflow /data
USER airflow
RUN pip install -i https://pypi.douban.com/simple pip -U \
  && pip config set global.index-url https://pypi.douban.com/simple \
  && pip config set global.trusted-host pypi.douban.com

COPY ./requirements.txt  /opt/airflow/requirements.txt

WORKDIR /opt/airflow
RUN pip install --no-cache-dir -r requirements.txt
