FROM apache/airflow:2.2.2
USER root
RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
USER airflow
# 替换pip国内源
RUN pip install -i https://pypi.douban.com/simple pip -U \
    && pip config set global.index-url https://pypi.douban.com/simple \
    && pip config set global.trusted-host pypi.douban.com
COPY ./requirements.txt  /opt/airflow/requirements.txt

WORKDIR /opt/airflow
RUN pip install -r requirements.txt