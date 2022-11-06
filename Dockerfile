FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN rm -rf /var/lib/apt/lists
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir torch numpy cython pymsteams flask waitress matplotlib
RUN pip3 install --upgrade setuptools
RUN pip3 install --no-cache-dir spacy==3.2.3

ENV LANG=C.UTF-8
RUN python3 -m spacy download en_core_web_md

COPY PrepareDataset.py ./PrepareDataset.py
COPY Model.py ./Model.py
COPY TrainModel.py ./TrainModel.py
COPY TrainModel_Outliers.py ./TrainModel_Outliers.py
COPY EvaluateModel.py ./EvaluateModel.py
COPY Monitoring.py ./Monitoring.py
COPY MonitoringService.py ./MonitoringService.py
COPY startup.sh ./startup.sh
RUN chmod +x startup.sh

ENV HOOK_TEAMS=
ENV ACTION=MONITOR

CMD ./startup.sh
