#https://medium.com/@tasnuva2606/dockerize-flask-app-4998a378a6aa
#FROM python:3.6.5-alpine
#WORKDIR /project
#ADD . /project
#RUN pip install -r requirements.txt
#CMD ["python","app.py"]

#https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
#COPY ./iris_trained_model.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
