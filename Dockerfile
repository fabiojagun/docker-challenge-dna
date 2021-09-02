FROM python:3.8-slim-buster

#Make a directory for the application in the container, WORKDIR creates a directory (folder)
WORKDIR /data

#Install dependacies
RUN pip install pandas==1.1.5 \
                scikit-learn==0.23.2

#Copy everything in root on my WORKDIR
COPY . .

#Run application
CMD ["./start.sh"]