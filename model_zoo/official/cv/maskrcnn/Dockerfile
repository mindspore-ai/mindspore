ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN apt install libgl1-mesa-glx -y
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt
