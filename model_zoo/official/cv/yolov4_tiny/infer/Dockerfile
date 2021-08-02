ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG SDK_PKG

RUN ln -s  /usr/local/python3.7.5/bin/python3.7 /usr/bin/python

RUN apt-get update && \
    apt-get install libglib2.0-dev -y || \
    rm -rf /var/lib/dpkg/info && \
    mkdir /var/lib/dpkg/info && \
    apt-get install libglib2.0-dev -y && \
    pip install pytest-runner==5.3.0
COPY sdk/requirements.txt .
RUN pip3.7 install -r requirements.txt

# pip install sdk_run
COPY $SDK_PKG .
RUN ls -hrlt
RUN chmod +x ${SDK_PKG} && \
    ./${SDK_PKG}  --install && \
     bash -c "source ~/.bashrc"
