ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG SDK_PKG

RUN ln -s /usr/local/python3.7.5/bin/python3.7 /usr/bin/python

COPY $SDK_PKG .
RUN chmod +x ${SDK_PKG} && \
    ./${SDK_PKG} --install-path=/home/run --install && \
    bash -c "source ~/.bashrc"