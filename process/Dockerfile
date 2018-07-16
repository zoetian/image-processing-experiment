FROM tensorflow/magenta

# Install fwatchdog
ADD https://github.com/openfaas/faas/releases/download/0.6.9/fwatchdog /usr/bin
RUN chmod +x /usr/bin/fwatchdog

# Set work directory under root
WORKDIR /root/

# Install and run tensorflow 1.0.1
ADD install.sh .
RUN sh install.sh

# Run index.py
COPY index.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy functionalities
COPY function function

RUN touch ./function/__init__.py

WORKDIR /root/function/
COPY function/requirements.txt	.
RUN pip install -r requirements.txt

WORKDIR /root/

# Copy raw_iamge content
COPY raw_iamge raw_iamge

ENV fprocess="python index.py"

HEALTHCHECK --interval=1s CMD [ -e /tmp/.lock ] || exit 1

CMD [ "/usr/bin/fwatchdog"]