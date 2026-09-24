# syntax=docker/dockerfile:1.7
FROM ghcr.io/openfaas/of-watchdog:0.12.1 AS watchdog

FROM python:3.12-slim

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog

WORKDIR /home/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pretrained multi-style pastiche generators from Magenta.
ADD --checksum=sha256:7044c0a6cad9a2865a0aded7567be31437fd6dcc289a40df1efb5096d9a29d88 \
    https://storage.googleapis.com/download.magenta.tensorflow.org/models/multistyle-pastiche-generator-monet.ckpt \
    /models/
ADD --checksum=sha256:f0de79e9b91e28e260873cf26cb68589179fe0519c6d1a86a4eb2d4aa5023083 \
    https://storage.googleapis.com/download.magenta.tensorflow.org/models/multistyle-pastiche-generator-varied.ckpt \
    /models/
RUN chmod 644 /models/*

COPY function function

RUN useradd --create-home app
USER app

# of-watchdog in HTTP mode keeps one long-lived gunicorn process, so the
# models are loaded once at startup instead of on every request.
ENV fprocess="gunicorn --bind 127.0.0.1:5000 --workers 1 --threads 4 --timeout 300 function.handler:app" \
    mode="http" \
    upstream_url="http://127.0.0.1:5000" \
    exec_timeout="300s" \
    read_timeout="300s" \
    write_timeout="300s" \
    TF_CPP_MIN_LOG_LEVEL="3"

EXPOSE 8080

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
