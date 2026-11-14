FROM python:3.11-slim
RUN pip install --no-cache-dir sympy==1.14.0
USER 65534:65534
WORKDIR /tmp
CMD ["python", "-I", "-B", "-"]
