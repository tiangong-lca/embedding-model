# Exact deployed-library reproduction, deliberately independent of GPU training lock.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
LABEL org.tiangong.owner="embedding-model-42" org.tiangong.purpose="isolated-cpu-qualification"
# NumPy matches the existing evaluation host baseline, not latest-package policy.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers==4.51.3 numpy==2.3.5 \
 && pip freeze > /runtime-freeze.txt
ENV CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 HF_HOME=/tmp/huggingface
WORKDIR /test
ENTRYPOINT ["python", "/test/qualify_cpu_embedding.py"]
