FROM vllm/vllm-openai:v0.11.0


RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system flashinfer-cubin==0.4.1 \
    && uv pip install --system flashinfer-jit-cache==0.4.1 \
        --extra-index-url https://flashinfer.ai/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
    && flashinfer show-config


RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system flashinfer-python==0.4.1 --prerelease=allow