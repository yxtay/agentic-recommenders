# hadolint global ignore=DL3008
FROM ghcr.io/astral-sh/uv:0.11.12@sha256:3a59a3cdd5f7c217faa36e32dbc7fddbb0412889c2a0a5229f6d790e5a019dd7 AS uv

FROM mcr.microsoft.com/devcontainers/base:debian

ENV UV_LOCKED=1

COPY --from=uv /uv /uvx /bin/
COPY .python-version pyproject.toml uv.lock ./
RUN uv python install && \
    uv sync && \
    chown -R vscode:vscode .venv
