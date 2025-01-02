FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /code

# Install the application dependencies.
WORKDIR /code

RUN uv sync --frozen --no-cache --no-dev
 