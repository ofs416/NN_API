# Final stage
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y curl

# Copy the application into the container
COPY . /code
WORKDIR /code

# Add build argument for the dependency group
ARG DEPS_GROUP
RUN uv sync --frozen --no-cache --group ${DEPS_GROUP}
