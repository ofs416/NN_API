# Build stage
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container
COPY . /code
WORKDIR /code

# Add build argument for the dependency group
ARG DEPS_GROUP=prod-back
RUN uv sync --frozen --no-cache --group ${DEPS_GROUP}

# Final stage
FROM python:3.12-slim

# Copy uv and dependencies from builder
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx
COPY --from=builder /code /code

WORKDIR /code
 