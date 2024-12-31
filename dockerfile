FROM python:3.12-slim


# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /code

# Install the application dependencies.
WORKDIR /code

USER root

RUN uv sync --frozen --no-cache --no-dev

EXPOSE 8080

# Run the application.
CMD  ["sh", "-c", "ray start --head --port=6379 --object-manager-port=8076 --include-dashboard=False && serve run app.main:SolubilityInference"]