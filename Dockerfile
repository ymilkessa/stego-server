# Steganographic LLM server.
#
# The Llama model is NOT baked into the image. It is downloaded on first run
# into HF_HOME (/models), which docker-compose mounts as a named volume so the
# download survives container stop/restart/rebuild. See README "Docker".

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace caches downloads (the model weights) here. This path is a
    # mounted volume at runtime, so the model is fetched once and persisted.
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # main.py reads PORT; fix the in-container port and map it on the host side.
    PORT=3000

WORKDIR /app

# Tk runtime libs for the optional --add-gui visualizer. The slim image omits
# them; without these `import tkinter` fails. The window itself is drawn on an
# X server you forward in from the host (see README "Step-by-step GUI").
RUN apt-get update \
    && apt-get install -y --no-install-recommends tk \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer is cached across code changes.
# Pipfile is the single source of truth; --skip-lock avoids cross-platform
# hash mismatches from a lockfile generated on macOS.
COPY Pipfile ./
RUN pip install pipenv && pipenv install --system --skip-lock

# Application code (secrets and bulky docs are excluded via .dockerignore).
COPY . .

EXPOSE 3000

# Append --add-gui when ADD_GUI is set (see docker-compose.gui.yml / `make gui`).
# Tk needs an X server forwarded from the host via DISPLAY to actually draw.
CMD ["sh", "-c", "exec python main.py ${ADD_GUI:+--add-gui}"]
