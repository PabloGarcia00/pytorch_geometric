# SM -> PV disaggregation Streamlit app (graphgym/deploy).
# Build context: this directory (graphgym/).
#   docker compose build   (from graphgym/, see docker-compose.yml)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Deps first, separately from source, for Docker layer-cache reuse --
# this layer only invalidates when pyproject.toml changes.
#
# The shared uv.lock pins a CUDA torch build (torch==2.13.0+cu130) sized
# for a GPU dev box; potro has no GPU and the model is tiny (~150K params,
# single-threaded CPU inference already ~1.4s for the largest realistic
# batch). Tried `uv sync --frozen --no-install-package torch` + a separate
# CPU-index install first -- that swaps the torch wheel itself but
# --frozen still installs every OTHER pinned entry verbatim, including
# torch's now-orphaned CUDA runtime deps (nvidia-cublas/cudnn/cufft/nccl,
# triton, ...), since --frozen means "install exactly what's locked" and
# never re-resolves to notice they're no longer needed -- bloated the
# image to 16.8GB. Fixed by appending a CPU-only index override for torch
# to a COPY of pyproject.toml (the real one on disk is untouched -- this
# doesn't affect local dev, which may need CUDA elsewhere) and resolving
# fresh (no uv.lock, no --frozen): a CPU-only torch wheel's own metadata
# declares zero nvidia-* dependencies, so they're never pulled in at all.
COPY pyproject.toml ./
RUN printf '\n[tool.uv.sources]\ntorch = [{ index = "pytorch-cpu" }]\n\n[[tool.uv.index]]\nname = "pytorch-cpu"\nurl = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n' >> pyproject.toml
RUN uv sync --no-dev

# Only what disaggregate_pv.py actually needs at runtime (not the full
# monorepo, not the rest of results/ or datasets/ -- see GRAPHGYM_ROOT /
# CKPT_PATH / TRANSFORM_PATH / CFG_PATH in deploy/disaggregate_pv.py).
COPY custom_graphgym/ ./custom_graphgym/
COPY configs/pyg/res_eval_cvae_5min_dualmask.yaml ./configs/pyg/res_eval_cvae_5min_dualmask.yaml
COPY results/res_eval_cvae_5min_dualmask/0/ckpt/ ./results/res_eval_cvae_5min_dualmask/0/ckpt/
COPY datasets/earne_cvae_dual_physmask_5min/ ./datasets/earne_cvae_dual_physmask_5min/
COPY deploy/ ./deploy/

# deploy/.env (secret) and deploy/data/ (runtime cache) are excluded by
# .dockerignore -- never baked into an image layer, supplied at container
# start via docker-compose volume mounts instead. Pre-create the mount
# target so a fresh volume mount has somewhere to land; world-writable
# since the bind-mounted host directory's ownership won't generally match
# the container's non-root UID below (acceptable for a personal,
# single-service deployment; the app process itself still runs unprivileged).
RUN mkdir -p deploy/data && chmod 777 deploy/data

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Invoke the venv's own binaries directly rather than `uv run` -- uv run
# implicitly re-syncs against pyproject.toml at every invocation and
# doesn't remember the build-time --no-dev preference, so it was
# downloading the (dev-only, browser-testing) playwright dependency fresh
# on every container start. Direct invocation needs no network access at
# runtime and starts faster.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD .venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD [".venv/bin/streamlit", "run", "deploy/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
