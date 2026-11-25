# Inkling2 support is not in a tagged vLLM release yet (the model recipe asks
# for >= 0.35.0), so this builds on a nightly. The default is the exact
# nightly the SM86 patch and every number in results/ were validated on
# (0.34.0rc1.dev1235+gf2f9edcac, manifest sha256:eaa63fd6ee3a...bfaf19a7).
# Set VLLM_VERSION=nightly to track upstream instead.
ARG VLLM_VERSION=nightly-f2f9edcac30d128bc090a3c032e4ba125716a641
FROM vllm/vllm-openai:${VLLM_VERSION}

# The container runs as the host user so Hugging Face downloads and compile
# caches stay user-owned. A fresh named volume inherits ownership from the
# image path it covers, so /home/vllm must already be chowned here or the
# mount lands root-owned and unwritable.
#
# The uid also needs a real /etc/passwd entry: torch's inductor cache calls
# getpass.getuser(), which raises KeyError on a uid with no passwd record and
# kills the process during `import vllm`.
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
# NB: the base image already owns the name `vllm` (uid 2000), hence `hostuser`.
RUN set -eux; \
    if ! getent group "${DOCKER_GID}" >/dev/null; then \
        groupadd -g "${DOCKER_GID}" hostuser; \
    fi; \
    if ! getent passwd "${DOCKER_UID}" >/dev/null; then \
        useradd -u "${DOCKER_UID}" -g "${DOCKER_GID}" -M -d /home/vllm -s /bin/bash hostuser; \
    fi; \
    mkdir -p /home/vllm /hf-cache; \
    chown -R "${DOCKER_UID}:${DOCKER_GID}" /home/vllm /hf-cache

# Inkling2 on Ampere. Upstream gates Inkling2 to SM90+ and the RTX 3090 is
# SM86; inkling_sm86_patch.py documents and applies the four small edits that
# make the existing code paths run there. NOT an upstream-supported
# configuration. Set INKLING_SM86_PATCH=0 to build an unpatched image.
ARG INKLING_SM86_PATCH=1
COPY inkling_sm86_patch.py /tmp/inkling_sm86_patch.py
RUN if [ "${INKLING_SM86_PATCH}" = "1" ]; then python3 /tmp/inkling_sm86_patch.py; fi

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

# Base image sets ENTRYPOINT ["vllm", "serve"]; replace it wholesale so the
# env-driven flag assembly in entrypoint.sh runs first.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
