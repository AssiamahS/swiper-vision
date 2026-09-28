# swiper-vision

A vision-model seat that runs on a GitHub Actions runner: pulls an open-weights vision model with ollama,
connects to a relay over a WebSocket and answers JSON questions about images. No model API key involved.

Secrets: `RELAY_URL`, `RELAY_KEY`. Optional repo variable `VISION_MODEL` (default `gemma3:4b`).
