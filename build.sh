#!/usr/bin/env bash
# Builds lambda.zip: our code + the libraries Lambda doesn't already include.
# Libraries are downloaded for Lambda's OS (Linux, ARM64), not for this Mac.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf build lambda.zip
mkdir build

.venv/bin/pip install requests \
  --target build \
  --platform manylinux2014_aarch64 \
  --python-version 3.12 \
  --only-binary=:all: \
  --quiet

cp lambda_function.py pipeline.py build/
(cd build && zip -rq ../lambda.zip . -x '*.pyc' -x '__pycache__/*')

echo "Built lambda.zip ($(du -h lambda.zip | cut -f1))"
