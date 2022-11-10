#!/bin/sh

echo "Waiting for triton..."

while ! curl -o - tritonserver:8001/v2/health/ready; do
  sleep 0.3
done

echo "Triton started"
exec "$@"