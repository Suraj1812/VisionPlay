#!/bin/sh
set -eu

umask 027

prepare_dir() {
  target="$1"
  mkdir -p "$target"
  if [ "$(id -u)" = "0" ]; then
    chown -R visionplay:visionplay "$target"
  fi
}

prepare_dir /app/backend/uploads/input
prepare_dir /app/backend/uploads/output
prepare_dir /app/backend/database
prepare_dir /app/backend/model_cache
prepare_dir /app/runtime
prepare_dir /app/runtime/uploads/input
prepare_dir /app/runtime/uploads/output

if [ "$(id -u)" = "0" ]; then
  exec gosu visionplay "$@"
fi

exec "$@"
