#!/usr/bin/env bash
set -uo pipefail

endpoint="${HF_ENDPOINT:-https://huggingface.co}"
revision="${HF_REVISION:-main}"
max_parallel="${MAX_PARALLEL:-1}"
model_dir="${MODEL_DIR:-models}"
model="${1:-}"

download_model() {
  local repo="$1"
  local target="$2"
  shift 2
  local files=("$@")
  local log_dir="${MODEL_DIR:-models}/_download_meta/${target##*/}-logs"

  mkdir -p "$target" "$log_dir"
  cd "$target" || return 1

  download_one() {
    local file="$1"
    local url remote_size local_size
    mkdir -p "$(dirname "$file")"
    url="${endpoint}/${repo}/resolve/${revision}/${file}"
    echo "[$(date '+%F %T')] start ${file}"

    remote_size="$(
      curl -fsSIL --connect-timeout 30 --max-time 120 --retry 5 --retry-delay 5 "$url" \
        | awk 'tolower($1) == "content-length:" { size=$2 } END { gsub("\r", "", size); print size }'
    )" || remote_size=""

    if [[ -n "${remote_size}" && -f "$file" ]]; then
      local_size="$(stat -c %s "$file")"
      if [[ "$local_size" == "$remote_size" ]]; then
        echo "    complete, skip"
        return 0
      fi
      if (( local_size > remote_size )); then
        mv "$file" "${file}.oversize.$(date +%Y%m%d%H%M%S)"
      elif [[ ! -f "${file}.part" ]]; then
        mv "$file" "${file}.part"
      fi
    fi

    curl \
      --fail \
      --location \
      --http1.1 \
      --continue-at - \
      --retry 999 \
      --retry-all-errors \
      --retry-connrefused \
      --retry-delay 10 \
      --connect-timeout 30 \
      --speed-time 120 \
      --speed-limit 1024 \
      --output "${file}.part" \
      --silent \
      --show-error \
      --write-out "    downloaded %{size_download} bytes, speed %{speed_download} B/s\n" \
      "$url"

    if [[ -n "${remote_size}" ]]; then
      local_size="$(stat -c %s "${file}.part")"
      if [[ "$local_size" != "$remote_size" ]]; then
        echo "    size mismatch after curl: local=${local_size} remote=${remote_size}" >&2
        return 1
      fi
    fi
    mv "${file}.part" "$file"
    echo "[$(date '+%F %T')] done ${file}"
  }

  for file in "${files[@]}"; do
    while (( $(jobs -rp | wc -l) >= max_parallel )); do
      sleep 5
    done
    safe_name="${file//\//__}"
    download_one "$file" > "${log_dir}/${safe_name}.log" 2>&1 &
  done

  wait
  echo "Downloaded ${repo} to ${target}"
}

qwen14_files=(
  ".gitattributes"
  "LICENSE"
  "README.md"
  "config.json"
  "generation_config.json"
  "merges.txt"
  "model-00001-of-00008.safetensors"
  "model-00002-of-00008.safetensors"
  "model-00003-of-00008.safetensors"
  "model-00004-of-00008.safetensors"
  "model-00005-of-00008.safetensors"
  "model-00006-of-00008.safetensors"
  "model-00007-of-00008.safetensors"
  "model-00008-of-00008.safetensors"
  "model.safetensors.index.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

qwen32_awq_files=(
  ".gitattributes"
  "LICENSE"
  "README.md"
  "config.json"
  "generation_config.json"
  "merges.txt"
  "model-00001-of-00005.safetensors"
  "model-00002-of-00005.safetensors"
  "model-00003-of-00005.safetensors"
  "model-00004-of-00005.safetensors"
  "model-00005-of-00005.safetensors"
  "model.safetensors.index.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

case "$model" in
  14b)
    download_model "Qwen/Qwen2.5-14B-Instruct" "${model_dir}/Qwen2.5-14B-Instruct" "${qwen14_files[@]}"
    ;;
  32b-awq)
    download_model "Qwen/Qwen2.5-32B-Instruct-AWQ" "${model_dir}/Qwen2.5-32B-Instruct-AWQ" "${qwen32_awq_files[@]}"
    ;;
  all|"")
    download_model "Qwen/Qwen2.5-14B-Instruct" "${model_dir}/Qwen2.5-14B-Instruct" "${qwen14_files[@]}" || exit 1
    download_model "Qwen/Qwen2.5-32B-Instruct-AWQ" "${model_dir}/Qwen2.5-32B-Instruct-AWQ" "${qwen32_awq_files[@]}"
    ;;
  *)
    echo "usage: $0 [14b|32b-awq|all]" >&2
    exit 2
    ;;
esac
