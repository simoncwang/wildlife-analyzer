#!/usr/bin/env bash
set -e

echo "🚀 Running wildlife pipeline with config:"
cat pipeline_config.yaml

# Extract run_mode from config
RUN_MODE=$(grep "^run_mode" pipeline_config.yaml | awk '{print $2}' | tr -d '"')

echo "🔧 Pipeline mode: $RUN_MODE"

# Shared stages
python pipeline/fetch_and_log.py
python pipeline/preprocess.py

# Clustering
if [[ "$RUN_MODE" == "clustering"]; then
  echo "🔢 Running clustering step..."
  python pipeline/feature_engineering.py
  python models/cluster.py
fi

# LLM Summary
if [[ "$RUN_MODE" == "llm_summary"]; then
  echo "🧠 Running LLM summary step..."
  python models/llm_summary.py
fi

echo "✅ Pipeline finished at $(date)."