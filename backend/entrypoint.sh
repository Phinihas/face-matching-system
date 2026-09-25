#!/bin/bash
set -e

echo "⚙️  Configuring service endpoints..."

# Find installed package files directly
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

QDRANT_CFG="$SITE_PACKAGES/face_matcher/components/qdrant_config.py"
MAIN_PY="$SITE_PACKAGES/face_matcher/main.py"
SETTINGS_INI="$SITE_PACKAGES/face_matcher/database/settings.ini"

# Qdrant
sed -i "s|http://localhost:6333|${QDRANT_URL:-http://qdrant:6333}|g" "$QDRANT_CFG"
echo "  ✓ Qdrant → ${QDRANT_URL:-http://qdrant:6333}"

# OTel
sed -i "s|http://localhost:4317|${OTEL_ENDPOINT:-http://otel-collector:4317}|g" "$MAIN_PY"
echo "  ✓ OTel   → ${OTEL_ENDPOINT:-http://otel-collector:4317}"

echo "🚀 Starting FaceMatch backend on port 8050..."
exec uvicorn face_matcher.main:app --host 0.0.0.0 --port 8050