#!/bin/bash
# Quick check script for Roomer integration
#
# Usage: ./check_integration.sh [HA_CONFIG_PATH]
# Example: ./check_integration.sh /config

# Set your HA config path here or pass as argument
HA_CONFIG="${1:-/config}"
INTEGRATION_DST="$HA_CONFIG/custom_components/roomer"

echo "🔍 Roomer Integration Check"
echo "======================================"
echo ""

# Check symlink
if [ -L "$INTEGRATION_DST" ]; then
    echo "✅ Symlink exists"
    echo "   → $(readlink -f "$INTEGRATION_DST")"
else
    echo "❌ Symlink missing!"
    exit 1
fi

echo ""

# Check key files
echo "📁 Checking key files..."
for file in "manifest.json" "__init__.py" "config_flow.py"; do
    if [ -f "$INTEGRATION_DST/$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file MISSING!"
    fi
done

echo ""

# Check manifest
echo "📋 Manifest.json:"
python3 -m json.tool "$INTEGRATION_DST/manifest.json" 2>/dev/null | grep -E "(domain|name|version)" | sed 's/^/   /'

echo ""

# Check for errors in logs
echo "🔍 Recent Home Assistant logs (last 50 lines, filtering for roomer/error):"
docker logs homeassistant --tail 50 2>&1 | grep -iE "(roomer|error.*llm|import.*llm)" | tail -5 || echo "   (No relevant errors found)"

echo ""
echo "📝 Next Steps:"
echo "   1. Open Home Assistant UI"
echo "   2. Go to: Settings → Devices & Services"
echo "   3. Click: + Add Integration (bottom right button)"
echo "   4. Search for: 'Roomer'"
echo ""
echo "💡 If integration doesn't appear:"
echo "   - Wait for HA to fully start (check UI shows 'Home' loaded)"
echo "   - Try hard refresh (Ctrl+Shift+R) or incognito mode"
echo "   - Check full logs: docker logs homeassistant | grep -i roomer"

