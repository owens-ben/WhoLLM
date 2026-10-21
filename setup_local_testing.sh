#!/bin/bash
# Setup script for local Home Assistant testing
# This script helps you link the integration to your Home Assistant instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_SRC="$SCRIPT_DIR/custom_components/roomer"
INTEGRATION_NAME="roomer"

echo "🔧 Roomer - Local Testing Setup"
echo "==========================================="
echo ""

# Check if integration source exists
if [ ! -d "$INTEGRATION_SRC" ]; then
    echo "❌ Error: Integration source not found at $INTEGRATION_SRC"
    exit 1
fi

echo "✅ Integration source found: $INTEGRATION_SRC"
echo ""

# Common HA config locations
HA_CONFIG_PATHS=(
    "$HOME/.homeassistant"
    "$HOME/.config/homeassistant"
    "/config"
    "/usr/share/hassio/homeassistant"
    "/data/homeassistant"
)

echo "🔍 Searching for Home Assistant config directory..."
echo ""

HA_CONFIG=""
for path in "${HA_CONFIG_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/configuration.yaml" ]; then
        HA_CONFIG="$path"
        echo "✅ Found Home Assistant config at: $path"
        break
    fi
done

if [ -z "$HA_CONFIG" ]; then
    echo "⚠️  Could not automatically find Home Assistant config directory."
    echo ""
    echo "Please enter the path to your Home Assistant config directory:"
    read -r HA_CONFIG
    
    if [ ! -d "$HA_CONFIG" ]; then
        echo "❌ Error: Directory does not exist: $HA_CONFIG"
        exit 1
    fi
    
    if [ ! -f "$HA_CONFIG/configuration.yaml" ]; then
        echo "⚠️  Warning: configuration.yaml not found. Is this the correct directory?"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

CUSTOM_COMPONENTS_DIR="$HA_CONFIG/custom_components"
INTEGRATION_DST="$CUSTOM_COMPONENTS_DIR/$INTEGRATION_NAME"

echo ""
echo "📁 Target directory: $INTEGRATION_DST"
echo ""

# Create custom_components directory if it doesn't exist
if [ ! -d "$CUSTOM_COMPONENTS_DIR" ]; then
    echo "📂 Creating custom_components directory..."
    mkdir -p "$CUSTOM_COMPONENTS_DIR"
fi

# Check if integration already exists
if [ -e "$INTEGRATION_DST" ]; then
    echo "⚠️  Integration already exists at: $INTEGRATION_DST"
    echo ""
    echo "Options:"
    echo "  1) Remove existing and create symlink (recommended for development)"
    echo "  2) Remove existing and copy files"
    echo "  3) Cancel"
    echo ""
    read -p "Choose option (1-3): " -n 1 -r
    echo ""
    
    case $REPLY in
        1)
            echo "🗑️  Removing existing integration..."
            rm -rf "$INTEGRATION_DST"
            echo "🔗 Creating symlink..."
            ln -s "$INTEGRATION_SRC" "$INTEGRATION_DST"
            echo "✅ Symlink created!"
            ;;
        2)
            echo "🗑️  Removing existing integration..."
            rm -rf "$INTEGRATION_DST"
            echo "📋 Copying files..."
            cp -r "$INTEGRATION_SRC" "$INTEGRATION_DST"
            echo "✅ Files copied!"
            ;;
        3)
            echo "❌ Cancelled."
            exit 0
            ;;
        *)
            echo "❌ Invalid option."
            exit 1
            ;;
    esac
else
    echo "🔗 Creating symlink (recommended for development)..."
    echo "   This allows you to edit code and see changes after restart."
    echo ""
    read -p "Create symlink? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ln -s "$INTEGRATION_SRC" "$INTEGRATION_DST"
        echo "✅ Symlink created!"
    else
        echo "📋 Copying files instead..."
        cp -r "$INTEGRATION_SRC" "$INTEGRATION_DST"
        echo "✅ Files copied!"
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Restart Home Assistant"
echo "   2. Go to Settings → Devices & Services → Add Integration"
echo "   3. Search for 'Roomer'"
echo "   4. Follow the setup wizard"
echo ""
echo "💡 Tip: Enable debug logging in configuration.yaml:"
echo "   logger:"
echo "     default: info"
echo "     logs:"
echo "       custom_components.roomer: debug"
echo ""

