#!/bin/bash

# Multi-Agent AI Setup Script
echo "🚀 Setting up Multi-Agent AI Response Engine..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f .env ]; then
    echo "🔑 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your Tavily API key!"
    echo "   Get your API key from: https://tavily.com/"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your Tavily API key"
echo "2. Run the application: python main.py"
echo "3. Enjoy your multi-agent AI system!"
echo ""
echo "🌟 Star this repo if you found it helpful!"
