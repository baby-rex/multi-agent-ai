# Multi-Agent AI Response Engine

🤖 An intelligent AI routing system that automatically classifies user queries and dispatches them to specialized agents using entirely open-source Hugging Face models.

## 🚀 Features

- **🧮 Math Agent**: Handles equations and calculations with custom solver functions
- **🔍 Technical Agent**: Performs semantic search through documentation using FAISS vector stores
- **🎯 Smart Classification**: Automatically routes questions to the appropriate agent
- **🤗 Hugging Face Models**: Uses open-source models (no OpenAI API required)
- **⚡ Cost-Effective**: Completely free to run with Hugging Face transformers

## 🛠️ Technology Stack

- **LangChain**: Agent orchestration and chain management
- **Hugging Face Transformers**: Text generation and embeddings
- **FAISS**: Vector database for document retrieval
- **Sentence Transformers**: Document embeddings
- **Tavily**: Web search capabilities

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/multi-agent-ai.git
   cd multi-agent-ai
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Tavily API key
   ```

## 🔑 API Keys Setup

1. **Tavily API Key** (Required):
   - Sign up at [https://tavily.com/](https://tavily.com/)
   - Get your API key
   - Add it to `.env`: `TAVILY_API_KEY=your_key_here`

2. **Hugging Face Token** (Optional):
   - For private models or higher rate limits
   - Get token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

## 🎮 Usage

Run the application:
```bash
python main.py
```

### Example Interactions:

**Math Questions:**
- "What's the solution to 3x+5=14?" → **Math Agent** → `x = 3.0`
- "Calculate 15 + 27 * 2" → **Math Agent** → `69`

**Technical Questions:**
- "What is LangSmith?" → **Technical Agent** → Documentation search
- "How does LangChain work?" → **Technical Agent** → Knowledge retrieval

## 🏗️ Architecture

```
User Question
     ↓
Classification Logic
     ↓
┌─────────────────┐    ┌──────────────────┐
│   Math Agent    │    │ Technical Agent  │
│                 │    │                  │
│ • Calculator    │    │ • Web Loader     │
│ • Equation      │    │ • FAISS Vector   │
│   Solver        │    │   Store          │
│ • DialoGPT      │    │ • Retrieval QA   │
└─────────────────┘    └──────────────────┘
```

## 🔧 Configuration

### Models Used:
- **Text Generation**: `microsoft/DialoGPT-medium`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Document Processing**: LangChain + FAISS

### Customization:
- Modify `classify_question()` to change routing logic
- Update models in agent creation functions
- Add new agents for different domains

## 📊 Performance

- **Startup Time**: ~30-60 seconds (model loading)
- **Response Time**: 2-5 seconds per query
- **Memory Usage**: ~2-4 GB (with models loaded)
- **Cost**: $0 (except for Tavily API calls)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the agent framework
- [Hugging Face](https://huggingface.co/) for open-source models
- [Tavily](https://tavily.com/) for web search capabilities
- [FAISS](https://faiss.ai/) for vector similarity search

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ **Star this repo if you found it helpful!**
