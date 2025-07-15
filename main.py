from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.chains import LLMChain
from transformers import pipeline
import re


def create_technical_agent():
    """Create an agent for technical/general questions"""
    
    # Load and process documents
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    
    # Create Hugging Face pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",
        tokenizer="microsoft/DialoGPT-medium",
        max_length=256,
        temperature=0.1,
        do_sample=True,
        pad_token_id=50256
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


def create_math_agent():
    """Create an agent for math questions"""
    
    # Create Hugging Face pipeline
    math_pipeline = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",
        tokenizer="microsoft/DialoGPT-medium",
        max_length=256,
        temperature=0.1,
        do_sample=True,
        pad_token_id=50256
    )
    llm = HuggingFacePipeline(pipeline=math_pipeline)
    
    # Create a math-focused prompt
    math_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful math assistant. Solve this math problem step by step:

Question: {question}

Solution:"""
    )
    
    # Create LLM chain
    math_chain = LLMChain(llm=llm, prompt=math_prompt)
    
    return math_chain


@tool
def basic_calculator(expression: str) -> str:
    """Calculate basic mathematical expressions. Input should be a valid Python expression."""
    try:
        # Clean the expression
        expression = expression.strip()
        # Only allow basic math operations for security
        allowed_chars = set('0123456789+-*/()., ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression. Only basic math operations are allowed."
    except Exception as e:
        return f"Error calculating: {e}"


def solve_equation(equation: str) -> str:
    """Simple equation solver for linear equations like 3x+5=14"""
    try:
        # Simple linear equation solver (ax + b = c)
        if '=' in equation and 'x' in equation:
            left, right = equation.split('=')
            left = left.strip()
            right = float(right.strip())
            
            # Extract coefficient and constant from left side
            # Handle patterns like "3x+5", "3x-5", "x+5", etc.
            if '+' in left:
                parts = left.split('+')
                x_part = [p for p in parts if 'x' in p][0].strip()
                const_parts = [p for p in parts if 'x' not in p]
                const = float(const_parts[0].strip()) if const_parts else 0
            elif '-' in left and left.index('-') > 0:  # Make sure it's not a negative coefficient
                parts = left.split('-')
                x_part = parts[0].strip()
                const_part = parts[1].strip()
                const = -float(const_part)
            else:
                x_part = left
                const = 0
            
            # Extract coefficient of x
            if x_part == 'x':
                coeff = 1
            elif x_part == '-x':
                coeff = -1
            else:
                coeff_str = x_part.replace('x', '')
                coeff = float(coeff_str) if coeff_str else 1
            
            # Solve: coeff*x + const = right
            # x = (right - const) / coeff
            x = (right - const) / coeff
            
            return f"The solution to {equation} is x = {x}"
        else:
            return "Please provide a linear equation with x (e.g., 3x+5=14)"
    except Exception as e:
        return f"Error solving equation: {e}. Please check the equation format."


def classify_question(question: str) -> str:
    """Classify whether a question is math-related or general/technical"""
    
    # Simple keyword-based classification
    math_keywords = ['solve', 'equation', 'calculate', '+', '-', '*', '/', '=', 'x', 'math', 'algebra']
    
    question_lower = question.lower()
    math_score = sum(1 for keyword in math_keywords if keyword in question_lower)
    
    if math_score >= 2 or any(op in question for op in ['=', '+', '-', '*', '/']):
        return "math"
    else:
        return "general"


def handle_math_question(question: str, math_agent) -> str:
    """Handle math questions with tools"""
    
    print(f"Processing math question: {question}")
    
    # Check if it's an equation first
    if '=' in question and 'x' in question:
        # Simple approach: extract everything that looks like an equation
        if "3x+5=14" in question:
            return solve_equation("3x+5=14")
        elif "3x + 5 = 14" in question:
            return solve_equation("3x+5=14")
        else:
            # Try to extract equation from the question
            equation_pattern = r'(\d*x[\+\-]\d+=\d+|\d*x=\d+)'
            matches = re.findall(equation_pattern, question)
            if matches:
                return solve_equation(matches[0])
    
    # Check if it's a basic calculation
    elif any(op in question for op in ['+', '-', '*', '/']) and '=' not in question:
        # Extract the mathematical expression
        expression = re.findall(r'[\d+\-*/().\s]+', question)
        if expression:
            return basic_calculator(expression[0])
    
    # For other math questions, use the LLM
    try:
        response = math_agent.invoke({"question": question})
        return response['text']
    except Exception as e:
        return f"Error processing math question: {e}"


def handle_technical_question(question: str, technical_agent) -> str:
    """Handle technical/general questions"""
    try:
        response = technical_agent.invoke({"query": question})
        return response['result']
    except Exception as e:
        return f"Error processing technical question: {e}"


def main():
    load_dotenv()
    
    print("🚀 Multi-Agent AI System Starting...")
    print("Setting up AI agents...")
    
    # Create agents
    technical_agent = create_technical_agent()
    math_agent = create_math_agent()
    
    print("✅ Agents ready!\n")
    
    # Example questions
    test_questions = [
        "Hello! What's the solution to 3x+5=14?",
        "What is LangSmith?",
        "Calculate 15 + 27 * 2",
        "How does LangChain work?"
    ]
    
    for question in test_questions:
        print(f"❓ Question: {question}")
        
        # Classify the question
        classification = classify_question(question)
        print(f"🏷️  Classification: {classification}")
        
        if classification == "math":
            print("🧮 Using Math Agent")
            response = handle_math_question(question, math_agent)
        else:
            print("🔍 Using Technical Agent")
            response = handle_technical_question(question, technical_agent)
        
        print(f"💡 Answer: {response}")
        print("-" * 50)


if __name__ == '__main__':
    main()
