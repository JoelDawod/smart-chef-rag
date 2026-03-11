import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = api_key

# Global dictionary to maintain model state across requests
ai_engine = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting server and loading models into memory...")
    
    # Initialize LLM
    ai_engine["llm"] = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    # Initialize local embedding model
    ai_engine["embedder"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder="./local_models", 
        model_kwargs={'device': 'cpu'}
    )
    
    # Connect to local ChromaDB
    ai_engine["db"] = Chroma(
        persist_directory="./Scrapping and indexing/recipe_vector_db",
        embedding_function=ai_engine["embedder"]
    )
    
    # Define translation pipeline (Dialect to MSA)
    translation_prompt = PromptTemplate.from_template(
        """أنت مساعد ذكي. قم بتحويل سؤال الطبخ التالي المكتوب بالعامية إلى لغة عربية فصحى دقيقة ومناسبة للبحث في قاعدة بيانات. اكتب جملة البحث المترجمة فقط.
        السؤال الأصلي: {question}
        سؤال البحث بالفصحى:"""
    )
    ai_engine["translator"] = translation_prompt | ai_engine["llm"] | StrOutputParser()
    
    # Define RAG generation pipeline
    generation_prompt = PromptTemplate.from_template(
        """أنت شيف محترف وخبير طهي. استخدم الوصفات المتاحة أدناه فقط للإجابة على سؤال المستخدم.
        
        القواعد الصارمة (يجب الالتزام بها حرفياً):
        1. التزم تماماً بالمعلومات المذكورة في الوصفات المرفقة فقط. لا تخترع أو تضف أي مقادير أو خطوات من خارج النص أبداً.
        2. أجب دائماً بلغة عربية فصحى سليمة، واضحة، ومهنية. يُمنع منعاً باتاً استخدام اللهجة المصرية أو أي لهجات عامية.
        3. إذا لم تجد إجابة لسؤال المستخدم أو لم تكن المكونات متطابقة مع الوصفات المتاحة، اعتذر بتهذيب وأخبره أنك لا تملك وصفة لذلك.
        
        يجب أن يكون هيكل إجابتك ثابتاً ومقسماً كالتالي:
        
        (مقدمة ترحيبية قصيرة ومفيدة باللغة العربية الفصحى)
        
        **المقادير:**
        - (استخرج المكونات واكتبها مع كمياتها الدقيقة في شكل نقاط)
        
        **طريقة التحضير:**
        1. (اكتب الخطوات بالترتيب في قائمة مرقمة وواضحة)
        
        ---
        المصدر: (اكتب اسم الوصفة التي استخدمتها من السياق)
        
        السؤال: {question}
        
        الوصفات المتاحة (Context):
        {context}
        
        إجابة الشيف:"""
    )
    ai_engine["generator"] = generation_prompt | ai_engine["llm"] | StrOutputParser()
    
    print("Models loaded successfully. Server ready.")
    yield 
    
    print("Shutting down server and clearing memory...")
    ai_engine.clear()

app = FastAPI(lifespan=lifespan)

class RecipeRequest(BaseModel):
    query: str

@app.post("/ask-chef")
def ask_chef(request: RecipeRequest):
    user_query = request.query
    print(f"[NEW REQUEST] Query: '{user_query}'")
    
    # 1. Translate dialect to MSA for better retrieval
    msa_query = ai_engine["translator"].invoke({"question": user_query})
    
    # 2. Vector search
    results = ai_engine["db"].similarity_search(msa_query, k=3)
    context_text = ""
    for i, doc in enumerate(results):
        context_text += f"\n--- وصفة {i+1}: {doc.metadata['title']} ---\n"
        context_text += f"{doc.page_content}\n"
        
    # 3. Generate augmented response
    final_answer = ai_engine["generator"].invoke({
        "question": user_query,
        "context": context_text
    })
    
    return {"status": "success", "answer": final_answer}