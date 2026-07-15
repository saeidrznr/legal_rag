import re
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv(override=True)

model = SentenceTransformer('intfloat/multilingual-e5-large')
db = lancedb.connect("legal_rag_db")
table = db.open_table("iran_legal_docs")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    intent: str  # 'law', 'greeting', 'abusive'
    metadata_filter: dict
    contexts: List[str]
    final_answer: str
    retry_count: int = 0

def rewrite_query(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """تو یک دستیار حقوقی هستی. وظیفه تو "فقط" بازنویسی سوال کاربر به زبان رسمی و حقوقی است.

        قوانین:
        1. فقط و فقط متن بازنویسی شده را برگردان.
        2. هیچ مقدمه، تعارف، یا پیشنهادی (مثل "آیا مایل هستید...") اضافه نکن.
        3. اگر سوال مبهم است، آن را به دقیق‌ترین شکل حقوقی شفاف کن.
        4.به هيچ وجه از نيم فاصله استفاده نکن

        سوال کاربر: {query}
        بازنویسی حقوقی:"""
    )

    chain = prompt | llm
    query = state["query"].replace('\u200c', ' ')
    response = chain.invoke({"query": query})
    return {"rewritten_query": response.content}

def classify_intent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "طبقه‌بندی قصد کاربر: فقط یکی از کلمات 'greeting'، 'abusive' یا 'law' را برگردان.\nسوال: {query}"
    )
    intent = llm.invoke(prompt.format(query=state["query"])).content.strip().lower()

    if 'greeting' in intent:
        return {"intent": "greeting", "final_answer": "سلام! خوشحالم که اینجا هستید. چطور می‌توانم در مسائل حقوقی به شما کمک کنم؟"}
    elif 'abusive' in intent:
        return {"intent": "abusive", "final_answer": "من به عنوان یک دستیار هوشمند طراحی شده‌ام تا به شما کمک کنم. لطفاً سوالات خود را با احترام بپرسید."}
    return {"intent": "law"}

def extract_metadata(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
       """تو یک تحلیلگر داده‌های حقوقی هستی. وظیفه تو استخراج اطلاعات ساختاریافته از پرسش کاربر است.

قوانین استخراج:
1. law_name: فقط نام رسمی قانون را برگردان. اگر سوال درباره چک است، بنویس "قانون صدور چک". اگر درباره کار است، بنویس "قانون کار".
2. article_number: فقط شماره ماده را به صورت عدد (Integer) استخراج کن.
3. اگر هر کدام از این موارد در سوال نبود، مقدار آن را حتما null بگذار.
4. خروجی فقط یک JSON خالص باشد.

سوال: {rewritten_query}

خروجی مورد انتظار:
{{
  "law_name": "string or null",
  "article_number": integer or null
}}
"""
    )

    response = llm.invoke(prompt.format(rewritten_query=state["rewritten_query"]))
    try:
        metadata = json.loads(response.content)
    except:
        metadata = {}
    return {"metadata_filter": metadata}


def context_retrieve(state: AgentState):
    query_vector = model.encode(f"query: {state['rewritten_query']}")
    search_query = table.search(query_vector)

    metadata = state.get("metadata_filter", {})

    if metadata and "law_name" in metadata:
        filter_str = f"law_name = '{metadata['law_name']}'"

        try:
            # search with filter
            results = search_query.where(filter_str).limit(10).to_list()
            # search without filter if nothing found
            if not results:
                results = table.search(query_vector).limit(10).to_list()
        except:
            results = table.search(query_vector).limit(10).to_list()
    else:
        results = search_query.limit(10).to_list()

    return {"contexts": [r["text"] for r in results]}


def check_relevance(query: str, contexts: List[str]) -> bool:
    context_text = "\n\n".join(contexts)
    prompt = ChatPromptTemplate.from_template(
        """تو یک ارزیاب دقیق هستی. بررسی کن آیا مستندات ارائه شده شامل اطلاعاتی برای پاسخ به سوال کاربر هستند یا خیر.

        مستندات: {contexts}
        سوال کاربر: {query}

        فقط و فقط یکی از دو کلمه 'yes' (اگر مرتبط است) یا 'no' (اگر مرتبط نیست) را برگردان."""
    )

    chain = prompt | llm
    response = chain.invoke({"query": query, "contexts": context_text})

    result = response.content.strip().lower()
    return "yes" in result


def rerank(state: AgentState):
    contexts = state.get("contexts", [])
    query = state.get("rewritten_query")
    
    if not contexts:
        return {"retry_count": state.get("retry_count", 0) + 1}


    rerank_prompt = ChatPromptTemplate.from_template(
        """تو یک تحلیل‌گر خبره هستی. لیست مستندات زیر را بررسی کن و از بین آن‌ها، حداکثر ۳ مورد که واقعاً برای پاسخ به سوال مفید هستند را انتخاب کن.
        
        سوال کاربر: {query}
        
        مستندات:
        {numbered_contexts}
        
        فقط و فقط شماره موارد انتخاب شده را به صورت یک لیست پایتونی برگردان. مثال: [0, 2]"""
    )

    numbered_contexts = "\n".join([f"{i}: {c[:300]}..." for i, c in enumerate(contexts)])
    
    chain = rerank_prompt | llm
    response = chain.invoke({"query": query, "numbered_contexts": numbered_contexts})
    
    try:
        import ast
        selected_indices = ast.literal_eval(re.search(r'\[.*\]', response.content).group())
        top_contexts = [contexts[i] for i in selected_indices if i < len(contexts)]
    except:
        top_contexts = contexts[:3]


    is_relevant = check_relevance(query, top_contexts)

    if not is_relevant and state.get("retry_count", 0) < 1:
        return {"retry_count": 1, "contexts": []}

    state["contexts"] = top_contexts

    return {"contexts": top_contexts}


def generate_answer(state: AgentState):
    context_text = "\n\n".join(state["contexts"])
    prompt = ChatPromptTemplate.from_template(
        "با استفاده از مستندات زیر به سوال پاسخ دقیق بده:\nمستندات: {context}\nسوال: {query}"
    )
    response = llm.invoke(prompt.format(context=context_text, query=state["rewritten_query"]))
    return {"final_answer": response.content,"contexts":state["contexts"]}


workflow = StateGraph(AgentState)

workflow.add_node("rewrite", rewrite_query)
workflow.add_node("classify", classify_intent)
workflow.add_node("metadata", extract_metadata)
workflow.add_node("retrieve", context_retrieve)
workflow.add_node("rerank", rerank)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "classify")



def after_classify_condition(state):
    if state["intent"] == "law":
        return "Law"
    return "Greeting or Abusive"


workflow.add_conditional_edges(
    "classify", 
    after_classify_condition,
    {
        "Law": "metadata",
        "Greeting or Abusive": END
    }
)
workflow.add_edge("metadata", "retrieve")
workflow.add_edge("retrieve", "rerank")

# repeat search
def after_rerank_condition(state):
    if not state["contexts"] and state.get("retry_count", 0) == 1:
        return "Max 1 Retry"
    return "generate"


workflow.add_conditional_edges(
    "rerank", 
  after_rerank_condition,
    {
        "Max 1 Retry": "retrieve",
        "generate": "generate"
    }
)
workflow.add_edge("generate", END)

# graph compile
legal_graph = workflow.compile()