import chainlit as cl
from legal_agent import legal_graph 

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", legal_graph)
    
    await cl.Message(content="سلام! من دستیار حقوقی شما هستم. سوال خود را بپرسید تا با بررسی قوانین پاسخ دهم.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    
    initial_state = {
        "query": message.content
    }

    msg = cl.Message(content="در حال پردازش")
    await msg.send()


    final_state = await cl.make_async(agent.invoke)(initial_state)

    # extract answer
    answer = final_state.get("final_answer", "متاسفانه پاسخی برای این سوال پیدا نکردم.")
    intent = final_state.get("intent", "law")

    # updating final answer
    msg.content = answer
    await msg.update()

    # showing user intent
    if intent != 'law':
        await cl.Message(content=f"*(تشخیص سیستم: این یک پیام از نوع {intent} بود)*").send()