import os
import chainlit as cl

from loguru import logger

from wikichat.vectordb import VectorDB
from wikichat.llm import LLM


cohere_api_key = os.environ['COHERE_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']

if cohere_api_key is None or openai_api_key is None:
    logger.error('API key is not provided; set COHERE_API_KEY and OPENAI_API_KEY environment variables')
    sys.exit(0)

vector_db = VectorDB(
    cohere_api_key=cohere_api_key
)

llm = LLM(
    openai_api_key=openai_api_key
)


@cl.on_message
async def main(message: cl.Message):
    task_list = cl.TaskList()
    task_list.status = 'Running...'

    logger.info(f"Received message: {message.content}")

    task1 = cl.Task(title='Querying vectordb', status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task1)
    await task_list.send()
    results = vector_db.query(message.content)
    texts = '\n'.join([item['text'] for item in results])

    logger.debug(texts)

    task1.status = cl.TaskStatus.DONE
    await task_list.send()

    task2 = cl.Task(title='Querying LLM', status=cl.TaskStatus.RUNNING)
    await task_list.add_task(task2)
    await task_list.send()
    llm_response = llm.apply_prompt(
        prompt_name='qna',
        template_args={
            'question': message.content,
            'documents': texts
        }
    )
    task2.status = cl.TaskStatus.DONE

    task_list.status = 'Done'
    await task_list.send()

    await cl.Message(
        content=llm_response.response,
    ).send()
