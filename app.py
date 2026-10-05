"""Single-process Alice webhook. TLS is terminated by the hosting provider."""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)
PROMPT = os.getenv('SYSTEM_PROMPT', 'Отвечай по-русски, коротко, двумя-тремя предложениями. Ты голосовой помощник. Помогай с компьютерами и 1С. Не используй Markdown. Не выдумывай факты.')
WAIT_SECONDS = 2.5
TTL = 600
MAX_SESSIONS = 100


class Session(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    skill_id: str
    message_id: int = Field(ge=0)
    new: bool = False
    application: dict[str, str] = Field(default_factory=dict)


class Utterance(BaseModel):
    type: str = 'SimpleUtterance'
    command: str = Field(default='', max_length=4096)
    original_utterance: str = Field(default='', max_length=4096)


class AliceRequest(BaseModel):
    session: Session
    request: Utterance
    version: str = '1.0'


@dataclass
class Conversation:
    touched: float = field(default_factory=time.monotonic)
    history: list = field(default_factory=list)
    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_id: int = -1
    last_reply: dict | None = None


def reply(text: str, end: bool = False):
    text = ' '.join(text.split())
    if len(text) > 1024:
        text = text[:1023] + '…'
    # Omit tts: Alice voices text; generated text cannot inject TTS markup.
    return {'response': {'text': text, 'end_session': end}, 'version': '1.0'}


@asynccontextmanager
async def lifespan(app):
    key = os.getenv('OPENAI_API_KEY', '').strip()
    app.state.client = AsyncOpenAI(api_key=key, timeout=30, max_retries=0) if key else None
    app.state.conversations = {}
    yield
    tasks = [c.task for c in app.state.conversations.values() if c.task]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if app.state.client:
        await app.state.client.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


async def generate(conversation, text):
    try:
        result = await app.state.client.responses.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
            instructions=PROMPT,
            input=conversation.history + [{'role': 'user', 'content': text}],
            max_output_tokens=220,
            store=False,
        )
        answer = result.output_text.strip()
        if not answer:
            return 'Не удалось получить ответ. Попробуйте переформулировать вопрос.'
        answer = reply(answer)['response']['text']
        conversation.history = (conversation.history + [
            {'role': 'user', 'content': text},
            {'role': 'assistant', 'content': answer},
        ])[-8:]
        return answer
    except (OpenAIError, asyncio.TimeoutError):
        # Do not log exceptions: they may contain request text or credentials.
        logger.warning('OpenAI request failed')
        return 'Не удалось связаться с OpenAI. Проверьте ключ, доступ к модели и баланс API. Попробуйте ещё раз.'
    except Exception:
        logger.error('Unexpected generation failure')
        return 'Ошибка помощника. Попробуйте ещё раз.'


@app.get('/health')
async def health():
    return {'status': 'ok', 'configured': app.state.client is not None}


@app.post('/alice')
async def alice(body: AliceRequest):
    if body.version != '1.0':
        raise HTTPException(400, 'Unsupported protocol')
    skill_id = os.getenv('ALICE_SKILL_ID', '')
    if skill_id and body.session.skill_id != skill_id:
        raise HTTPException(403, 'Wrong skill')
    allowed = {s.strip() for s in os.getenv('ALLOWED_APPLICATION_IDS', '').split(',') if s.strip()}
    if allowed and body.session.application.get('application_id') not in allowed:
        return reply('Этот помощник доступен только владельцу.', True)
    conversations = app.state.conversations
    now = time.monotonic()
    for sid, old in list(conversations.items()):
        if now - old.touched > TTL and not old.lock.locked():
            if old.task:
                old.task.cancel()
            del conversations[sid]
    sid = body.session.session_id
    if sid not in conversations:
        if len(conversations) >= MAX_SESSIONS:
            return reply('Помощник занят. Попробуйте через минуту.')
        conversations[sid] = Conversation()
    c = conversations[sid]
    # Duplicate/concurrent deliveries must not create a second paid API call.
    if c.lock.locked():
        return reply('Готовлю ответ. Через несколько секунд скажите «готово».')
    async with c.lock:
        c.touched = now
        if body.session.message_id == c.last_id and c.last_reply:
            return c.last_reply
        if body.session.message_id < c.last_id:
            return reply('Повторите последнюю реплику.')
        text = (body.request.command or body.request.original_utterance).strip()
        command = text.lower().rstrip('.!?')
        if command in {'стоп', 'выход', 'хватит', 'закончи'}:
            if c.task:
                c.task.cancel()
            c.task = None
            c.history.clear()
            result = reply('До встречи!', True)
        elif body.request.type != 'SimpleUtterance':
            result = reply('Задайте вопрос голосом.')
        elif c.task:
            if c.task.done():
                result = reply(c.task.result())
                c.task = None
            else:
                result = reply('Ответ ещё готовится. Через несколько секунд скажите «готово».')
        elif command in {'сброс', 'забудь разговор'}:
            c.history.clear()
            result = reply('История очищена. Задайте новый вопрос.')
        elif not text:
            result = reply('Здравствуйте! Я помощник на OpenAI. Задайте вопрос. Если ответ задержится, скажите «готово».')
        elif command in {'готово', 'ответ', 'проверь ответ'}:
            result = reply('Сейчас нет ожидающего ответа. Задайте вопрос.')
        elif not app.state.client:
            result = reply('Помощник ещё не настроен. Добавьте OPENAI_API_KEY в настройки сервера.')
        else:
            c.task = asyncio.create_task(generate(c, text))
            try:
                answer = await asyncio.wait_for(asyncio.shield(c.task), WAIT_SECONDS)
                c.task = None
                result = reply(answer)
            except asyncio.TimeoutError:
                result = reply('Готовлю ответ. Через несколько секунд скажите «готово».')
        c.last_id, c.last_reply = body.session.message_id, result
        return result
