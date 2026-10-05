import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import app as service

def payload(text='', message=0, sid='session-1'):
    return {'version': '1.0', 'session': {'session_id': sid, 'skill_id': 'test', 'message_id': message, 'new': message == 0}, 'request': {'type': 'SimpleUtterance', 'command': text}}

def setup(monkeypatch, key=''):
    monkeypatch.setenv('OPENAI_API_KEY', key)
    monkeypatch.delenv('ALICE_SKILL_ID', raising=False)
    monkeypatch.delenv('ALLOWED_APPLICATION_IDS', raising=False)

def test_greeting_missing_key_exit_and_validation(monkeypatch):
    setup(monkeypatch)
    with TestClient(service.app) as client:
        assert client.get('/health').json()['configured'] is False
        assert client.post('/alice', json=payload()).json()['version'] == '1.0'
        assert 'OPENAI_API_KEY' in client.post('/alice', json=payload('вопрос', 1)).json()['response']['text']
        assert client.post('/alice', json=payload('выход', 2)).json()['response']['end_session']
        assert client.post('/alice', json={}).status_code == 422

def test_answer_history_isolation_dedup_and_truncation(monkeypatch):
    setup(monkeypatch, 'test-key')
    with TestClient(service.app) as client:
        create = AsyncMock(return_value=SimpleNamespace(output_text='Я' * 2000))
        service.app.state.client.responses.create = create
        data = payload('первый вопрос', 1)
        response = client.post('/alice', json=data).json()
        assert len(response['response']['text']) == 1024
        assert client.post('/alice', json=data).json() == response
        assert create.call_count == 1
        client.post('/alice', json=payload('уточнение', 2))
        assert len(create.call_args.kwargs['input']) == 3
        client.post('/alice', json=payload('другой пользователь', 1, 'other'))
        assert len(create.call_args.kwargs['input']) == 1
        assert create.call_args.kwargs['store'] is False

def test_slow_answer_survives_webhook_timeout(monkeypatch):
    setup(monkeypatch, 'test-key')
    monkeypatch.setattr(service, 'WAIT_SECONDS', 0.01)
    async def slow(**kwargs):
        await asyncio.sleep(0.08)
        return SimpleNamespace(output_text='Готовый ответ')
    with TestClient(service.app) as client:
        create = AsyncMock(side_effect=slow)
        service.app.state.client.responses.create = create
        assert 'готово' in client.post('/alice', json=payload('вопрос', 1)).json()['response']['text']
        assert 'готовится' in client.post('/alice', json=payload('готово', 2)).json()['response']['text']
        client.portal.call(asyncio.sleep, 0.1)
        assert client.post('/alice', json=payload('готово', 3)).json()['response']['text'] == 'Готовый ответ'
        assert create.call_count == 1

def test_api_error_and_skill_restriction(monkeypatch):
    from openai import APIConnectionError
    import httpx
    setup(monkeypatch, 'test-key')
    monkeypatch.setenv('ALICE_SKILL_ID', 'test')
    with TestClient(service.app) as client:
        service.app.state.client.responses.create = AsyncMock(side_effect=APIConnectionError(request=httpx.Request('POST', 'https://example.com')))
        assert 'Не удалось' in client.post('/alice', json=payload('вопрос', 1)).json()['response']['text']
        data = payload('вопрос', 2)
        data['session']['skill_id'] = 'wrong'
        assert client.post('/alice', json=data).status_code == 403

def test_personal_access_and_reset(monkeypatch):
    setup(monkeypatch)
    monkeypatch.setenv('ALLOWED_APPLICATION_IDS', 'owner')
    with TestClient(service.app) as client:
        data = payload()
        assert client.post('/alice', json=data).json()['response']['end_session']
        data['session']['application'] = {'application_id': 'owner'}
        assert not client.post('/alice', json=data).json()['response']['end_session']
        c = service.app.state.conversations['session-1']
        c.history = [{'role': 'user', 'content': 'old'}]
        data['session']['message_id'] = 1
        data['request']['command'] = 'сброс'
        client.post('/alice', json=data)
        assert c.history == []
