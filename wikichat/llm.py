import os
import openai
import tiktoken

from typing import Optional
from loguru import logger

from wikichat.models import LLMResponse


class LLM:
    MODEL = 'gpt-4-1106-preview'
    ENCODING_NAME = 'cl100k_base'
    TOKEN_LIMIT = 128000
    BUFFER_TOKENS = 25
    PROMPT_DIR = os.path.abspath(
        os.path.join(
            os.path.abspath('.'), 'data', 'prompts'
        )
    )

    def __init__(self, openai_api_key: str) -> None:
        if not openai_api_key:
            logger.error('OpenAI API key is not provided')
            raise ValueError("API key must be provided")

        openai.api_key = openai_api_key

        try:
            openai.Model.list()
        except Exception as err:
            logger.error(f'Error connecting to OpenAI: {err}')
            raise

        logger.success('Loaded OpenAI API key')

    def num_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.get_encoding(self.ENCODING_NAME)
            return len(encoding.encode(text))
        except Exception as err:
            logger.error(f'Error retrieving encoding: {err}')
            return 0

    def _truncate_text(self, text: str, excess_tokens: int) -> str:
        encoding = tiktoken.get_encoding(self.ENCODING_NAME)
        tokens = list(encoding.encode(text))
        truncated_tokens = tokens[:-excess_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text

    def _read_prompt(self, prompt_name: str) -> Optional[str]:
        path = os.path.join(self.PROMPT_DIR, f'{prompt_name}.txt')
        if os.path.exists(path):
            with open(path, 'r') as fp:
                return fp.read()
        logger.error(f'Prompt not found: {path}')
        return None

    def _call_openai(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        logger.info('Calling OpenAI with user prompt')
        system_prompt = system_prompt or 'You are a helpful AI assistant.'

        system_tokens = self.num_tokens(system_prompt)
        user_tokens = self.num_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        while total_tokens > self.TOKEN_LIMIT:
            logger.warning(f'Token limit exceeded, truncating to {self.TOKEN_LIMIT} (cnt: {total_tokens})')
            excess_tokens = total_tokens - self.TOKEN_LIMIT + self.BUFFER_TOKENS
            user_prompt = self._truncate_text(user_prompt, excess_tokens)
            user_tokens = self.num_tokens(user_prompt)
            total_tokens = system_tokens + user_tokens

        logger.info(f'Token count: {total_tokens} / {self.TOKEN_LIMIT} ({(total_tokens / self.TOKEN_LIMIT) * 100:.2f}%)')

        try:
            params = {
                'model': self.MODEL,
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            }
            response = openai.ChatCompletion.create(**params)
            return response.choices[0].message['content']
        except Exception as err:
            logger.error(f'Error calling OpenAI: {err}')
            return None

    def apply_prompt(self, prompt_name: str, template_args: dict) -> Optional[LLMResponse]:
        template = self._read_prompt(prompt_name)

        if template:
            formatted_prompt = template.format(**template_args)
            response = self._call_openai(user_prompt=formatted_prompt)
            if response:
                return LLMResponse(
                    response=response,
                    prompt_name=prompt_name
                )
        return None