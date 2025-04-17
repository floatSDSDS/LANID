import json
import time

import httpx
from functools import partial

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from typing import List
import openai

try:
    from typing import Literal, Final, Annotated, TypedDict, Protocol
except Exception:
    from typing_extensions import Literal, Final, Annotated, TypedDict, Protocol

class TextGenViaApi:
    def __init__(self,
                 openai_api_key: str = None,
                 huggingface_authorize_token: str = None,
                 default_temperature: float = 1.0,
                 default_p_value: float = 1.0,
                 default_max_new_tokens: int = 256,
                 ):
        self.config = {}
        self._HUGGING_FACE_MODELS = (
            'gpt2', 'gpt2-xl', 'gpt-j-6b', 'bloom-7b', 'palmyra-3b', 'flan-t5-xxl',
            'unifiedqa-11b', 'blenderbot-9b','bloomz')
        self._OPEN_AI_MODELS = ('gpt-3.5-turbo', 'gpt-4-0314', 'gpt-4')
        self.huggingface_authorize_token = huggingface_authorize_token
        self.openai_key = openai_api_key

        for model_name in self._HUGGING_FACE_MODELS:
            self.config[model_name] = self.huggingface_official_init(huggingface_authorize_token,
                                                                     model_name,
                                                                     t=default_temperature,
                                                                     p=default_p_value,
                                                                     max_new_tokens=default_max_new_tokens)
        for model_name in self._OPEN_AI_MODELS:
            self.config[model_name] = self.openai_init(model_name,
                                                       access_key=self.openai_key,
                                                       t=default_temperature,
                                                       p=default_p_value,
                                                       max_new_tokens=default_max_new_tokens,
                                                       )
    def huggingface_official_init(self,
                                  authorize_token: str,
                                  model_name: str,
                                  t: float = 1.0,
                                  p: float = 1.0,
                                  use_cache=True,
                                  wait_for_model=True,
                                  max_new_tokens=1024
                                  ):
        if model_name == 'gpt2':
            url = "https://api-inference.huggingface.co/models/gpt2"
        elif model_name == 'gpt2-xl':
            url = "https://api-inference.huggingface.co/models/gpt2-xl"
        elif model_name == 'gpt-j-6b':
            url = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
        elif model_name == 'bloom-7b':
            url = "https://api-inference.huggingface.co/models/bigscience/bloom-7b1"
        elif model_name == 'palmyra-3b':
            url = "https://api-inference.huggingface.co/models/Writer/palmyra-3B"
        elif model_name == 'flan-t5-xxl':
            url = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
        elif model_name == 'unifiedqa-11b':
            url = "https://api-inference.huggingface.co/models/allenai/unifiedqa-v2-t5-11b-1363200"
        elif model_name == 'bloomz':
            url = "https://api-inference.huggingface.co/models/bigscience/bloomz"
        elif model_name == 'blenderbot-9b':
            url = "https://api-inference.huggingface.co/models/hyunwoongko/blenderbot-9B"
        else:
            raise NotImplementedError

        headers = {"Authorization": f"Bearer {authorize_token}",
                   "Content-Type": "application/json"}
        body = {'inputs': None,
                'parameters': {
                    'max_new_tokens': max_new_tokens,
                    'num_return_sequences': 1,
                    'do_sample': True,
                    'temperature': t,
                    'num_beams': 1,
                    'top_p': p,
                    'top_k': None,
                    'return_full_text': False},
                'options': {'use_cache': use_cache,
                            'wait_for_model': wait_for_model
                            }
                }
        return {'headers': headers, 'body': body, 'url': url, 'timeout': None}

    def openai_init(self,
                       model_name: str,
                       access_key: str,
                       t: float = 1.0,
                       p: float = 1.0,
                       max_new_tokens: int = 1024):
        """
        maxTokens: The maximum number of tokens to generate in the completion.
        The token count of your prompt plus max_tokens cannot exceed the model's context length.
        """
        headers = {'Access-Key': access_key,
                   'Content-Type': 'application/json',
                   'Accept-Encoding': 'identity'
                   }
        body = {"maxTokens": max_new_tokens,
                "model": model_name,
                "temperature": t,
                "topP": p,
                "stop": None,
                "presencePenalty": 0,
                "frequencyPenalty": 0}
        if model_name in self._OPEN_AI_MODELS:
            body['prompt'] = None
            body['messages'] = None
        else:
            raise NotImplementedError
        return {'headers': headers, 'body': body, 'timeout': None}

    def _prepare_gen(self, text, turbo_system_text, model_name, temperature, top_p_value, max_new_tokens, stop):

        if model_name in self._HUGGING_FACE_MODELS:
            assert self.huggingface_authorize_token is not None

        if model_name in self._OPEN_AI_MODELS:
            assert self.openai_key is not None

        request_body = self.config[model_name]['body']
        request_header = self.config[model_name]['headers']

        # set_temperature and p_value
        if temperature is not None:
            if model_name in list(self._OPEN_AI_MODELS):
                request_body['temperature'] = temperature
            elif model_name in self._HUGGING_FACE_MODELS:
                request_body['parameters']['temperature'] = temperature
            else:
                raise NotImplementedError

        if top_p_value is not None:
            if model_name in self._OPEN_AI_MODELS:
                request_body['topP'] = top_p_value
            elif model_name in self._HUGGING_FACE_MODELS:
                request_body['parameters']['top_p'] = top_p_value
            else:
                raise NotImplementedError

        if max_new_tokens is not None:
            if model_name in self._OPEN_AI_MODELS:
                request_body['maxTokens'] = max_new_tokens
            elif model_name in self._HUGGING_FACE_MODELS:
                request_body['parameters']['max_new_tokens'] = max_new_tokens
            else:
                raise NotImplementedError

        if stop is not None:
            if model_name in self._OPEN_AI_MODELS:
                request_body['stop'] = [stop]
            else:
                raise NotImplementedError

        if model_name in self._OPEN_AI_MODELS:
            if turbo_system_text is None:
                request_body['messages'] = [
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            else:
                request_body['messages'] = [
                    {"role": "system",
                     "content": turbo_system_text
                     },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
        elif model_name in self._HUGGING_FACE_MODELS:
            request_body['inputs'] = text
        else:
            raise NotImplementedError
        return request_header, request_body

    def llm_gen(self,
                text: str,
                model_name: str,
                turbo_system_text: str = None,
                request_sleep: float = 0.0,
                time_out: int = 5,
                max_try: int = 3,
                temperature: float = None,
                top_p_value: float = None,
                max_new_tokens: int = None,
                verbose: int = 1,
                return_time_gap: bool = False,
                stop: str = None
                ):
        # Use a different library: Consider using a different HTTP library like aiohttp
        # which has built-in support for asynchronous requests and is better suited for use with multiprocessing.
        t1 = time.time()

        # single turn
        request_header, request_body = self._prepare_gen(text,
                                                         turbo_system_text,
                                                         model_name,
                                                         temperature,
                                                         top_p_value,
                                                         max_new_tokens,
                                                         stop)
        status_code = None
        try_count = 0
        response = None

        while status_code != 200:

            if try_count >= max_try:
                break

            try:
                try_count += 1
                if model_name in self._FXNLPR_MODELS:
                    with httpx.Client() as client:
                        raw_response = client.get(self.config[model_name]['url'],
                                                  params={"kwargs": str(request_body)},
                                                  headers=request_header,
                                                  timeout=time_out)

                else:
                    with httpx.Client() as client:
                        if verbose >= 2:
                            print(f"Request head: {json.dumps(request_header)}")
                            print(f"Request body: {json.dumps(request_body)}")
                            print(f"Request url: {self.config[model_name]['url']}")
                        # print(f"resquest! {os.getpid()}, time: {datetime.datetime.now()}")
                        raw_response = client.post(self.config[model_name]['url'],
                                                   json=request_body,
                                                   headers=request_header,
                                                   timeout=time_out)

                response = raw_response.json()
                status_code = raw_response.status_code
            except Exception as e:
                if verbose >= 2:
                    print(
                        f"Try count: {try_count}, exception: {e}, status_code: {status_code},"
                        f" response text: {raw_response.text}")
                    print(e)
                continue
            else:
                if model_name in self._OPEN_AI_MODELS:
                    detail = response.get('detail', None)
                    if detail is not None:
                        # response = response['detail']['choices'][0]
                        # response = response['text']
                        response = response['message']['content']
                    else:
                        if verbose >= 1:
                            print(f"Detail is None. Response: {raw_response.text}")
                        continue
                elif model_name in self._HUGGING_FACE_MODELS:
                    if status_code == 200:
                        response = response[0]['generated_text']
                else:
                    raise NotImplementedError

            if status_code != 200 and verbose >= 1 and response is not None:
                print(f"Try count: {try_count}, response: {raw_response.text}")

            time.sleep(request_sleep)

        if status_code != 200:
            response = None

        t2 = time.time()
        if return_time_gap:
            return response, t2 - t1
        else:
            return response

    def llm_gen_concurrent(self,
                           texts: List[str],
                           model_name: str,
                           turbo_system_texts: List[str] = None,
                           max_batch: int = 1024,
                           **kwargs):
        if len(texts) == 0:
            return []

        max_workers = kwargs.get('max_workers', 10)
        return_time_gap = kwargs.get('return_time_gap', False)
        verbose = kwargs.get('verbose', 1)
        if verbose == 0:
            tqdm_disable = True
        else:
            tqdm_disable = False

        kwargs.pop('max_workers')

        total_step = len(texts) // max_batch + 1
        responses = []

        if turbo_system_texts is not None:
            assert len(turbo_system_texts) == len(texts)
        else:
            turbo_system_texts = [None for _ in texts]

        t1 = time.time()
        # print(f"Total step: {total_step}")
        for step_i in tqdm(range(total_step), colour='blue', leave=False, desc=model_name, disable=tqdm_disable):
            batch_texts = texts[step_i * max_batch:(step_i + 1) * max_batch]
            batch_model_names = [model_name for _ in batch_texts]
            batch_turbo_system_texts = turbo_system_texts[step_i * max_batch:(step_i + 1) * max_batch]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_responses = list(
                    tqdm(executor.map(partial(self.llm_gen, **kwargs), batch_texts, batch_model_names,
                                      batch_turbo_system_texts),
                         total=len(batch_texts), desc=f'Step: {step_i}', disable=tqdm_disable, leave=False))
                responses.extend(batch_responses)
        t2 = time.time()
        avg_response_time = (t2 - t1) / len(responses)

        if return_time_gap:
            none_count = len([x for x in responses if x[0] is None])
        else:
            none_count = len([x for x in responses if x is None])

        if kwargs['verbose'] >= 2:
            print("-" * 78)
            print(f"kwargs: {kwargs}")
            print("-" * 78)
            print(f"None ratio: {none_count}/{len(responses)}")
            print(f"total_response_time / total_sample: {avg_response_time}")
            print("-" * 78)

        assert len(responses) == len(texts)

        if return_time_gap:
            responses, t_gaps = list(zip(*responses))
            return responses, t_gaps
        else:
            return responses
