# -*- coding: utf-8 -*-
import base64
import mimetypes
import os
import time
import json
import traceback
from typing import Union, List, Any, Dict, Optional

import numpy as np
from openai import OpenAI

from utils.constants import *  # NOTE: make sure this file does NOT hardcode secrets!


TOTAL_INPUT_TOKENS: int = 0
TOTAL_OUTPUT_TOKENS: int = 0


# -----------------------------
# Helpers: OpenAI client
# -----------------------------
def _get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    Create OpenAI client using (priority):
      1) explicit args
      2) env vars: OPENAI_API_KEY / OPENAI_BASE_URL
      3) fallback: None (OpenAI SDK will error with clear message)
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return OpenAI(api_key=key, base_url=url)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _to_image_url_payload(path_or_url: str) -> dict:
    """Return an image_url payload for OpenAI chat: local files -> data: URL, else pass through."""
    if os.path.exists(path_or_url):
        mime, _ = mimetypes.guess_type(path_or_url)
        if not mime:
            mime = "image/jpeg"
        with open(path_or_url, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    else:
        return {"type": "image_url", "image_url": {"url": path_or_url}}


# -----------------------------
# VLM APIs
# -----------------------------
def vlm_generate_multi(
    prompt: str = "Describe this image in one sentence.",
    image: Union[str, List[str]] = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    model: str = "gpt-4.1",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Supports single or multiple images.
    - `image` can be a string (path/URL) or a list of strings.
    - Local paths are embedded as data: URLs.
    - Returns chat message content (string).
    """
    images = image if isinstance(image, list) else [image]
    content = [{"type": "text", "text": prompt}]
    content.extend(_to_image_url_payload(img) for img in images)

    client = _get_openai_client(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return (resp.choices[0].message.content or "").strip()


def vlm_generate(
    prompt: str = "Describe this image in one sentence.",
    image: str = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    other_content: Optional[str] = None,
    json_format: bool = False,
    *,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    VLM single image call.
    - Supports json_format via response_format={"type":"json_object"}
    - Uses env OPENAI_API_KEY / OPENAI_BASE_URL by default.
    """
    if os.path.exists(image):
        image = f"data:image/jpeg;base64,{encode_image(image)}"

    client = _get_openai_client(api_key=api_key, base_url=base_url)

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    if other_content:
        content.append({"type": "text", "text": other_content})

    if json_format:
        chat_response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
        )
        return json.loads(chat_response.choices[0].message.content)

    chat_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
    )
    return chat_response.choices[0].message.content


def vlm_generate_fewshot(
    prompt: str = "Describe this image in one sentence.",
    image: str = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    json_format: bool = False,
    *,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Few-shot VLM. NOTE: This still ships example images from local path.
    Make sure those example assets are safe to publish.
    """
    if os.path.exists(image):
        image = f"data:image/jpeg;base64,{encode_image(image)}"

    basic_fpath = "../examples/table_images"
    image1 = f"data:image/jpeg;base64,{encode_image(os.path.join(basic_fpath, 'example_1.png'))}"
    image2 = f"data:image/jpeg;base64,{encode_image(os.path.join(basic_fpath, 'example_2.png'))}"
    image3 = f"data:image/jpeg;base64,{encode_image(os.path.join(basic_fpath, 'example_3.png'))}"

    output1 = {
        "single_subtable": "yes",
        "has_row_header": "yes",
        "row_header_groups": [
            {
                "group": "Segment",
                "children": [
                    {"group": "Innovation Systems", "children": ["Product", "Service"]},
                    {"group": "Mission Systems", "children": ["Product", "Service"]},
                ],
            }
        ],
        "column_header_groups": [
            {
                "group": "Years Ended December 31,",
                "children": [
                    {"group": "2018", "children": ["Sales", "Expenses"]},
                    {"group": "2017", "children": ["Sales", "Expenses"]},
                ],
            }
        ],
        "summary_text": "Total Expenses for 2017 are 0.",
    }
    output2 = {
        "single_subtable": "yes",
        "has_row_header": "no",
        "row_header_groups": [],
        "column_header_groups": [
            {"group": None, "children": ["Segment"]},
            {"group": "2018", "children": ["Funded"]},
            {"group": "2017", "children": ["Funded", "% Change"]},
        ],
        "summary_text": "",
    }
    output3 = {"single_subtable": "no"}

    client = _get_openai_client(api_key=api_key, base_url=base_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Example 1 input:"},
                {"type": "image_url", "image_url": {"url": image1}},
                {"type": "text", "text": "Example 1 output:"},
                {"type": "text", "text": json.dumps(output1, ensure_ascii=False)},
                {"type": "text", "text": "Example 2 input:"},
                {"type": "image_url", "image_url": {"url": image2}},
                {"type": "text", "text": "Example 2 output:"},
                {"type": "text", "text": json.dumps(output2, ensure_ascii=False)},
                {"type": "text", "text": "Example 3 input:"},
                {"type": "image_url", "image_url": {"url": image3}},
                {"type": "text", "text": "Example 3 output:"},
                {"type": "text", "text": json.dumps(output3, ensure_ascii=False)},
                {"type": "text", "text": "Now process this table:"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]

    if json_format:
        chat_response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return json.loads(chat_response.choices[0].message.content)

    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content


def vlm_generate_fewshot_setup(
    prompt: str,
    image: str,
    json_format: bool = False,
    single_tab: bool = True,  # kept for backward compatibility; no longer used
    *,
    model: str = "gpt-4.1",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Minimal VLM call:
    - No few-shot examples
    - Send (prompt + image) and return model output + token usage
    """
    if os.path.exists(image):
        image = f"data:image/jpeg;base64,{encode_image(image)}"

    client = _get_openai_client(api_key=api_key, base_url=base_url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Now process this table:"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]

    if json_format:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        text = response.choices[0].message.content or ""
        token_info = _extract_token_usage(getattr(response, "usage", None))
        return {
            "text": json.loads(text),
            "input_tokens": token_info["input_tokens"],
            "output_tokens": token_info["output_tokens"],
            "total_tokens": token_info["total_tokens"],
        }

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    token_info = _extract_token_usage(getattr(response, "usage", None))
    return {
        "text": text,
        "input_tokens": token_info["input_tokens"],
        "output_tokens": token_info["output_tokens"],
        "total_tokens": token_info["total_tokens"],
    }


# -----------------------------
# LLM APIs
# -----------------------------
def llm_generate(
    prompt,
    key=LLM_API_KEY,
    url=LLM_API_URL,
    model=LLM_MODEL_TYPE,
    max_tokens=8192,
    temperature=0.3,
):
    """
    NOTE: key/url/model default comes from utils.constants.
    For publishing: make sure constants.py does NOT contain real secrets.
    Prefer setting OPENAI_API_KEY / OPENAI_BASE_URL env vars and passing key=None/url=None.
    """
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS

    # If key/url are placeholders or empty, allow env fallback:
    api_key = key or os.getenv("OPENAI_API_KEY")
    base_url = url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    res = "None"
    cnt = 0
    while cnt < 20:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant skilled in handling tabular data."},
                {"role": "user", "content": prompt},
            ]

            kwargs = dict(
                model=model,
                temperature=temperature,
                stream=False,
                top_p=1,
            )

            if str(model).startswith("gpt-5"):
                kwargs["reasoning"] = {"effort": "none"}
                kwargs["input"] = messages
                response = client.responses.create(**kwargs)
                res = getattr(response, "output_text", "")
            else:
                kwargs["messages"] = messages
                kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(**kwargs)
                res = response.choices[0].message.content

            token_usage = _extract_token_usage(getattr(response, "usage", None))
            TOTAL_INPUT_TOKENS += token_usage["input_tokens"]
            TOTAL_OUTPUT_TOKENS += token_usage["output_tokens"]
            break

        except Exception:
            print(f"LLM API Request Failed! Retry {cnt}!")
            traceback.print_exc()
            time.sleep(0.2)
            cnt += 1

    return res


def get_llm_usage() -> Dict[str, int]:
    return {
        "total_input_tokens": TOTAL_INPUT_TOKENS,
        "total_output_tokens": TOTAL_OUTPUT_TOKENS,
    }


def _extract_token_usage(usage):
    """
    Normalize the OpenAI usage object (dict or Pydantic model) into:
      {input_tokens, output_tokens, total_tokens}
    """
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input", 0)
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output", 0)
        total_tokens = usage.get("total_tokens") or usage.get("total", None)
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
    else:
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def llm_generate_setup(
    prompt: str,
    model: str,
    key=LLM_API_KEY,
    url=LLM_API_URL,
    max_tokens: int = 8192,
    temperature: float = 0.9,
    max_retries: int = 2,
    json_format: bool = False,
) -> Dict[str, Any]:
    """
    Call an OpenAI model (Chat or Responses API depending on GPT version).
    NOTE: key/url defaults come from utils.constants. Prefer env vars for publishing.
    """
    api_key = key or os.getenv("OPENAI_API_KEY")
    base_url = url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    if json_format:
        sys_msg = (
            "You are a helpful assistant skilled in handling tabular data. "
            "You MUST respond with a single valid JSON object only. "
            "Do NOT wrap it in markdown or code fences. "
            "Do NOT include ```json, ``` or any explanation text. "
            "The output must be directly parseable by Python json.loads."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in handling tabular data."},
            {"role": "user", "content": prompt},
        ]

    last_error = None
    for attempt in range(max_retries):
        try:
            if str(model).startswith("gpt-5"):
                response = client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    reasoning={"effort": "none"},
                )
                text = getattr(response, "output_text", "") or ""
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
            else:
                if json_format:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=1,
                        response_format={"type": "json_object"},
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=1,
                        stream=False,
                    )
                text = response.choices[0].message.content if response.choices else ""
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", 0)
                output_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

            return {
                "text": text,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "model": model,
                "retries": attempt,
            }

        except Exception as e:
            last_error = e
            print(f"LLM API Request Failed! Retry {attempt}! ({type(e).__name__}: {e})")
            traceback.print_exc()
            time.sleep(0.3)

    return {
        "text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model": model,
        "error": str(last_error),
    }


# -----------------------------
# Embedding
# -----------------------------
def embedding_generate(
    input_texts: list,
    key=EMBEDDING_API_KEY,
    url=EMBEDDING_API_URL,
    model=EMBEDDING_MODEL_TYPE,
    dimensions=1024,
):
    """
    NOTE: key/url defaults come from utils.constants.
    For publishing: ensure constants.py has NO real secrets.
    """
    api_key = key or os.getenv("OPENAI_API_KEY")
    base_url = url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    embeddings = []
    for i in range(0, len(input_texts), 10):
        inputs = input_texts[i : i + 10]

        cnt = 0
        while cnt < 20:
            try:
                response = client.embeddings.create(
                    model=model,
                    input=inputs,
                    dimensions=dimensions,
                )
                res = json.loads(response.model_dump_json())["data"]
                embeddings.extend([x["embedding"] for x in res])
                break
            except Exception:
                print(f"EMBEDDING API Request Failed! Retry {cnt}!")
                traceback.print_exc()
                time.sleep(0.1)
                cnt += 1

    return np.array(embeddings)


def main():
    print(embedding_generate(["123", "345", "678"]))


if __name__ == "__main__":
    main()