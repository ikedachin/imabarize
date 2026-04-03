import json
import random
import httpx
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from commons.utils_msg import msg_debug, msg_success, msg_error, msg_info

class QAPipeline:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.inference_config = dict(settings.get("infer_config", {}))

        if settings.get("openrouter", False):
            api_key = settings.get("openrouter_api_key", "dummy")
            server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
            model_name = settings.get("openrouter_model_name", None)
            self.inference_config.update(
                {
                    "API_KEY": api_key,
                    "SERVER_URL": server_url,
                    "MODEL_NAME": model_name,
                }
            )

            self.client = OpenAI(
                base_url=self.inference_config.get("SERVER_URL"),
                api_key=self.inference_config.get("API_KEY"),
            )
            self.runtime_label = "openrouter"
        else:
            api_key = "dummy"
            server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
            model_name = settings.get("MODEL_NAME", None)
            self.runtime_label = "local"
            self.inference_config.update(
                {
                    "API_KEY": api_key,
                    "SERVER_URL": server_url,
                    "MODEL_NAME": model_name,
                }
            )
            self.client = OpenAI(
                base_url=self.inference_config.get("SERVER_URL"),
                api_key=self.inference_config.get("API_KEY"),
            )

        self.output_dir = (
            Path(settings.get("output_path", "./json_output/qa"))
            .expanduser()
            .resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.max_retries = int(settings.get("max_retries", 8))
        self.wait_seconds = float(settings.get("wait_seconds", 1.0))

    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

    def list_files(self, source_path: Union[str, Path, None]) -> Tuple[List[Path], List[Path]]:
        if source_path is None:
            return [], []
        source_path = Path(source_path)
        text_extensions = {".md", ".txt"}
        json_extensions = {".json", ".jsonl"}

        if source_path.is_dir():
            text_files = sorted(
                p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in text_extensions
            )
            json_files = sorted(
                p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in json_extensions
            )
            return text_files, json_files

        if source_path.is_file():
            suffix = source_path.suffix.lower()
            if suffix in text_extensions:
                return [source_path], []
            if suffix in json_extensions:
                return [], [source_path]
        return [], []

    def build_md_pair(self, index: int, source_files: List[Path]) -> Tuple[str, List[Path]]:
        first_path = source_files[index]
        second_path = source_files[index + 1] if index + 1 < len(source_files) else None
        first_text = first_path.read_text(encoding="utf-8")
        second_text = second_path.read_text(encoding="utf-8") if second_path else ""
        sources = [first_path]
        if second_path is not None:
            sources.append(second_path)
        return f"{first_text}\n\n{second_text}", sources

    def build_json_pair(
        self, index: int, json_data: List[Dict], target_key: Optional[str]
    ) -> Tuple[str, List[int]]:
        first_item = json_data[index]
        second_item = json_data[index + 1] if index + 1 < len(json_data) else None

        first_text = first_item.get(target_key, "")
        second_text = second_item.get(target_key, "") if second_item else ""
        return f"{first_text}\n\n{second_text}", [index, index + 1] if second_item else [index]

    def _infer_text(self, prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]

        max_tokens = int(self.inference_config.get("max_tokens", 2048))
        temperature = self.inference_config.get("temperature", 0)
        top_p = self.inference_config.get("top_p", 1.0)

        last_exc = None
        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                content = response.choices[0].message.content
                if content is None:
                    return ""
                if isinstance(content, str):
                    return content.strip()
                return str(content).strip()

            except Exception as e:
                last_exc = e

                # ネットワーク系も軽くリトライ（短め）
                if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
                    sleep = min((self.wait_seconds * (2 ** i)) + random.random() * 0.2, 10.0)
                    time.sleep(sleep)
                    continue
            if i == self.max_retries - 1:
                print(msg_error(f"Inference failed after {self.max_retries} attempts. last_error={last_exc}"))
                # raise  # その他は即死

        # raise RuntimeError(f"Inference retry exhausted. last_error={last_exc}")
        return ""


    def _infer_texts(self, prompts: List[str], batch_size: int) -> List[str]:
        if not prompts:
            return []

        max_workers = max(1, min(batch_size, len(prompts)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._infer_text, prompts))
            
    def _extract_tag(self, text: Optional[str], tag: str) -> str:
        if (not text) or (not isinstance(text, str)):
            return ""
        text = str(text)
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if (start_tag in text) and (end_tag in text):
            text = text.split(start_tag)[-1]
            text = text.split(end_tag)[0]
        elif start_tag in text:
            text = text.split(start_tag)[-1]
        elif end_tag in text:
            text = text.split(end_tag)[0]
        return text.strip()

    def create_qa_batch(self, texts: List[str], batch_size: int) -> List[Dict[str, str]]:
        if not texts:
            return []

        question_prompt = self.prompts.get("question_prompt", None)
        thinking_prompt = self.prompts.get("thinking_prompt", None)
        answer_prompt = self.prompts.get("answer_prompt", None)
        eval_prompt = self.prompts.get("eval_prompt", None)
        # refine_thinking_prompt = self.prompts.get("refine_thinking_prompt", None)
        # refine_answer_prompt = self.prompts.get("refine_answer_prompt", None)
        
        # ==========================================================
        # ここでquestion_textを生成する
        # ==========================================================

        
        if not question_prompt or not answer_prompt:
            raise ValueError("question_prompt and answer_prompt must be set in prompts.")

        random_tokens = [
            "".join(random.sample(text, min(len(text), 10))) if text else "" for text in texts
        ]
        question_prompts = [
            question_prompt.format(text=text, random_token=random_token)
            for text, random_token in zip(texts, random_tokens)
        ]

        print(msg_info(f"now generating QUESTION..."))
        question_texts = [
            self._extract_tag(text, "question")
            for text in self._infer_texts(question_prompts, batch_size)
        ]

        # ==========================================================
        # ここでanswer_textを生成する
        # ==========================================================
        answer_prompts = [
            answer_prompt.format(text=text, question=question_text)
            for text, question_text in zip(texts, question_texts)
        ]

        print(msg_info(f"now generating ANSWER..."))
        answer_texts = [
            self._extract_tag(text, "think")
            for text in self._infer_texts(answer_prompts, batch_size)
        ]

        # ==========================================================
        # ここでthinking_textを生成する
        # ==========================================================

        thinking_prompts = [
            thinking_prompt.format(text=text, question=question_text, answer=answer_text)
            for text, question_text, answer_text in zip(texts, question_texts, answer_texts)
        ]
        print(msg_info(f"now generating THINKING..."))
        think_texts = [
            self._extract_tag(text, "think")
            for text in self._infer_texts(thinking_prompts, batch_size)
        ]

        think_texts = [
            self._extract_tag(text, "think")
            for text in think_texts
        ]
        think_texts = [
            self._extract_tag(text, "thinking")
            for text in think_texts
        ]

        # ==========================================================
        # ここでevalを行う
        # ==========================================================
        eval_prompts = [
            eval_prompt.format(think=think_text, answer=answer_text)
            for think_text, answer_text in zip(think_texts, answer_texts)
        ]

        print(msg_info(f"Eval texts now..."))
    
        eval_texts = [
            self._extract_tag(text, "eval")
            for text in self._infer_texts(eval_prompts, batch_size)
        ]



        # ==========================================================
        # データセット化
        # ==========================================================


        results = []
        for question_text, thinking_text, answer_text, eval_text in zip(
            question_texts, think_texts, answer_texts, eval_texts
        ):
            if question_text and answer_text:
                results.append(
                    {
                        "question": question_text,
                        "thinking": thinking_text if thinking_prompt else "",
                        "answer": answer_text,
                        "eval": eval_text if eval_prompt else "",
                        # "refined_thinking": thinking_text if refine_thinking_prompt else "",
                        # "refined_answer": answer_text if refine_answer_prompt else "",
                        "qa_generator": self.inference_config.get("MODEL_NAME", ""),
                    }
                )
            else:
                pass
        return results

    def append_jsonl(self, save_path: Path, result: Dict) -> None:
        keys = result.keys()
        flag = True
        for key in keys:
            if not result[key]:
                flag = False
                break

        if flag:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")

    def add_cache(self, entry_id: str) -> None:
        # キャッシュ用のIDをファイルに保存するなどの実装をここに追加
        save_path = self.output_dir / "cache_ids.txt"
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(f"{entry_id}\n")
