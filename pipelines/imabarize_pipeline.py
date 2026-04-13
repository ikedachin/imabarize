import json
import math
import time
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success

class SanitizePipeline:
    """テキストのサニタイゼーション（翻訳による浄化）を行うパイプラインクラス。
    
    日本語テキストを英語に翻訳し、再度日本語に翻訳することで、
    テキストの品質を向上させるパイプラインを提供します。
    """
    
    def __init__(self, settings: Dict):
        """SanitizePipelineを初期化します。
        
        Args:
            settings: パイプラインの設定を含む辞書。
                以下のキーを含むことができます：
                - openrouter: OpenRouterを使用するかどうか (bool)
                - openrouter_api_key: APIキー (str)
                - openrouter_server_url: サーバーURL (str)
                - openrouter_model_name: モデル名 (str)
                - SERVER_URL: ローカルサーバーURL (str)
                - MODEL_NAME: モデル名 (str)
                - infer_config: 推論設定 (dict)
                - output_path: 出力先パス (str)
                - prompts: プロンプト設定のリスト (List[Dict])
        """
        self.settings = settings
        if settings.get("openrouter", False):
            api_key = settings.get("openrouter_api_key", "dummy")
            server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
            model_name = settings.get("openrouter_model_name", None)
            self.runtime_label = "openrouter"
        else:
            api_key = "dummy"
            server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
            model_name = settings.get("MODEL_NAME", None)
            self.runtime_label = "local"

        self.inference_config = dict(settings.get("infer_config", {}))
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
        output_path = settings.get("output_path") or "./json_output/qa"
        self.output_dir = (
            Path(output_path)
            .expanduser()
            .resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.batch_size = settings.get("batch_size", 4)
        self.max_retries = settings.get("max_retries", 3)
        self.wait_seconds = settings.get("wait_seconds", 5)
        self.nothink = settings.get("NOTHINK", False)


    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        """プロンプトファイルを読み込んで辞書形式で返します。
        
        Args:
            prompts_settings: プロンプトファイルのパス設定のリスト。
                各要素は {key: filepath} の形式の辞書。
        
        Returns:
            キーとプロンプト文字列のマッピング辞書。
        """
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            if prompt_path and Path(prompt_path).is_file():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompts_dict[key] = f.read()
        return prompts_dict

    def _infer_text(self, prompt: str) -> str:
        """単一のプロンプトに対して推論を実行します。
        
        Args:
            prompt: 推論に使用するプロンプト文字列。
        
        Returns:
            モデルの推論結果のテキスト。
        """
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[{"role": "user", "content":  [{"type": "text", "text": prompt}]}],
                    max_tokens=self.inference_config.get("max_tokens", 2048),
                    temperature=self.inference_config.get("temperature", 0),
                    top_p=self.inference_config.get("top_p", 1.0),
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": not self.nothink
                        }
                    }
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error during inference: {e}. Retrying...")
                time.sleep(self.wait_seconds)
        raise RuntimeError("Max retries exceeded for inference.")

    def _infer_texts(self, prompts: List[str]) -> List[str]:
        """複数のプロンプトに対して並列で推論を実行します。
        
        Args:
            prompts: 推論に使用するプロンプト文字列のリスト。
        
        Returns:
            各プロンプトに対するモデルの推論結果のリスト。
            空のリストが入力された場合は空のリストを返します。
        """
        if not prompts:
            return []
        
        max_workers = len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._infer_text, prompts))

    def tfidf_cosine_similarity(
        self,
        text_a: str,
        text_b: str,
        ngram_range: Tuple[int, int] = (1, 2),
        analyzer: str = "char",
    ) -> float:
        text_a = text_a or ""
        text_b = text_b or ""
        """n-gram TF-IDF と cosine similarity で2文の類似度を算出します。

        Args:
            text_a: 比較対象テキストA
            text_b: 比較対象テキストB
            ngram_range: n-gram の最小・最大 (min_n, max_n)
            analyzer: "char" または "word"

        Returns:
            cosine similarity (0.0 - 1.0)

        1.0: 完全に一致（または非常に類似）
        0.5～0.9: 高い類似性
        0.3～0.5: 中程度の類似性
        0.0～0.3: 低い類似性
        0.0: 全く類似していない、または比較不可能

        特殊ケース
        片方または両方のテキストが空の場合: 0.0
        n-gramが生成できない場合: 0.0
        ベクトルのノルムが0の場合: 0.0

        計算の仕組み
        n-gram抽出: 各テキストから文字または単語のn-gramを抽出

        デフォルト: 1-gram と 2-gram の文字ベース
        TF-IDF重み付け:

        TF (Term Frequency): 各n-gramの出現頻度
        IDF (Inverse Document Frequency): 希少性の重み
        コサイン類似度: 2つのTF-IDFベクトル間の角度から類似度を算出

        """
        def _ngrams(text: str, n: int) -> List[str]:
            if analyzer == "word":
                tokens = text.split()
                if len(tokens) < n:
                    return []
                return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            if len(text) < n:
                return []
            return [text[i : i + n] for i in range(len(text) - n + 1)]

        def _tfidf_vector(text: str, idf: Dict[str, float]) -> Dict[str, float]:
            counts: Dict[str, int] = {}
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in _ngrams(text, n):
                    counts[gram] = counts.get(gram, 0) + 1
            if not counts:
                return {}
            return {term: freq * idf.get(term, 0.0) for term, freq in counts.items()}

        texts = [text_a or "", text_b or ""]
        doc_terms: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for text in texts:
            counts: Dict[str, int] = {}
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in _ngrams(text, n):
                    counts[gram] = counts.get(gram, 0) + 1
            doc_terms.append(counts)
            for term in counts.keys():
                df[term] = df.get(term, 0) + 1

        if not df:
            return 0.0

        n_docs = len(texts)
        idf = {term: (math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0) for term, freq in df.items()}

        vec_a = _tfidf_vector(text_a, idf)
        vec_b = _tfidf_vector(text_b, idf)
        if not vec_a or not vec_b:
            return 0.0

        dot = sum(vec_a.get(term, 0.0) * vec_b.get(term, 0.0) for term in vec_a.keys())
        norm_a = sum(v * v for v in vec_a.values()) ** 0.5
        norm_b = sum(v * v for v in vec_b.values()) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def imabarize_batch(self, batched_data: list[dict]) -> List[Dict]:
        """バッチ処理でテキストの翻訳サニタイゼーションを実行します。
        
        日本語→英語→日本語の翻訳を通じてテキストを浄化し、
        オプションでリファインと評価を行います。
        
        Args:
            data: 入力データ(辞書)。
        
        Returns:
            処理結果を含む辞書のリスト。
        
        Raises:
            ValueError: jp_en_promptまたはen_jp_promptが設定されていない場合。
        """
        # print(msg_debug(batched_data))

        # --------------------
        # プロンプトの取得
        # --------------------
        imabarize_prompt = self.prompts.get("imabarize_prompt", None)

        if not imabarize_prompt:
            raise ValueError("imabarize_prompt must be set in prompts for translation.") 
        
        # --------------------
        # ターゲットキーの設定
        # --------------------

        if self.settings.get("target_key"):
            target_key = self.settings["target_key"]
            if isinstance(target_key, str):
                if "," in target_key:
                    target_keys = [key.strip() for key in target_key.split(",")]
                else:
                    target_keys = [target_key]
        else:
            print(msg_error(f"Target key must be set in settings."))
            sys.exit(1)

        imabarized = {}
        for key in target_keys:
            if key not in batched_data[0]:
                print(msg_error(f"Target key '{key}' not found in data. Available keys: {list(batched_data[0].keys())}"))
                continue
            print(msg_info(f"Sanitizing key: {target_key}"))
            # 今治語化プロンプトの生成と推論
            imabarize_prompts = [imabarize_prompt.format(text=data[target_key]) for data in batched_data]
            imabarized_texts = self._infer_texts(imabarize_prompts)
            imabarized[key] = imabarized_texts
            print(msg_debug(f"Imabarize: {imabarized_texts[0][:100]}..."))


        if self._check_result_length(imabarized):
            print(msg_info("All generated texts have consistent lengths."))
            generator_name = self.inference_config.get("MODEL_NAME") or "unknown"

            for key in target_keys:
                for i, imabarized_text in enumerate(imabarized[key]):
                    batched_data[i][key] = imabarized_text
                    batched_data[i]['generator'] = generator_name
        return batched_data

    def _check_result_length(self, imabarized: dict) -> bool:
        """生成されたテキストが指定された最大長を超えていないかを確認し、必要に応じてトリミングします。
        
        Args:
            imabarized: 生成されたテキストの辞書。
            original: 元のテキストのリスト。
            max_length: 許容される最大文字数。

        Returns:
            データ数が全て一致しているかどうかを判定。
        """
        check_results = True
        result_length = None
        for key, texts in imabarized.items():
            if not isinstance(texts, list):
                check_results = False
                break
            if result_length is None:
                result_length = len(texts)
            elif len(texts) != result_length:
                check_results = False
                break
        return check_results

    def save_results(self, data: Dict) -> None:
        """処理結果をJSONL形式で保存します（1件単位）。        
        Args:
            data: 保存する辞書。
        """
        if not data:
            return
        result_path = self.output_dir / f"imabarized_{self.settings.get('source_path', 'unknown')}.jsonl"
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

