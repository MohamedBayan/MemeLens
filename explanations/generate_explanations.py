#!/usr/bin/env python3


import argparse
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

from config import (
    DATA_PATH,
    DATASET_CONFIG,
    ENV_FILE,
    LANGUAGE_MAP,
    OUTPUT_PATH,
    SKIP_DATASETS,
    TASK_DEFINITIONS,
)

GPT_DICT = {
    "MODEL_NAME": "gpt-4.1-global-batch",
    "CHAT_REQUEST_URL": "/chat/completions",
    "MAX_TOKENS": 8000,
}


def load_env(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    api_base = os.environ["AZURE_API_URL"].rstrip("/")
    api_key = os.environ["AZURE_API_KEY"]
    api_version = os.environ["AZURE_API_VERSION"]
    deployment = os.environ.get("AZURE_ENGINE_NAME", GPT_DICT["MODEL_NAME"])
    return api_key, api_base, api_version, deployment


def load_env_azure_storage(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    api_url = os.environ["AZURE_STORAGE_ACCOUNT_URL"].rstrip("/")
    container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]
    sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    return api_url, sas_token, container_name


class BatchBuilder:
    def __init__(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        batch_file_size_limit: int = 190 * 1024 * 1024,
        data_size_limit: int = 2 * 1024 * 1024,
    ):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_file_size_limit = batch_file_size_limit
        self.data_size_limit = data_size_limit

    def get_system_prompt(self, dataset_name: str, config: dict) -> str:
        """Generate system prompt for explanation generation."""
        language = LANGUAGE_MAP.get(config["language"], "English")
        labels = ", ".join(config["labels"])
        native_labels = ", ".join(config.get("native_labels", config["labels"]))
        is_english = config["language"] == "en"

        # Get task definition from JSON file
        task_definition = TASK_DEFINITIONS.get(dataset_name, "")

        if is_english:
            response_format = """
            {
                "en_explanation": "A clear, comprehensive explanation in English (4-6 sentences) for why this meme was classified with the given label. Reference specific visual elements from the image and text content that support the classification."
            }"""
            guidelines = """Guidelines:
            - Be objective and analytical
            - Reference BOTH visual elements from the image AND text content
            - Explain how image and text work together to convey the classification
            - Keep explanations clear and informative (4-6 sentences)
            - Provide detailed reasoning that connects visual and textual elements to the assigned label
            - Do not include any text outside the JSON object"""
        else:
            response_format = f"""
            {{
                "en_explanation": "A clear, comprehensive explanation in English (4-6 sentences) for why this meme was classified with the given label. Reference specific visual elements from the image and text content that support the classification.",
                "native_explanation": "The same explanation in {language}."
            }}"""
        guidelines = f"""Guidelines:
            - Be objective and analytical
            - Reference BOTH visual elements from the image AND text content
            - Explain how image and text work together to convey the classification
            - Keep explanations clear and informative (4-6 sentences)
            - Provide detailed reasoning that connects visual and textual elements to the assigned label
            - Ensure the native explanation accurately conveys the same meaning as the English explanation
            - Do not include any text outside the JSON object"""

        # Build the system prompt with task definition
        task_definition_section = f"\n\n{task_definition}" if task_definition else ""
        prompt = f"""You are an expert annotator for multimodal meme analysis. Your task is to provide explanations for meme classifications.{task_definition_section}

                Task: {config["task"]}
                Description: {config["description"]}
                Labels: {labels}
                {f"Native labels ({language}): {native_labels}" if config.get("native_labels") else ""}

                For each meme, you will receive:
                1. The image of the meme
                2. The text content from the meme (OCR extracted or embedded text)
                3. The assigned classification label

                Your response must be a valid JSON object:{response_format}

                {guidelines}"""
        return prompt

    def get_user_prompt(self, sample: dict, config: dict) -> list:
        """Generate user prompt for a single sample with image."""
        language = LANGUAGE_MAP.get(config["language"], "English")

        text = sample.get("text", "").strip()
        if not text:
            text = "[No text content in meme]"

        label = sample.get("class_label", sample.get("label", "unknown"))
        img_path = sample.get("img_path", sample.get("image_path", ""))

        # Encode image to base64
        try:
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Determine image format
            ext = os.path.splitext(img_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")

            image_url = f"data:{mime_type};base64,{img_data}"
        except Exception as e:
            print(f"Warning: Failed to encode image {img_path}: {e}")
            image_url = ""

        # Build message content with image
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high",  # Use high detail for best quality (~765 tokens per image)
                },
            },
            {
                "type": "text",
                "text": f"""Analyze this meme to explain why it was classified as: {label}

    Meme Text: "{text}"

    Explain how the visual elements (imagery, expressions, symbols, colors) combined with the text content justify this classification. {'Provide your explanation in English only.' if config['language'] == 'en' else f'Provide your explanation in both English and {language}.'}""",
            },
        ]

        return content

    def load_records(self):
        """
        Stream records from .jsonl or .json and de-duplicate by obj["data_id"].

        - Keeps the FIRST occurrence of each data_id.
        - If data_id is missing/empty, the record is yielded (not deduped).
        - Works in streaming mode for JSONL; for JSON list it iterates the list.
        """
        suffix = self.input_path.suffix.lower()
        seen_ids = set()

        def _maybe_yield(obj, where: str, idx: int | None = None):
            if obj is None:
                return

            # Only dedupe when data_id exists and is non-empty
            data_id = obj.get("new_id") if isinstance(obj, dict) else None
            if data_id:
                if data_id in seen_ids:
                    # Optional: log once in a while or at debug level
                    logger.debug(
                        "Skipping duplicate data_id=%s (%s%s)",
                        data_id,
                        where,
                        f":{idx}" if idx is not None else "",
                    )
                    return
                seen_ids.add(data_id)

            yield obj

        if suffix == ".jsonl":
            with self.input_path.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping invalid JSONL line %d in %s: %s",
                            lineno,
                            self.input_path,
                            e,
                        )
                        continue

                    # yield via helper generator
                    yield from _maybe_yield(obj, where="jsonl", idx=lineno)
            return

        if suffix == ".json":
            with self.input_path.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON file %s: %s", self.input_path, e)
                    return

            if isinstance(data, list):
                for i, obj in enumerate(data):
                    yield from _maybe_yield(obj, where="json", idx=i)
            elif isinstance(data, dict):
                yield from _maybe_yield(data, where="json")
            else:
                logger.warning(
                    "Unsupported JSON root type in %s: %s",
                    self.input_path,
                    type(data).__name__,
                )
            return

        raise ValueError(
            f"Unsupported input format: {self.input_path} (expected .jsonl or .json)"
        )

    def make_messages(self, user_prompt: str, system_prompt: str) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def calc_size(self, obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def build_batches(self, dataset_name):
        current = []
        current_size = 0
        batch_idx = 1
        total_tasks = 0
        logger.info("Building batches from %s -> %s", self.input_path, self.output_dir)

        config = DATASET_CONFIG[dataset_name]

        for rec in tqdm(self.load_records()):
            data_id = str(rec.get("new_id"))

            user_prompt = self.get_user_prompt(rec, config)
            system_prompt = self.get_system_prompt(dataset_name, config)

            if self.calc_size(user_prompt + system_prompt) > self.data_size_limit:
                logger.warning(
                    "Skipping conversation_id=%s because size %d exceeds limit %d",
                    data_id,
                    self.calc_size(user_prompt + system_prompt),
                    self.data_size_limit,
                )
                continue

            custom_id = data_id
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": GPT_DICT["CHAT_REQUEST_URL"],
                "body": {
                    "model": GPT_DICT["MODEL_NAME"],
                    "messages": self.make_messages(user_prompt, system_prompt),
                    "max_tokens": GPT_DICT["MAX_TOKENS"],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                },
            }

            task_size = self.calc_size(task)
            if current_size + task_size > self.batch_file_size_limit:
                self.save_batch(current, batch_idx)
                batch_idx += 1
                current = []
                current_size = 0

            current.append(task)
            current_size += task_size
            total_tasks += 1

        if current:
            self.save_batch(current, batch_idx)
        logger.info(
            "Finished building batches. Total tasks: %d, total batches: %d",
            total_tasks,
            batch_idx,
        )

    def save_batch(self, tasks: list, idx: int):
        out_file = self.output_dir / f"batch_{idx}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        logger.info("Saved batch %d with %d tasks -> %s", idx, len(tasks), out_file)


class AzureBatchManager:
    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        api_version: str,
        deployment_name: str,
        track_file: Union[str, Path],
        azure_task_dir: str,
        task_data_sudir: str,
        storage_url: str,
        storage_container_name: str,
        storage_sas_token: Optional[str] = None,
    ):
        #### openai api related
        self.client = AzureOpenAI(
            api_key=api_key, api_version=api_version, azure_endpoint=api_endpoint
        )
        self.track_file = Path(track_file)
        self.track_file.parent.mkdir(parents=True, exist_ok=True)
        self.azure_task_dir = azure_task_dir
        self.task_data_sudir = task_data_sudir
        GPT_DICT["MODEL_NAME"] = deployment_name

        ##### Storage related
        self.storage_container_name = storage_container_name
        self.storage_sas_token = storage_sas_token
        self.storage_url = storage_url
        self.blob_service_client = self.get_blob_service_client()
        self.container_client = self.get_container_client(
            self.storage_container_name, self.blob_service_client
        )

    def get_blob_service_client(self) -> Optional[BlobServiceClient]:
        """Get Azure Blob Service Client with multiple authentication options."""
        # Option 1: SAS token
        if self.storage_sas_token:
            sas_token = self.storage_sas_token.lstrip("?")
            account_url = f"{self.storage_url}?{sas_token}"
            return BlobServiceClient(account_url=account_url)

        # Option 2: DefaultAzureCredential (managed identity, Azure CLI, etc.)
        try:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            return BlobServiceClient(
                account_url=self.storage_url, credential=credential
            )
        except Exception as e:
            print(f"Failed to use DefaultAzureCredential: {e}")

        return None

    def get_container_client(
        self, storage_container_name, blob_service_client
    ) -> Optional[ContainerClient]:
        """Get Azure Container Client."""
        if not blob_service_client:
            return None
        container_client = blob_service_client.get_container_client(
            storage_container_name
        )

        return container_client

    def parse_blob_path(
        self, path: str, storage_container_name: str
    ) -> tuple[str, str]:
        """
        Parse a blob path in format 'container/blob_path' or just 'blob_path'.
        Returns (container_name, blob_name).
        """
        if path.startswith("az://") or path.startswith("azure://"):
            path = path.split("://", 1)[1]

        parts = path.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return storage_container_name, parts[0]

    def is_blob_path(self, path: str) -> bool:
        """Check if path is an Azure Blob Storage path."""
        return (
            path.startswith("az://")
            or path.startswith("azure://")
            or (
                self.blob_service_client is not None
                and not Path(path).exists()
                and "/" in path
            )
        )

    def upload_string_to_blob(self, blob_name: str, content: str) -> None:
        """Upload string content to blob."""
        if not self.container_client:
            raise RuntimeError("Azure Blob Storage not configured")

        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(content.encode("utf-8"), overwrite=True)
        print(f"Uploaded to blob: {blob_name}")

    def upload_file_to_blob(self, blob_name: str, file_path: Path) -> None:
        """Upload a local file to blob."""
        if not self.container_client:
            raise RuntimeError("Azure Blob Storage not configured")

        blob_client = self.container_client.get_blob_client(blob_name)
        with open(file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
        print(f"Uploaded file to blob: {blob_name}")

    def get_blob_url_with_sas(self, container_name: str, blob_name: str) -> str:
        """Get the full URL for a blob, including SAS token if available."""
        base_url = f"{self.storage_url}/{container_name}/{blob_name}"
        if self.storage_sas_token:
            sas_token = self.storage_sas_token
            if sas_token.startswith("?"):
                sas_token = sas_token[1:]
            return f"{base_url}?{sas_token}"
        return base_url

    def _load_existing(self) -> Dict[str, Dict[str, str]]:
        """
        Returns mapping of resolved batch file paths to metadata from track file.
        Supports legacy CSV lines and JSON list formats.
        """
        mapping: Dict[str, Dict[str, str]] = {}
        if not self.track_file.exists():
            return mapping
        content = self.track_file.read_text(encoding="utf-8").strip()
        if not content:
            return mapping

        # Try JSON list format first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    file_path = item.get("local_path") or item.get("file_path")
                    if not file_path:
                        continue
                    mapping[Path(file_path).resolve().as_posix()] = {
                        "batch_id": item.get("batch_id", ""),
                        "file_id": item.get("file_id", ""),
                        "blob_path": item.get("blob_path", ""),
                        "output_prefix": item.get("output_prefix", ""),
                    }
                return mapping
        except Exception:
            pass  # fall back to CSV

        # Legacy CSV (batch_id,file_path,file_id,blob_path)
        for line in content.splitlines():
            line = line.strip()
            if not line or "," not in line:
                continue
            parts = [p.strip() for p in line.split(",")]
            batch_id = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""
            file_id = parts[2] if len(parts) > 2 else ""
            blob_path = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
            mapping[Path(file_path).resolve().as_posix()] = {
                "batch_id": batch_id,
                "file_id": file_id,
                "blob_path": blob_path,
                "output_prefix": "",
            }
        return mapping

    def _save_tracking(self, mapping: Dict[str, Dict[str, str]]) -> None:
        """Persist tracking metadata as JSON list."""
        records = []
        for local_path, meta in sorted(mapping.items()):
            records.append(
                {
                    "batch_id": meta.get("batch_id", ""),
                    "local_path": local_path,
                    "file_id": meta.get("file_id", ""),
                    "blob_path": meta.get("blob_path", ""),
                    "output_prefix": meta.get("output_prefix", ""),
                }
            )
        self.track_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def submit_batch(self, file_path: Path, use_blob: bool = True) -> Dict[str, Any]:
        # 1) Upload to Azure Blob Storage (mandatory when use_blob is True)
        blob_url = ""
        uploaded = None
        if use_blob:
            if not (self.blob_service_client and self.container_client):
                raise RuntimeError(
                    "Azure Blob Storage not configured but use_blob=True"
                )
            blob_name = f"{self.azure_task_dir}/{self.task_data_sudir}/{file_path.name}"
            self.upload_file_to_blob(blob_name, file_path)
            blob_url = self.get_blob_url_with_sas(
                self.storage_container_name, blob_name
            )
            logger.info("Uploaded batch file to blob: %s", blob_name)

        # 2) Upload file to OpenAI (optional when using blob, but keeps file_id for tracking)
        else:
            try:
                with file_path.open("rb") as bf:
                    uploaded = self.client.files.create(
                        file=bf,
                        purpose="batch",
                        extra_body={
                            "expires_after": {
                                "seconds": 2592000,
                                "anchor": "created_at",
                            },
                        },
                    )
                logger.info("Uploaded file_id=%s for %s", uploaded.id, file_path.name)
            except Exception as exc:
                logger.warning(
                    "OpenAI file upload failed (continuing with blob input only): %s",
                    exc,
                )

        # 3) Create batch using blob URL as input
        extra_body = {
            # API expects a direct URL string to the blob with SAS
            "input_blob": blob_url,
        }
        batch_results_prefix = (
            f"{self.azure_task_dir}/{self.task_data_sudir}/batch_results/"
        )
        extra_body["output_folder"] = {
            "url": self.get_blob_url_with_sas(
                self.storage_container_name, batch_results_prefix
            )
        }

        batch_job = self.client.batches.create(
            input_file_id=None,
            endpoint=GPT_DICT["CHAT_REQUEST_URL"],
            completion_window="24h",
            extra_body=extra_body,
        )
        print(f"Batch created: {batch_job.id}, status: {batch_job.status}")

        return {
            "batch": batch_job,
            "file_id": uploaded.id if uploaded else batch_job.id,
            "blob_path": f"{blob_url.split('?')[0]}",
            "output_folder": extra_body["output_folder"]["url"],
            "output_prefix": batch_results_prefix,
        }

    def submit_all(self, batch_dir: Path):
        logger.info("Submitting all batches in %s", batch_dir)
        existing = self._load_existing()
        for file in sorted(batch_dir.glob("*.jsonl")):
            key = file.resolve().as_posix()
            if key in existing and file.exists():
                logger.info(
                    "Skipping already-submitted batch file %s (batch_id=%s)",
                    file,
                    existing[key].get("batch_id"),
                )
                continue
            result = self.submit_batch(file)
            batch_job = result["batch"]
            existing[key] = {
                "batch": batch_job,
                "batch_id": batch_job.id,
                "file_id": result.get("file_id", ""),
                "blob_path": result.get("blob_path", ""),
                "output_prefix": result.get("output_prefix", ""),
            }
            self._save_tracking(existing)
        logger.info("Finished submitting all batches in %s", batch_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build and submit Azure batch requests for persona summarization."
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL/TSV/CSV with persona objects."
    )
    parser.add_argument("--dataset_name", required=True, help="name of the dataset.")
    parser.add_argument(
        "--env_storage",
        required=True,
        help="Azure storage access).",
    )
    parser.add_argument(
        "--env_file",
        required=True,
        help="Azure env file (AZURE_API_URL, AZURE_API_KEY, AZURE_API_VERSION, AZURE_ENGINE_NAME).",
    )
    parser.add_argument(
        "--azure_data_dir",
        default="cached_dir",
        required=False,
        help="Directory to write batch JSONL files on Azure.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write batch JSONL files."
    )
    parser.add_argument(
        "--batch_file", required=True, help="Path to track submitted batch IDs."
    )
    parser.add_argument(
        "--no_submit",
        action="store_true",
        help="If set, only build batches without submission.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    storage_url, storage_sas_token, storage_container_name = load_env_azure_storage(
        args.env_storage
    )
    api_key, api_base, api_version, deployment = load_env(args.env_file)

    logger.info("Loaded env; model=%s; endpoint=%s", deployment, api_base)

    builder = BatchBuilder(args.input, args.output_dir)
    builder.build_batches()

    if args.no_submit:
        logger.info("Batches built; submission skipped (--no_submit).")
        return

    manager = AzureBatchManager(
        api_key,
        api_base,
        api_version,
        deployment,
        args.batch_file,
        args.azure_data_dir,
        args.output_dir,
        storage_url,
        storage_container_name,
        storage_sas_token,
    )
    manager.submit_all(Path(args.output_dir))


if __name__ == "__main__":
    main()
