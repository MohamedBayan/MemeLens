#!/usr/bin/env python3


import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv
from openai import AzureOpenAI


def load_env(env_path: str) -> Tuple[str, str, str]:
    load_dotenv(dotenv_path=env_path, override=True)
    api_base = os.environ["AZURE_API_URL"].rstrip("/")
    api_key = os.environ["AZURE_API_KEY"]
    api_version = os.environ["AZURE_API_VERSION"]
    return api_key, api_base, api_version


def load_env_azure_storage(env_path: str):
    load_dotenv(dotenv_path=env_path, override=True)
    api_url = os.environ["AZURE_STORAGE_ACCOUNT_URL"].rstrip("/")
    container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]
    sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    return api_url, sas_token, container_name


def get_container_client(
    storage_container_name, blob_service_client
) -> Optional[ContainerClient]:
    """Get Azure Container Client."""
    if not blob_service_client:
        return None
    container_client = blob_service_client.get_container_client(storage_container_name)

    return container_client


def get_blob_service_client(
    storage_sas_token, storage_url
) -> Optional[BlobServiceClient]:
    """Get Azure Blob Service Client with multiple authentication options."""
    # Option 1: SAS token
    if storage_sas_token:
        sas_token = storage_sas_token.lstrip("?")
        account_url = f"{storage_url}?{sas_token}"
        return BlobServiceClient(account_url=account_url)


def load_tracking(batch_file: Path):
    """
    Load tracking info from JSON list (preferred) or legacy CSV/line format.
    Returns a list of dicts with batch_id and optional output_prefix.
    """
    text = batch_file.read_text(encoding="utf-8")
    if text.strip():
        try:
            data = json.loads(text)
            if isinstance(data, list):
                records = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    bid = item.get("batch_id")
                    if bid:
                        records.append(
                            {
                                "batch_id": str(bid),
                                "output_prefix": item.get("output_prefix", ""),
                            }
                        )
                if records:
                    return records
        except Exception:
            pass
    # Legacy CSV/line
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            batch_id, _ = line.split(",", 1)
        else:
            batch_id = line
        records.append({"batch_id": batch_id.strip(), "output_prefix": ""})
    return records


# def check_batch_output(client: AzureOpenAI, batch_id: str, save_dir: Path) -> Path:
#     resp = client.batches.retrieve(batch_id)
#     if resp.status != "completed":
#         print(f"[WARN] Batch {batch_id} not completed (status={resp.status})")
#         return resp.status

#     return None


def retrieve_batch_output(blob_client, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as file:
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())
    return target_path


def parse_results(
    batch_output_path: Path,
    out_ok: Path,
    out_err: Path,
    original_map: Dict[str, Dict],
    languages: Optional[Set[str]] = None,
):
    """
    Legacy one-file parser (kept for compatibility); prefers collect_all_results for bulk.
    """
    ok = out_ok.open("w", encoding="utf-8")
    err = out_err.open("w", encoding="utf-8")
    total = 0
    ok_count = 0
    err_count = 0

    with batch_output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            # lang_prefix, _ = parse_custom_id(custom_id)
            # if languages and lang_prefix and lang_prefix not in languages:
            #     continue
            resp_data = obj.get("response", {})
            content = (
                resp_data.get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            try:
                parsed = json.loads(content)
            except Exception as e:
                print(f"error :{e}")
                err.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            # "language": lang_prefix,
                            "error": "parse_failed",
                            "raw": content,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err_count += 1
                continue
            rec = {
                "custom_id": custom_id,
                "data": original_map.get(custom_id),
                "labels": parsed,
            }
            ok.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok_count += 1
    ok.close()
    err.close()
    print(f"[STATS] total={total} ok={ok_count} err={err_count}")


def parse_custom_id(custom_id: str) -> Tuple[Optional[str], str]:
    """
    Extract language prefix and id payload from the custom_id.
    Expected format from batch builder: <language>_<conversation_id>.
    """
    if not custom_id or "_" not in custom_id:
        return None, custom_id or ""
    lang, rest = custom_id.split("_", 1)
    return lang, rest


def load_original_map(
    original_file: Path, languages: Optional[Set[str]] = None
) -> Dict[str, Dict]:
    """
    Build a mapping from custom_id (lang_conversation) -> original record.
    Only include records whose language is in `languages` if provided.
    """
    mapping: Dict[str, Dict] = {}
    if not original_file or not original_file.exists():
        return mapping
    with original_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = obj.get("new_id")
            if not custom_id:
                continue
            mapping[str(custom_id)] = obj
    print(f"Number of items in the mapping file: {len(mapping)}")
    return mapping


def load_original_map_json(original_file: Path) -> Dict[str, Dict]:
    """
    Build a mapping from custom_id (lang_conversation) -> original record.
    Only include records whose language is in `languages` if provided.
    """
    mapping: Dict[str, Dict] = {}
    if not original_file or not original_file.exists():
        return mapping

    with original_file.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse original file {original_file}: {e}")
            return mapping
    if isinstance(data, list):
        for obj in data:
            custom_id = obj.get("id")
            if not custom_id:
                continue
            mapping[str(custom_id)] = obj
    print(f"Number of items in the mapping file: {len(mapping)}")
    return mapping


def collect_all_results(out_dir: Path) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    ok_map: Dict[str, Dict] = {}
    err_map: Dict[str, Dict] = {}

    for file in sorted(out_dir.glob("*.jsonl")):
        if not file.name.endswith("_results.jsonl"):
            continue
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                resp_data = obj.get("response", {})
                content = (
                    resp_data.get("body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )

                if not custom_id:
                    # treat missing id as error (keep it grouped under a synthetic key)
                    synthetic_id = f"_missing_custom_id::{file.name}"
                    err_map[synthetic_id] = {
                        "custom_id": None,
                        "error": "missing_custom_id",
                        "raw": content,
                        "source_file": str(file),
                    }
                    continue

                try:
                    parsed = json.loads(content)
                    ok_map[custom_id] = {
                        "custom_id": custom_id,
                        "labels": parsed,
                        "source_file": str(file),
                    }
                    # if it was previously in err_map, clean it up
                    err_map.pop(custom_id, None)
                except Exception as e:
                    print(f"Parsing issue {e}\n content: {content}")
                    err_map[custom_id] = {
                        "custom_id": custom_id,
                        "error": "parse_failed",
                        "raw": content,
                        "source_file": str(file),
                    }
                    # if it was previously in ok_map, clean it up
                    ok_map.pop(custom_id, None)
    print(f"number of ok={len(ok_map)}, err={len(err_map)}")
    return ok_map, err_map


def write_joined_outputs(
    ok_map: Dict[str, Dict],
    err_map: Dict[str, Dict],
    out_ok: Path,
    out_err: Path,
    original_map: Dict[str, Dict],
) -> None:
    """
    Writes final outputs once (no overwriting per-file).
    """
    out_ok.parent.mkdir(parents=True, exist_ok=True)
    out_err.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    err_count = 0

    with out_ok.open("w", encoding="utf-8") as ok_f:
        for custom_id in sorted(ok_map.keys()):
            rec = ok_map[custom_id]
            ok_f.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "data": original_map.get(custom_id),
                        "labels": rec.get("labels"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ok_count += 1

    if err_map:
        with out_err.open("w", encoding="utf-8") as err_f:
            for key in sorted(err_map.keys()):
                rec = err_map[key]
                err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                err_count += 1

    print(f"[STATS] ok={ok_count} err={err_count} total={ok_count + err_count}")
    print(f"Output file: {out_ok}")
    if err_count:
        print(f"Error file: {out_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve and parse persona summary batch results from Azure."
    )
    parser.add_argument(
        "--batch_file",
        required=True,
        help="File with batch IDs (submitted_batches.txt).",
    )
    parser.add_argument(
        "--env_file",
        required=True,
        help="Azure openai api access).",
    )
    parser.add_argument(
        "--env_storage",
        required=True,
        help="Azure storage access).",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Dir to save retrieved batch outputs."
    )
    parser.add_argument(
        "--output_file", required=True, help="JSONL file for parsed successes."
    )
    parser.add_argument(
        "--output_error_file", required=True, help="JSONL file for parse/errors."
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="If set, retrieve batch outputs; otherwise parse existing files in output_dir.",
    )
    parser.add_argument(
        "--original_file",
        type=str,
        default=None,
        help="Optional original personas JSONL for attaching persona content.",
    )

    args = parser.parse_args()

    storage_url, storage_sas_token, storage_container_name = load_env_azure_storage(
        args.env_storage
    )
    api_key, api_base, api_version = load_env(args.env_file)
    client = AzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=api_base
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retrieve:
        blob_service_client = get_blob_service_client(storage_sas_token, storage_url)
        container_client = get_container_client(
            storage_container_name, blob_service_client
        )

        for rec in load_tracking(Path(args.batch_file)):
            batch_id = rec["batch_id"].replace("batch_", "")
            output_prefix = rec.get("output_prefix", "").strip("/")
            batch_id_nohyphen = batch_id.replace("-", "")

            if not output_prefix:
                print(
                    f"[WARN] No output_prefix for batch {batch_id}; skipping download"
                )
                continue
            # Try expected paths: with hyphens and without
            candidate_paths = [
                # f"{output_prefix}/{batch_id}/results.jsonl",
                f"{output_prefix}/{batch_id_nohyphen}/results.jsonl",
            ]
            downloaded = False
            for expected_blob in candidate_paths:
                print(f"expected_blob: {expected_blob}")
                blob_client = container_client.get_blob_client(expected_blob)
                target_path = out_dir / f"{batch_id}_results.jsonl"
                try:
                    # check_batch_output(client, batch_id)
                    resp = client.batches.retrieve(rec["batch_id"])
                    print(f"Batch status: {resp.status}")
                    if resp.status == "completed":
                        print(f"[INFO] Batch {batch_id} (status={resp.status})")
                        retrieve_batch_output(blob_client, target_path)
                        print(f"[INFO] Downloaded {expected_blob} -> {target_path}")
                        downloaded = True
                    else:
                        print(f"[WARN] Batch {batch_id} (status={resp.status})")
                    break
                except Exception as exc:
                    print(
                        f"[WARN] Failed to download expected blob ({expected_blob}): {exc}"
                    )

            if downloaded:
                continue

            # Fallback: search for a results.jsonl that contains the batch_id under the prefix
            matched = [
                b
                for b in container_client.list_blobs(name_starts_with=output_prefix)
                if (batch_id in b.name or batch_id_nohyphen in b.name)
                and b.name.lower().endswith("results.jsonl")
            ]
            if not matched:
                # one more try with a sanitized prefix (strip dots)
                alt_prefix = output_prefix.replace(".", "")
                matched = [
                    b
                    for b in container_client.list_blobs(name_starts_with=alt_prefix)
                    if (batch_id in b.name or batch_id_nohyphen in b.name)
                    and b.name.lower().endswith("results.jsonl")
                ]

            if not matched:
                print(f"[WARN] No matching results.jsonl found for batch {batch_id}")
                continue

            blob = matched[0]
            blob_client = container_client.get_blob_client(blob.name)
            target_path = out_dir / f"{batch_id}_results.jsonl"
            try:
                retrieve_batch_output(blob_client, target_path)
                print(f"[INFO] Downloaded {blob.name} -> {target_path}")
            except Exception as exc:
                print(f"[WARN] Failed to download {blob.name}: {exc}")

    original_map = (
        load_original_map(Path(args.original_file)) if args.original_file else {}
    )

    ok_path = Path(args.output_file)
    err_path = Path(args.output_error_file)

    ok_map, err_map = collect_all_results(out_dir)
    write_joined_outputs(ok_map, err_map, ok_path, err_path, original_map)


if __name__ == "__main__":
    main()
