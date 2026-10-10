import itertools
import json
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import ray

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from slime.utils.types import MultimodalTypes, Sample

from .timer import Timer

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(
            f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s",
                    path, row_slice)
        reader = itertools.islice(
            reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def _normalize_prompt_data_paths(path) -> list[str]:
    if isinstance(path, str):
        return [path]
    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            raise ValueError("prompt dataset path list cannot be empty.")
        if not all(isinstance(p, str) for p in path):
            raise TypeError("All prompt dataset paths must be strings.")
        return list(path)
    raise TypeError(f"Unsupported prompt dataset path type: {type(path)}")


def _normalize_prompt_dataset_names(dataset_names, n_paths: int) -> list[str] | None:
    if dataset_names is None:
        return None
    if isinstance(dataset_names, str):
        names = [dataset_names]
    elif isinstance(dataset_names, (list, tuple)):
        names = list(dataset_names)
    else:
        raise TypeError(
            f"Unsupported prompt dataset name type: {type(dataset_names)}")

    if len(names) != n_paths:
        raise ValueError(
            "The number of prompt dataset names must match the number of prompt dataset paths: "
            f"{len(names)} != {n_paths}"
        )
    if not all(isinstance(name, str) for name in names):
        raise TypeError("All prompt dataset names must be strings.")

    return names


def _build_prompt_dataset_tags(paths: list[str], dataset_names: list[str] | None = None) -> list[str]:
    if dataset_names is not None:
        normalized_names = [name.strip() or Path(_parse_generalized_path(path)[
            0]).stem or "dataset" for name, path in zip(dataset_names, paths, strict=True)]
        return normalized_names

    used_tags: dict[str, int] = {}
    tags: list[str] = []

    for p in paths:
        real_path, _ = _parse_generalized_path(p)
        base_tag = Path(real_path).stem or "dataset"
        suffix = used_tags.get(base_tag, 0)
        used_tags[base_tag] = suffix + 1
        tag = base_tag if suffix == 0 else f"{base_tag}_{suffix}"
        tags.append(tag)

    return tags


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)[
                "input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from slime.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(
                    sample.prompt, processor)
                processor_output = processor(
                    text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)[
            "input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(
        f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodal_data = data.get(data_key)
                if multimodal_data is not None:
                    multimodals[mt.placeholder] = (mt, list(multimodal_data))

        pattern = "(" + "|".join(re.escape(p)
                                 for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        assert len(content) > 0, (
                            f"Not enough {mt.name} data: more '{mt.placeholder}' placeholders in prompt "
                            f"than {mt.name}s provided in data"
                        )
                        content_list.append(
                            {"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning(
                    "message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

        for placeholder, (mt, remaining) in multimodals.items():
            assert len(remaining) == 0, (
                f"Multimodal data count mismatch: {len(remaining)} more {mt.name}(s)"
                f"than '{placeholder}' placeholders in prompt"
            )

    return prompt


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        dataset_names=None,
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        ):
        origin_samples = []
        all_paths = _normalize_prompt_data_paths(path)
        normalized_names = _normalize_prompt_dataset_names(
            dataset_names, len(all_paths))
        dataset_tags = _build_prompt_dataset_tags(all_paths, normalized_names)
        for data_path, source_tag in zip(all_paths, dataset_tags, strict=True):
            for data in read_file(data_path):
                # Both chat templates and multimodal inputs require conversation format (list of message dicts)
                as_conversation = apply_chat_template or (
                    multimodal_keys is not None)
                prompt = _build_messages(
                    data, prompt_key, as_conversation, multimodal_keys)

                metadata = dict(data.get(metadata_key) or {})
                if "progress_percent" in data and "progress_percent" not in metadata:
                    metadata["progress_percent"] = data["progress_percent"]
                metadata.setdefault("prompt_dataset", source_tag)
                metadata.setdefault("prompt_dataset_path",
                                    _parse_generalized_path(data_path)[0])
                tools = None
                if tool_key is not None and tool_key in data:
                    tools = data[tool_key]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    elif isinstance(tools, np.ndarray):
                        tools = tools.tolist()
                    assert isinstance(
                        tools, list), f"tools must be a list, got {type(tools)} instead"
                    metadata["tools"] = tools

                if apply_chat_template:
                    output_prompt = tokenizer.apply_chat_template(
                        prompt,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=True,
                        **(apply_chat_template_kwargs or {}),
                    )
                else:
                    output_prompt = prompt

                if processor:
                    from slime.utils.processing_utils import process_vision_info

                    assert isinstance(
                        prompt, list
                    ), f"prompt must be a list when processor is not None, got {type(prompt)} instead"
                    multimodal_inputs = process_vision_info(prompt, processor)
                else:
                    multimodal_inputs = None

                label = data[label_key] if label_key is not None else None

                origin_samples.append(
                    Sample(
                        prompt=output_prompt,
                        label=label,
                        metadata=metadata,
                        multimodal_inputs=multimodal_inputs,
                    )
                )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(
                origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    assert len(rollout_data_ref) == dp_size
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]
    if getattr(args, "use_positive_nll_loss", False):
        # raw_reward intentionally stays in global prompt order for pass@k.
        # Training samples are reordered by balance-data, even with DP=1.
        rollout_data["positive_nll_rewards"] = [rollout_data["raw_reward"][i] for i in partition]

    return rollout_data
