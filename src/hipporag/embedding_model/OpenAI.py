from copy import deepcopy
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import AutoModel
from openai import OpenAI
from openai import AzureOpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        if self.global_config.azure_embedding_endpoint is None:
            self.client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=self.global_config.embedding_base_url
            )
        else:
            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint)


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        # print(f"mengyao_debug embedding_model_name is {self.embedding_model_name}, texts size is {len(texts)}")
        response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
        results = np.array([v.embedding for v in response.data])

        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 10)

        # 准备批次
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        if len(batches) <= 1:
            # 只有一个批次，直接处理
            results = self.encode(texts)
        else:
            # 使用线程池并发处理
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []

            # 方法1: 使用ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(batches), 8)) as executor:  # 限制最大线程数
                # 提交所有任务
                future_to_batch = {
                    executor.submit(self.encode, batch): idx
                    for idx, batch in enumerate(batches)
                }

                # 按提交顺序收集结果
                ordered_results = [None] * len(batches)
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        ordered_results[batch_idx] = batch_result
                        pbar.update(len(batches[batch_idx]))
                    except Exception as E:
                        print(f"Exception in batch {batch_idx}: {E}")
                        import ipdb;
                        ipdb.set_trace()
                        # 可以选择重新尝试失败的任务或跳过

            # 按原始顺序组合结果
            results = np.concatenate([res for res in ordered_results if res is not None])
            pbar.close()

        # 后续处理
        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results