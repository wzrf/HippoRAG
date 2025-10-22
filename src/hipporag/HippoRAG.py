import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import random
import glob
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from igraph import Graph, plot as GraphPlot
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
from copy import deepcopy

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.llm_utils import fix_broken_generated_json

logger = logging.getLogger(__name__)


class HippoRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 llm_model_name_deepthink=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific HippoRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """

        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.info(f"HippoRAG init with config:\n  {_print_config}\n")

        # LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if llm_model_name_deepthink is not None:
            global_config_copy = deepcopy(self.global_config)
            global_config_copy.llm_name = llm_model_name_deepthink
            global_config_copy.max_new_tokens = 4096  ## give this a bigger quota
            self.llm_model_deepthink: BaseLLM = _get_llm_class(global_config_copy)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode == 'Transformers-offline':
            self.openie = TransformersOfflineOpenIE(self.global_config)

        """
        从本地缓存初始化一个graph
        """
        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        ## chunk 向量存储
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        ## entity 向量存储
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,
                                                f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.final_sort_results = []
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self, docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k: chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            # print(f"mengyao_debug doing batch_openie\n")
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        ## offline 实现open IE
        if self.global_config.openie_mode == 'offline':
            print(f"using offline")
            self.pre_openie(docs)

        """
        docs：原始文本
        """

        print(f"mengyao_debug docs are {docs}, inserting into chunk_embedding_store")
        self.chunk_embedding_store.insert_strings(docs)
        """
        hash_id -> text
        """
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        print(f"mengyao_debug all_openie_info is {all_openie_info}")
        print(f"mengyao_debug chunk_keys_to_process is {chunk_keys_to_process}")
        print(f"mengyao_debug new_openie_rows is {new_openie_rows}")

        ### 查询 triplets
        if len(chunk_keys_to_process) > 0:
            print(f"mengyao_debug new_openie_rows is {new_openie_rows}")
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)
        print(f"mengyao_debug ner_results_dict is {ner_results_dict}")
        print(f"mengyao_debug triple_results_dict is {triple_results_dict}")

        assert len(chunk_to_rows) == len(ner_results_dict) == len(
            triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        print(f"mengyao_debug chunk_triples is {chunk_triples}")
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        print(f"mengyao_debug entity_nodes is {entity_nodes}\n"
              f"chunk_triple_entities is {chunk_triple_entities}\n"
              f"facts is {facts}")

        logger.info(f"Encoding Entities")

        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")

        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        """
        num_new_chunks 所有新出现的文本片段；
        """
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            """
            如果出现了新的文本片段，增加图；
            """
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        # Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        # Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        # Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        # Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set(
            [self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        # Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        # Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None,
                 gold_chunk_id: str = "",
                 all_gold_chunk_ids: List[str] = None) -> List[QuerySolution] | Tuple[
        List[QuerySolution], Dict, Dict, Dict, List, List, List]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        """
        对query进行 embedding
        """
        self.get_query_embeddings(queries)

        retrieval_results_dpr = []
        retrieval_results_dpr_plus = []
        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            print(f"query is {query}")
            rerank_start = time.time()
            """
            mengyao_debug 找到和query相关的 fact。
            """
            query_fact_scores = self.get_fact_scores(query)
            print(f"query_fact_scores is {query_fact_scores}")
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            """
            mengyao_debug for query How did Cinderella reach her happy ending?
            , query_fact_scores is [0.11414555 0.26471992 0.13887044 1.         0.         0.09841411
             0.81096749 0.73974348 0.91281923 0.04663707 0.11163547 0.44101249
             0.09769077 0.16271761 0.12138226]
            , top_k_facts is [('cinderella', 'was reunited with', 'prince'), ('cinderella', 'attended', 'royal ball'), ('slipper', 'fit perfectly', 'cinderella'), ('the prince', 'used', 'the lost glass slipper'), ('the prince', 'searched', 'the kingdom')]
            """
            # print(f"mengyao_debug for query {query}\n, "
            #       f"query_fact_scores is {query_fact_scores}\n,"
            #       f" top_k_facts is {top_k_facts}\n")

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                ##todo mengyao 这里只是为了暂时避免warning
                dpr_sorted_doc_ids = sorted_doc_ids
                dpr_sorted_doc_scores = sorted_doc_scores
                dpr_plus_sorted_doc_ids = sorted_doc_ids
                dpr_plus_sorted_doc_scores = sorted_doc_scores
            else:
                """
                使用找到的fact 去搜索对应的doc
                """
                dpr_sorted_doc_ids, dpr_sorted_doc_scores, dpr_plus_sorted_doc_ids, dpr_plus_sorted_doc_scores, sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                    query=query,
                    link_top_k=self.global_config.linking_top_k,
                    query_fact_scores=query_fact_scores,
                    top_k_facts=top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    passage_node_weight=self.global_config.passage_node_weight,
                    all_gold_docs=gold_docs[q_idx])

            print(f"num_to_retrieve top is {num_to_retrieve}")
            top_k_docs_hash = [self.passage_node_keys[idx] for idx in sorted_doc_ids[:num_to_retrieve]]
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            top_k_docs_dpr = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                              dpr_sorted_doc_ids[:num_to_retrieve]]
            top_k_docs_dpr_plus = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                                   dpr_plus_sorted_doc_ids[:num_to_retrieve]]

            # print(f"top_k_docs is {top_k_docs}")
            # print(f"top_k_docs_hash is {top_k_docs_hash}")

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
            retrieval_results_dpr.append(
                QuerySolution(question=query, docs=top_k_docs_dpr, doc_scores=dpr_sorted_doc_scores[:num_to_retrieve]))
            retrieval_results_dpr_plus.append(
                QuerySolution(question=query, docs=top_k_docs_dpr_plus,
                              doc_scores=dpr_plus_sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        print(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        print(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        print(f"Total PPR Time {self.ppr_time:.2f}s")
        print(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            dpr_overall_retrieval_result, dpr_example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs,
                                                                   retrieved_docs=[retrieval_result_dpr.docs for
                                                                                   retrieval_result_dpr in
                                                                                   retrieval_results_dpr],
                                                                   k_list=k_list))
            print(
                f"Evaluation results for DPR retrieval: {dpr_overall_retrieval_result}, dpr_example_retrieval_results is {dpr_example_retrieval_results}")
            dpr_plus_overall_retrieval_result, dpr_plus_example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs,
                                                                   retrieved_docs=[retrieval_result_dpr.docs for
                                                                                   retrieval_result_dpr in
                                                                                   retrieval_results_dpr_plus],
                                                                   k_list=k_list))
            print(
                f"Evaluation results for DPR Plus retrieval: {dpr_plus_overall_retrieval_result}, dpr_example_retrieval_results is {dpr_plus_example_retrieval_results}")
            overall_retrieval_result, example_retrieval_results = (
                retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs,
                                                                   retrieved_docs=[retrieval_result.docs for
                                                                                   retrieval_result in
                                                                                   retrieval_results], k_list=k_list))
            print(
                f"Evaluation results for PPR retrieval: {overall_retrieval_result}, example_retrieval_results is {example_retrieval_results}")
            return (retrieval_results, overall_retrieval_result, dpr_overall_retrieval_result,
                    dpr_plus_overall_retrieval_result,
                    example_retrieval_results, dpr_example_retrieval_results, dpr_plus_example_retrieval_results)
        else:
            return retrieval_results

    def raise_question(self, multi_hop_result: dict, save_directory: str, index: int):
        """
        step 1: let the LLM extract all facts.
        """
        chunks_list = multi_hop_result["chunks_list"]
        fact_extract_prompt = self.prompt_template_manager.render(name='fact_extract', passage=chunks_list)
        print(f"mengyao_debug fact_extract_prompt is {fact_extract_prompt}")
        raw_response, metadata, cache_hit = self.llm_model.infer(fact_extract_prompt)
        if metadata['finish_reason'] == 'length':
            real_response = fix_broken_generated_json(raw_response)
        else:
            real_response = raw_response
        print(f"fact extract raw_response is {real_response}")
        question_generation_prompt = self.prompt_template_manager.render(name='question_generation',
                                                                         passage=real_response)
        raw_response, metadata, cache_hit = self.llm_model_deepthink.infer(question_generation_prompt)
        if metadata['finish_reason'] == 'length':
            question_generated = fix_broken_generated_json(raw_response)
        else:
            question_generated = raw_response

        print(f"question_generated is {question_generated}")

        question_finetuning_prompt = self.prompt_template_manager.render(name='question_finetuning',
                                                                         passage=question_generated)
        raw_response, metadata, cache_hit = self.llm_model.infer(question_finetuning_prompt)
        if metadata['finish_reason'] == 'length':
            question_finetuned = fix_broken_generated_json(raw_response)
        else:
            question_finetuned = raw_response

        question_finetuned = question_finetuned.replace("```json", "").replace("```", "").strip()
        question_finetuned_json = json.loads(question_finetuned)
        print(f"question_finetuned is {question_finetuned}")
        if question_finetuned_json["keep"] == False:
            print(f"mengyao_debug question is dumped, output is {question_finetuned_json}")
        else:
            print(f"mengyao_debug question is kept, output is {question_finetuned_json}")

            question_rate_prompt = self.prompt_template_manager.render(name='question_rate', passage=question_finetuned)
            raw_response, metadata, cache_hit = self.llm_model.infer(question_rate_prompt)
            if metadata['finish_reason'] == 'length':
                question_rate = fix_broken_generated_json(raw_response)
            else:
                question_rate = raw_response
            question_rate = question_rate.replace("```json", "").replace("```", "").strip()
            question_rate_json = json.loads(question_rate)
            print(f"question_rate_json is {question_rate_json}")
            merged_result = {**question_rate_json, **question_finetuned_json}
            merged_result["chunks_list"] = chunks_list
            with open(f"{save_directory}/multi_hop/multi_hop_question_{index}.json", 'w', encoding='utf-8') as f:
                json.dump(merged_result, f, ensure_ascii=False, indent=4)

    def list_all_documents(self, save_directory: str) -> list[str]:
        all_chunks = self.chunk_embedding_store.get_all_id_to_rows()
        all_docs = []
        for key, value in all_chunks.items():
            all_docs.append(value["content"])
        with open(f"{save_directory}/all_original_text.json", 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        return all_docs


    def refine_question(self, es_recall: dict):
        question = es_recall["query"]
        result_dict = {}

        for doc in es_recall.get('gold_docs_analysis', []):
            # 检查rank是否小于15
            if doc.get('rank', float('inf')) < 20:
                term_weights = doc.get('term_weights', {})
                sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:15]
                for term, weight in sorted_terms:
                    if term in result_dict:
                        result_dict[term] += weight
                    else:
                        result_dict[term] = weight

        question_finetuning_prompt = self.prompt_template_manager.render(name='question_refine_es',
                                                                         question=question, keywords=result_dict)
        raw_response, metadata, cache_hit = self.llm_model.infer(question_finetuning_prompt)
        if metadata['finish_reason'] == 'length':
            question_refined = fix_broken_generated_json(raw_response)
        else:
            question_refined = raw_response
        question_refined = question_refined.replace("```json", "").replace("```", "").strip()
        try:
            question_refined_json = json.loads(question_refined)
            print(f"""[question refined] explanation is {question_refined_json["explain"]}""")
            return question_refined_json["question"]
        except Exception as E:
            print(f"[refine_question] fail to dump, question refine result is {question_refined}, exception {E}")
            return question




    def build_graph_and_raise_question(self, save_directory: str, questions_total=1):
        new_graph = self.graph.copy()

        # print(f"mengyao_debug entity_embedding_store are {self.entity_embedding_store.get_all_id_to_rows()}")
        # print(f"mengyao_debug fact_embedding_store are {self.fact_embedding_store.get_all_id_to_rows()}")
        # print(f"mengyao_debug chunk_embedding_store are {self.chunk_embedding_store.get_all_id_to_rows()}")
        # print(f"""mengyao_debug self.graph.vs["name"] are {self.graph.vs["name"]}""")

        vertices = new_graph.vs
        print("所有顶点:", vertices["name"])

        # 3. 遍历所有边，删除所有连接到chunk上面的边。
        edges_to_remove = []

        print("\n遍历所有边及其属性:")
        for edge in new_graph.es:
            # 获取边的所有属性
            edge_attrs = edge.attributes()
            # 检查权重是否为1
            """
            删除contains（指向chunks）以及synonymy
            """
            if ("attributes" in edge_attrs and "contains" in edge_attrs["attributes"]
                    or "synonymy" in edge_attrs["attributes"]):
                edges_to_remove.append(edge.index)
            else:
                if "attributes" in edge_attrs:
                    for attr in edge_attrs["attributes"]:
                        if "[reverse]" in attr:
                            edges_to_remove.append(edge.index)
                            break

        # 4. 删除标记的边（从后往前删除以避免索引问题）
        edges_to_remove.sort(reverse=True)
        for edge_index in edges_to_remove:
            new_graph.delete_edges(edge_index)
            # print(f"已删除边索引 {edge_index}")

        # 再删除chunk点，防止图太乱：
        vertices_to_remove = []

        for vertex in new_graph.vs:
            vertex_label = vertex["name"]
            # 检查标签是否包含"chunk"（不区分大小写）
            if vertex_label is not None and "chunk" in vertex_label.lower():
                vertices_to_remove.append(vertex.index)
        # 4. 删除标记的顶点（从后往前删除以避免索引问题）
        if vertices_to_remove:
            # 按索引降序排序，这样从后往前删除不会影响前面的索引
            vertices_to_remove.sort(reverse=True)

            for vertex_index in vertices_to_remove:
                new_graph.delete_vertices(vertex_index)
                # print(f"已删除顶点索引 {vertex_index}")
        else:
            print("\n没有找到标签包含'chunk'的顶点")

        passages_summary = ""
        all_fact_set = set()
        entities_count = {}
        ## 打印所有边：
        for edge in new_graph.es:
            source_vertex = edge.source
            target_vertex = edge.target
            source_name = new_graph.vs[source_vertex]["content"]
            target_name = new_graph.vs[target_vertex]["content"]
            edge_attrs = edge.attributes()
            all_fact_set.add(f"""{source_name}{edge_attrs["attributes"][0]}{target_name}""")

            if target_name not in entities_count:
                entities_count[target_name] = 0
            if source_name not in entities_count:
                entities_count[source_name] = 0
            entities_count[source_name] = entities_count[source_name] + 1
            entities_count[target_name] = entities_count[target_name] + 1

        sorted_items = sorted(entities_count.items(), key=lambda item: (-item[1], item[0]))
        top_10 = sorted_items[:10]

        print(f"""passages_summary is {all_fact_set}""")
        print(f"""top_10 is {top_10}""")

        all_vertices_names = []
        for name in new_graph.vs["name"]:
            if "chunk" in name:
                all_vertices_names.append(name)
            else:
                all_vertices_names.append(self.entity_embedding_store.get_all_id_to_rows()[name]["content"])
        ig.config["plotting.backend"] = "matplotlib"
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

        def not_fully_contains(current: list, to_be_chosen: list) -> bool:
            for entities in to_be_chosen:
                if entities not in current:
                    return True
            return False

        def get_next_hop(current: list, node_without_usable_edges: set) -> int:
            for index in reversed(range(len(current))):
                if current[index] not in node_without_usable_edges:
                    return current[index]
            return -1

        def draw_graph(new_nodes: dict, valid_edges: list, valid_attributes: dict, index: int):
            print(f"draw_graph new_nodes is {new_nodes}")
            graph = ig.Graph(directed=self.global_config.is_directed_graph)
            graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)
            graph.add_edges(
                valid_edges,
                attributes=valid_attributes
            )

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            ig.plot(graph,
                    # 顶点大小和颜色
                    vertex_size=20,  # 顶点大小
                    vertex_color="lightblue",  # 顶点颜色
                    vertex_frame_color="black",  # 顶点边框颜色
                    vertex_frame_width=1,  # 顶点边框宽度

                    # 标签设置
                    vertex_label=new_nodes["content"],
                    vertex_label_size=8,  # 标签字体大小
                    vertex_label_color="black",  # 标签颜色
                    vertex_label_dist=1,  # 标签与顶点的距离
                    vertex_label_family="sans-serif",  # 字体

                    edge_label=graph.es["attributes"],
                    edge_label_size=8,  # 标签字体大小

                    # 顶点形状
                    vertex_shape="circle"  # 顶点形状：circle, square, triangle, etc.
                    )

            # plt.show()
            plt.savefig(f"{save_directory}/multi_hop/multi_hop_{index}.jpeg", dpi=600)
            plt.close()
            print(f"mengyao_debug draw_graph graph is {graph}")

        def find_max_index_glob(folder_path):
            # 使用glob模式匹配文件
            pattern = os.path.join(folder_path, "multi_hop_question_*.json")
            files = glob.glob(pattern)

            if not files:
                return None

            # 提取数字并找到最大值
            indices = []
            for file_path in files:
                filename = os.path.basename(file_path)
                match = re.match(r"multi_hop_question_(\d+)\.json", filename)
                if match:
                    indices.append(int(match.group(1)))

            return max(indices) if indices else None

        def generate_multihop(index: int) -> bool:
            node_without_usable_edges = set()
            chunks_found = set()
            chunks_list = []
            nodes_hopped = []  ## (a, b)
            edges_index_hopped = []  ## just index
            edges_hopped = []
            valid_attributes = {
                "attributes": []
            }
            facts_list = []
            total_hop = 15
            total_chunks = 10
            """
            为了防止一个节点作为起跳点太多次；如果他作为起跳点4次以上，则不让他再跳；
            """
            total_jump_allowed_from_a_vertex = 4
            vertex_as_start_of_jump = {}
            ## 开始multihop
            ## 假设是5跳：
            vertex_ids = [v.index for v in new_graph.vs]  # 获取所有顶点ID
            initial_vertex = random.choice(vertex_ids)
            nodes_hopped.append(initial_vertex)
            while len(chunks_found) < total_chunks or len(edges_hopped) < total_hop:
                random_vertex = get_next_hop(nodes_hopped, node_without_usable_edges)
                if random_vertex == -1:
                    print(f"跳不下去了，结束！")
                    return False
                incident_edges = new_graph.incident(random_vertex, mode="all")
                print(f"随机选择的顶点: {random_vertex}, edges {incident_edges}")
                if incident_edges:
                    """
                    随机选择一条边
                    """
                    random_edge = random.choice(incident_edges)
                    """
                    如果这条边存在了就不要再走了；
                    
                    我们先不允许成环
                    """
                    if random_edge in edges_index_hopped:
                        if not_fully_contains(edges_index_hopped, incident_edges):
                            print(f"这条边已经走过了，重试")
                            continue
                        else:
                            ## 这个点的所有边都已经被选择过了，这个点已经不能再跳了必须回头了
                            node_without_usable_edges.add(random_vertex)
                            print(f"没有可跳的方向了，回头")
                            continue

                    edge_info = new_graph.es[random_edge]
                    source_vertex = edge_info.source
                    target_vertex = edge_info.target

                    if target_vertex in nodes_hopped:
                        """
                        如果发现成环了，也不可以；把这条边列为已经跳过的，不允许再跳；
                        """
                        edges_index_hopped.append(random_edge)
                        if not_fully_contains(edges_index_hopped, incident_edges):
                            print(f"【成环】这条边已经走过了，重试")
                            continue
                        else:
                            ## 这个点的所有边都已经被选择过了，这个点已经不能再跳了必须回头了
                            node_without_usable_edges.add(random_vertex)
                            print(f"【成环】没有可跳的方向了，回头")
                            continue

                    """
                    可以跳了
                    add vertex and edges to the nodes and edges hopped list
                    """
                    vertex_as_start_of_jump[random_vertex] = vertex_as_start_of_jump.get(random_vertex, 0) + 1
                    if vertex_as_start_of_jump[random_vertex] >= total_jump_allowed_from_a_vertex:
                        node_without_usable_edges.add(random_vertex)
                    if source_vertex == random_vertex:
                        nodes_hopped.append(target_vertex)
                    else:
                        nodes_hopped.append(source_vertex)
                    edges_hopped.append((source_vertex, target_vertex))
                    edges_index_hopped.append(random_edge)

                    valid_attributes["attributes"].append(new_graph.es[random_edge]["attributes"])

                    source_name = new_graph.vs[source_vertex]["content"]
                    target_name = new_graph.vs[target_vertex]["content"]

                    print(f"随机选择的边: {random_edge}, "
                          f"""{source_name} """
                          f"""{edge_info.attributes()["attributes"][0]}"""
                          f"""{target_name} """)
                    chunks_found.add(edge_info.attributes()["chunks"][0])
                    chunk = self.chunk_embedding_store.get_row(edge_info.attributes()["chunks"][0])
                    if chunk not in chunks_list:
                        chunks_list.append(chunk)
                    facts_list.append([source_name, edge_info.attributes()["attributes"][0], target_name])
                    # f"""chunk is {self.chunk_embedding_store.get_row(edge_info["chunks"][0]["hash_id"])}""")
                    # print(f"对应的文章是 {}")
                else:
                    print(f"已经走到尽头了")
                    break

            result = {
                "edges_hopped": edges_hopped,
                "nodes_hopped": nodes_hopped,
                "chunks": list(chunks_found),
                "chunks_list": chunks_list,
                "facts_list": facts_list,
            }

            new_nodes = {
                "name": [new_graph.vs[node]["hash_id"] for node in nodes_hopped],
                "content": [new_graph.vs[node]["content"] for node in nodes_hopped],
            }
            valid_edges = [(new_graph.vs[source_vertex]["hash_id"], new_graph.vs[target_vertex]["hash_id"])
                           for (source_vertex, target_vertex) in edges_hopped]

            print(f"最后选出的结果是 {result} new_nodes is {new_nodes}")
            try:
                os.makedirs(f"{save_directory}/multi_hop")
            except Exception as E:
                print("already exist.")
            index_save = find_max_index_glob(f"{save_directory}/multi_hop") + 1
            print(f"save to index {index_save}")
            try:
                draw_graph(new_nodes, valid_edges, valid_attributes, index_save)
            except Exception as E:
                print(f"fail to draw a picture, reason is {E}")
            with open(f"{save_directory}/multi_hop/multi_hop_{index_save}.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            ## raise a question
            self.raise_question(multi_hop_result=result, save_directory=save_directory, index=index_save)

            return True

        questions_generated = 0
        while questions_generated < questions_total:
            if generate_multihop(questions_generated):
                questions_generated += 1
            else:
                print("try again.")

    def rag_qa(self,
               queries: List[str | QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               gold_chunk_id: str = "",
               all_gold_chunk_ids: List[str] = None) -> (
            Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict, Dict, Dict] | Tuple[
        List[QuerySolution], List[str], List[Dict], Dict, Dict, Dict, Dict, Dict]):
        """
        Performs retrieval-augmented generation enhanced QA using the HippoRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                print(f"mengyao_debug queries is {queries}")
                print(f"mengyao_debug gold_docs is {gold_docs}")
                (queries, overall_retrieval_result,
                 dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result, example_retrieval_results,
                 dpr_example_retrieval_results, dpr_plus_example_retrieval_results) = self.retrieve(queries=queries,
                                                                                                    gold_docs=gold_docs,
                                                                                                    all_gold_chunk_ids=all_gold_chunk_ids,
                                                                                                    gold_chunk_id=gold_chunk_id)

                # print(f"mengyao_debug queries are {queries} overall_retrieval_result is {overall_retrieval_result}")
            else:
                queries = self.retrieve(queries=queries, gold_chunk_id=gold_chunk_id,
                                        all_gold_chunk_ids=all_gold_chunk_ids)

        # Performing QA
        """
        询问大模型
        """
        # queries_solutions, all_response_message, all_metadata = self.qa(queries)

        ##todo mengyao_debug we dont have to really do the QA in here
        queries_solutions, all_response_message, all_metadata = "", "", ""

        print(f"queries_solutions is {queries_solutions}")
        print(f"all_response_message is {all_response_message}")
        print(f"all_metadata is {all_metadata}")

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return (queries_solutions, all_response_message, all_metadata,
                    overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
                    overall_qa_results,
                    example_retrieval_results, dpr_example_retrieval_results, dpr_plus_example_retrieval_results)
        else:
            return (queries_solutions, all_response_message, all_metadata,
                    overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
                    example_retrieval_results, dpr_example_retrieval_results, dpr_plus_example_retrieval_results)

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
                   queries: List[str | QuerySolution],
                   gold_docs: List[List[str]] = None,
                   gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[
        List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        # print(f"mengyao_debug chunk_triples is {chunk_triples}")
        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            # print(f"mengyao_debug chunk_key is {chunk_key} triples is {triples}")
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    # self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                    #     (node_key, node_2_key), 0.0) + 1
                    # self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                    #     (node_2_key, node_key), 0.0) + 1

                    old_stat = self.node_to_node_stats.get((node_key, node_2_key),
                                                           {
                                                               "weight": 0.0,
                                                               "attributes": [],
                                                               "chunks": [],
                                                           })
                    old_stat["weight"] += 1
                    old_stat["attributes"].append(triple[1])
                    old_stat["chunks"].append(chunk_key)
                    self.node_to_node_stats[(node_key, node_2_key)] = old_stat

                    old_stat = self.node_to_node_stats.get((node_2_key, node_key),
                                                           {
                                                               "weight": 0.0,
                                                               "attributes": [],
                                                               "chunks": [],
                                                           })
                    old_stat["weight"] += 1
                    old_stat["attributes"].append(f"{triple[1]} [reverse]")
                    old_stat["chunks"].append(chunk_key)
                    self.node_to_node_stats[(node_2_key, node_key)] = old_stat

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    ## entities -> chunks
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(
                        set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        # print(f"mengyao_debug graph Vertex sequence is {self.graph.vs}")
        # print(f"mengyao_debug graph Edge sequence is {self.graph.es}")
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")
        """
        current_graph_nodes = 
        {'entity-8832a7190f55e53078e8aa42aad1e5b2', 'entity-5c2c2a6c9bed1b1e962c6800b4edfb11', 
        'entity-781056079c8858c93d50d48e995a0a5d', 'entity-6e4c0c8f04b4f89187eda8cc2c988ad4', 
        'entity-7c6d9030e4c7630407701d43317f0af1', 'chunk-733ff7b5b1080ca4eb636ab168e5662d', 
        'chunk-435eaa3536ea075eb9b3cee5c14a4840', 'chunk-d6df73e3b8e71d69e39075792fb855cf', 
        'chunk-05ebe6854219bf0492e050c241805da4', 'entity-4382e967a6b504ff11a3f78bd80a8a6d'}
        """
        # print(f"mengyao_debug current_graph_nodes are {current_graph_nodes}")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            # print(f"mengyao_debug appending idx {idx}, chunk key {chunk_key}")

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = {
                        "weight": 1,
                        "attributes": ["contains"],
                        "chunks": [chunk_key],
                    }

                num_new_chunks += 1

        """
        获取所有新出现的chunk，也就是文本片段；
        """
        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        """
        entity_id_to_row: 
        {
        'entity-583fea53729bcb119bc1099a0dc5e73d': 
        {'hash_id': 'entity-583fea53729bcb119bc1099a0dc5e73d', 'content': 'cinderella'}, 
        'entity-a92df5e4b3532f7dfdf8fca579bbbf70': 
        {'hash_id': 'entity-a92df5e4b3532f7dfdf8fca579bbbf70', 'content': 'erik hort'}, 
        'entity-b20945cd87385383acdd501bfd178936': 
        {'hash_id': 'entity-b20945cd87385383acdd501bfd178936', 'content': 'george rankin'}
        }
        
        entity_node_keys is 
        ['entity-583fea53729bcb119bc1099a0dc5e73d', 
        'entity-a92df5e4b3532f7dfdf8fca579bbbf70', 
        'entity-b20945cd87385383acdd501bfd178936', 
        'entity-ce5225d01c39d2567bc229501d9e610d', 
        'entity-5c2c2a6c9bed1b1e962c6800b4edfb11']
        
        """
        # print(f"mengyao_debug entity_id_to_row is {self.entity_id_to_row}\n"
        #       f"entity_node_keys is {entity_node_keys}")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        # print(f"mengyao_debug query_node_key2knn_node_keys is {query_node_key2knn_node_keys}")
        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        """
        根据embeddings找同义词，不过这个感觉效率有点低怎么把图遍历了一遍，是不是只遍历新增加的node就可以？
        """
        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                ## nn 是同义词 entity， score是相似分数
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]
                    # print(f"mengyao_debug nn_phrase is {nn_phrase}")

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        print(f"mengyao_debug sim_edge is {sim_edge}")
                        # self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        self.node_to_node_stats[sim_edge] = {
                            "weight": score,
                            "attributes": "synonymy",
                            "chunk": "synonymy",
                        }
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))
            # print(f"mengyao_debug synonym_candidates are {len(synonym_candidates)}")

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            # Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                     'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                     'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                     'extracted_entities': [],
                                     'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0

            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }

            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()
        #
        # print(f"mengyao_debug entity_to_row is {entity_to_row}\n"
        #       f"passage_to_row is {passage_to_row}")
        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        # print(f"mengyao_debug new_nodes are {new_nodes}")
        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        # print(f"mengyao_debug self.node_to_node_stats is {self.node_to_node_stats}")
        for edge, attributes in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            # print(f"mengyao_debug attributes is {}")
            graph_adj_list[edge[0]][edge[1]] = attributes["weight"]
            graph_inverse_adj_list[edge[1]][edge[0]] = attributes["weight"]

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append(attributes)

        valid_edges, valid_attributes = [], {
            "weight": [],
            "attributes": [],
            "chunks": [],
        }
        current_node_ids = set(self.graph.vs["name"])
        # print(f"mengyao_debug current_node_ids is {current_node_ids}")
        # print(f"mengyao_debug edge_source_node_keys is {edge_source_node_keys}")
        # print(f"mengyao_debug edge_target_node_keys is {edge_target_node_keys}")
        # print(f"mengyao_debug edge_metadata is {edge_metadata}")
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                attributes = edge_d.get("attributes", [])
                chunks = edge_d.get("chunks", [])
                valid_attributes["weight"].append(weight)
                valid_attributes["attributes"].append(attributes)
                valid_attributes["chunks"].append(chunks)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")

        # print(f"mengyao_debug valid_edges are {valid_edges}")
        # print(f"mengyao_debug valid_attributes are {valid_attributes}")
        res = self.graph.add_edges(
            valid_edges,
            attributes=valid_attributes
        )
        # print(f"mengyao_debug add graph edges res is {res}")

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")
        print(f"mengyao_debug prepare_retrieval_objects")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())  # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())  # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()

        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in
                                  enumerate(self.graph.vs)}  # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx

            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if
                                    node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if
                                     node_key not in igraph_name_to_idx]

            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(
                    f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx

            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in
                                     self.entity_node_keys]  # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in
                                      self.passage_node_keys]  # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        # print(f"mengyao_debug fact_node_keys is {self.fact_node_keys}")
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple),
                                                                                                set()).union(
                        set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(
                    f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")

                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in
                             self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        # print(f"mengyao_debug self.query_to_embedding is {self.query_to_embedding}")
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        # print(f"mengyao_debug all_query_strings are {all_query_strings}")

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction(
                                                                                'query_to_fact'),
                                                                            norm=True)

            # print(f"mengyao_debug query_embeddings_for_triple are {query_embeddings_for_triple}")
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction(
                                                                                 'query_to_passage'),
                                                                             norm=True)

            # print(f"mengyao_debug query_embeddings_for_passage are {query_embeddings_for_passage}")
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])

        # print(f"mengyao_debug self.fact_embeddings is {self.fact_embeddings}")
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)  # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        # print(f"mengyao_debug query_doc_scores is {query_doc_scores}")
        query_doc_scores = min_max_normalize(query_doc_scores)
        # print(f"mengyao_debug min_max_normalize query_doc_scores is {query_doc_scores}")

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        # print(f"mengyao_debug len all_phrase_weights is {np.count_nonzero(all_phrase_weights)}, "
        #       f"len linking_score_map is  {len(linking_score_map.keys())}")
        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05,
                                        all_gold_docs: List[str] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        # Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        dpr_node_keys = []
        dpr_node_scores = []
        ppr_node_keys = []
        ppr_node_scores = []

        phrases_and_ids = set()

        # print(f"mengyao_debug query_fact_scores is {query_fact_scores}")
        # print(f"mengyao_debug top_k_fact_indices is {top_k_fact_indices}")
        # print(f"mengyao_debug phrase_weights length is {len(phrase_weights)}")

        print(f"mengyao_debug gold_docs is {all_gold_docs}")

        for rank, f in enumerate(top_k_facts):
            """
            主谓宾
            """
            # subject_phrase = f[0].lower()
            # predicate_phrase = f[1].lower()
            # object_phrase = f[2].lower()

            ##todo@mengyao 这里不知道为什么改成了小写导致有问题
            subject_phrase = f[0]
            predicate_phrase = f[1]
            object_phrase = f[2]
            """
            score是通过 fact 和 query算出来的；
            """
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                # print(f"mengyao_debug phrase is {phrase}")
                # print(f"mengyao_debug phrase_key is {phrase_key}")
                # print(f"mengyao_debug phrase_id is {phrase_id}")

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        """
                        如果一个phrase指向多篇文章，那这个entity对应的weighted_fact_score 权重要下降；
                        """
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])
                        # print(f"mengyao_debug {phrase} weighted_fact_score "
                        #       f"scale by {len(self.ent_node_to_chunk_ids[phrase_key])} "
                        #       f"weighted_fact_score is {weighted_fact_score}")

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

        # print(f"mengyao_debug phrases_and_ids are {phrases_and_ids}， "
        #       f"phrase_weights is {phrase_weights}, number_of_occurs is {number_of_occurs}")

        ##todo 这里好像有问题 稍微改一下
        number_of_occurs[number_of_occurs == 0] = 1

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        print(f"mengyao_debug phrase_scores is {phrase_scores}")

        def compare_lists_basic(list1, list2):
            """
            基础版本：比较两个列表每个位置的差异
            """
            if len(list1) != len(list2):
                print(f"警告：列表长度不同！list1有{len(list1)}个元素，list2有{len(list2)}个元素")
                return

            differences = []
            for i in range(len(list1)):
                if list1[i] != list2[i]:
                    differences.append((i, list1[i], list2[i]))

            return differences

        """
        筛选高分
        """
        if link_top_k:
            print(f"mengyao_debug using link_top_k, phrase_weights is {phrase_weights}")
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                       phrase_weights,
                                                                       linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k
        print(f"mengyao_debug linking_score_map are {linking_score_map}")

        # Get passage scores according to chosen dense retrieval model
        """
        文章和问题之间的相关性；
        """
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        all_facts_paragraph = " ".join([" ".join(fact) for fact in top_k_facts])
        query_with_fact = f"{all_facts_paragraph} {query}"
        print(f"query_with_fact is {query_with_fact}")
        dpr_plus_sorted_doc_ids, dpr_plus_sorted_doc_scores = self.dense_passage_retrieval(query_with_fact)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
        # print(f"mengyao_debug dpr_sorted_doc_ids are {dpr_sorted_doc_ids}")
        # print(f"mengyao_debug dpr_sorted_doc_scores are {dpr_sorted_doc_scores}")
        # print(f"mengyao_debug normalized_dpr_sorted_scores are {normalized_dpr_sorted_scores}")

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            dpr_node_keys.append(passage_node_key)
            dpr_node_scores.append(passage_dpr_score)
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight
            if i < 10:
                print(
                    f"【top5】【第1步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_dpr_score is {passage_dpr_score}, content is {passage_node_text}\n")
            if passage_node_text in all_gold_docs:
                print(
                    f"【第1步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_dpr_score is {passage_dpr_score}, content is {passage_node_text}\n")

        for i, dpr_sorted_doc_id in enumerate(dpr_plus_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            dpr_node_keys.append(passage_node_key)
            dpr_node_scores.append(passage_dpr_score)
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight
            if i < 10:
                print(
                    f"【top5】【第1.5步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_dpr_score is {passage_dpr_score}, content is {passage_node_text}\n")
            if passage_node_text in all_gold_docs:
                print(
                    f"【第1.5步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_dpr_score is {passage_dpr_score}, content is {passage_node_text}\n")

        # Combining phrase and passage scores into one array for PPR
        """
        把两个加起来了，但是是两个列表，每个点代表了一个vertex，他可以是phrase（主语宾语）也可以是chunk（文章）
        """
        node_weights = phrase_weights + passage_weights
        non_zero_indices = np.nonzero(node_weights)[0]
        non_zero_weights = node_weights[non_zero_indices]

        top10_indices_in_nonzero = np.argsort(non_zero_weights)[-10:][::-1]
        top10_indices_original = non_zero_indices[top10_indices_in_nonzero]
        for idx in top10_indices_original:
            print(f"""mengyao_debug 索引 {idx}: 
            权重 = {node_weights[idx]:.6f}，
            节点 {self.graph.vs[idx]["content"]}""")

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        print(f"ppr time is {ppr_end - ppr_start}")

        # print(f"mengyao_debug ppr和dpr不同的位置有 {compare_lists_basic(dpr_sorted_doc_ids, ppr_sorted_doc_ids)}")

        for i, ppr_sorted_doc_id in enumerate(ppr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[ppr_sorted_doc_id]
            passage_ppr_score = ppr_sorted_doc_scores[i]
            ppr_node_keys.append(passage_node_key)
            ppr_node_scores.append(passage_ppr_score)
            content = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            if i < 10:
                print(
                    f"【top5】【第2步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_ppr_score is {passage_ppr_score}, content is {content}\n")
            if content in all_gold_docs:
                print(
                    f"【第2步匹配】排名第{i} passage_node_key is {passage_node_key}，passage_ppr_score is {passage_ppr_score}, content is {content}\n")

        final_sort_res = {
            "query": query,
            "dpr": {
                "dpr_node_keys": dpr_node_keys,
                "dpr_node_scores": dpr_node_keys
            },
            "ppr": {
                "ppr_node_keys": ppr_node_keys,
                "ppr_node_scores": ppr_node_scores
            }
        }
        self.final_sort_results.append(final_sort_res)

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return dpr_sorted_doc_ids, dpr_sorted_doc_scores, dpr_plus_sorted_doc_ids, dpr_plus_sorted_doc_scores, ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k

        """
        mengyao_debug query_fact_scores is [0.13013543 0.40800445 1.         0.10302908 0.03689147 0.29024954
         0.02906869 0.         0.06840486 0.18845899 0.30114114 0.10817363
         0.73208546 0.00154711 0.76821265], link_top_k is 5
        """
        print(f"mengyao_debug query_fact_scores is {query_fact_scores}, link_top_k is {link_top_k}")

        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}

        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]

            """
            mengyao_debug real_candidate_fact_ids is ['fact-a798cac753d0f061a97a12678ae0175c', 'fact-1341a29a71946ff5dfae70703714cab4', 'fact-55ffb21c5c0d711781232e80fda1baea', 'fact-2877d96752c9607ed3dae9a59dffc8d9', 'fact-4fd022a97d99c09abd4c9126a43319c2']
             fact_row_dict is {'fact-a798cac753d0f061a97a12678ae0175c': {'hash_id': 'fact-a798cac753d0f061a97a12678ae0175c', 'content': "('erik hort', 'birthplace', 'montebello')"}, 'fact-1341a29a71946ff5dfae70703714cab4': {'hash_id': 'fact-1341a29a71946ff5dfae70703714cab4', 'content': "('erik hort', 'is', 'football player')"}, 'fact-55ffb21c5c0d711781232e80fda1baea': {'hash_id': 'fact-55ffb21c5c0d711781232e80fda1baea', 'content': "('erik hort', 'is a', 'football player')"}, 'fact-2877d96752c9607ed3dae9a59dffc8d9': {'hash_id': 'fact-2877d96752c9607ed3dae9a59dffc8d9', 'content': "('marina', 'born in', 'minsk')"}, 'fact-4fd022a97d99c09abd4c9126a43319c2': {'hash_id': 'fact-4fd022a97d99c09abd4c9126a43319c2', 'content': "('montebello', 'located in', 'rockland county')"}}
             candidate_facts is [('erik hort', 'birthplace', 'montebello'), ('erik hort', 'is', 'football player'), ('erik hort', 'is a', 'football player'), ('marina', 'born in', 'minsk'), ('montebello', 'located in', 'rockland county')]
            """
            print(f"mengyao_debug query is {query}"
                  f"real_candidate_fact_ids is {real_candidate_fact_ids}\n "
                  f"fact_row_dict is {fact_row_dict}\n "
                  f"candidate_facts is {candidate_facts}")

            # Rerank the facts
            """
            用大模型对facts进行排序；
            """
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)

            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

            """
             mengyao_debug top_k_fact_indices is [2, 10]
             top_k_facts is [('erik hort', 'birthplace', 'montebello'), ('montebello', 'located in', 'rockland county')]
             reranker_dict is {'confidence': None}
            """
            print(f"mengyao_debug top_k_fact_indices is {top_k_fact_indices}\n "
                  f"top_k_facts is {top_k_facts}\n "
                  f"reranker_dict is {reranker_dict}")

            return top_k_fact_indices, top_k_facts, rerank_log

        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5  # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    def dump_all_results_to_local(self, save_directory: str):
        with open(f"{save_directory}/final_res.json", 'w', encoding='utf-8') as f:
            json.dump(self.final_sort_results, f, ensure_ascii=False, indent=4)
