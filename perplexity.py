from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator

import json
import math
import os
import sys
import torch
import torch.nn.functional as F
from llama_helpers import *
import numpy as np
import time 

import matplotlib.pyplot as plt 

'''
Passing in model, cache, tokenizer is a total hack because we don't want to have to reinitialize (or move all the globals into a shared state model)
'''

class Perplexity:
    def __init__(self, method="default", model = None, cache = None, tokenizer = None):
        # This needs to be loaded by calling .load()
        self.dataset_chunks = []

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer

        self._begin()


    def _begin(self):
        if self.cache is None:
            self.cache = ExLlamaCache(self.model)
        else:
            self.cache.current_seq_len = 0


    def _next_logits(self, input_ids, apply_lora, last_id_only = True):
        # n_logits = []
        # a = 0
        # while a < input_ids.shape[-1]:
        #     b = min(input_ids.shape[-1], a + 2048)
        #     n_logits.append(self.model.forward(input_ids[:, a:b], self.cache, last_id_only, lora = apply_lora))
        #     a = b
        #
        # return torch.cat(n_logits, dim = 1)

        return self.model.forward(input_ids, self.cache, last_id_only, lora = apply_lora)


    def _tokenize(self, text):
        return self.tokenizer.encode(text)


    # Load raw dataset from a text file and tokenize into chunks. Each chunk can optionally truncated to allow for
    # evaluating the same data at different sequence lengths

    def load(self, dataset_path, chunk_size, chunk_truncate = None, overlap = 0, minlength = 0, json_key = "text"):

        file_extension = os.path.splitext(dataset_path)[1]

        # JSON format: Returned chunks may be of variable length, with each chunk representing one list item

        if file_extension == '.jsonl' or file_extension == '.json':
            with open(dataset_path) as f:
                for line in f:
                    example = json.loads(line)[json_key]
                    if len(example) > minlength:
                        chunk = self._tokenize(example)
                        chunk = chunk[:, :chunk_size]
                        if chunk_truncate is not None: chunk = chunk[:, :chunk_truncate]
                        self.dataset_chunks.append(chunk)

        # Raw Text: Returned chunks are fixed length windows of the entire tokenized dataset

        else:
            with open(dataset_path, encoding="utf-8") as f:
                text = f.read()

            tokens = self._tokenize(text)

            # overlap shouldn't be bigger than the context, also need at least one token for predicting last...
            if overlap >= chunk_size:
                overlap = chunk_size-2

            # We can't use torch.chunks since it want's to split things into equal sized chunks. Instead, let's do our own chunking
            start = 0
            while start < tokens.size(1):
                chunk = tokens[:, start:start + chunk_size]
                start += chunk_size - overlap
                if chunk_truncate is not None: chunk = chunk[:, :chunk_truncate]
                self.dataset_chunks.append(chunk)

    @staticmethod
    def certainty(preds):
        # scores_sorted = sorted(preds)
        # scores_sorted = np.array(scores_sorted)
        # probabilities = scores_sorted / scores_sorted.sum()
        # return probabilities[3] - probabilities[2]

        # scores_sorted = sorted(preds)
        # scores_sorted = np.array(scores_sorted)
        # probabilities = scores_sorted / scores_sorted.sum()
        # print(probabilities)
        # print(probabilities[1], probabilities[0])
        # return probabilities[1] - probabilities[0]

        preds = np.array(preds)
        probabilities = np.exp(preds) / np.sum(np.exp(preds), axis=-1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-2), axis=-1)
        print(probabilities)
        # print(entropy)
        # return entropy
        # return probabilities[0]
        return entropy, probabilities

        # entropy = -np.sum(probabilities * np.log(probabilities))
        # entropy = -np.sum(preds * np.log(preds))
        # print(entropy)

        # return entropy

    



    @staticmethod
    def evaluate_thresholding(corr_samples, incorr_samples, num_threshs=100):
        """
        Given a list of certainties for all correctly predicted samples and one
        for all incorrectly predicted samples, try different thresholds to see
        how many samples would be forwarded to the next model vs. how many incorrectly
        predicted samples would be forwarded to the next model.
        :param corr_samples:
        :param incorr_samples:
        :param num_threshs:
        :return:
        """
        # Generate a range of thresholds to test
        minv = min(min(corr_samples), min(incorr_samples))
        maxv = max(max(corr_samples), max(incorr_samples))
        print("min max: ", minv,maxv)
        thresholds = np.linspace(minv, maxv, num_threshs)

        results = []
        for threshold in thresholds:
            incorr_fwd = 0
            total_fwd = 0
            total = len(incorr_samples)

            # Check how many samples from the correctly predicted would be forwarded
            for num in corr_samples:
                if num >= threshold:
                    total_fwd += 1

            # Check how many samples from the incorrectly predicted would be forwarded
            for num in incorr_samples:
                if num >= threshold:
                    incorr_fwd += 1
                    total_fwd += 1

            # format and append
            incorr_fwd = incorr_fwd / total
            results.append((total_fwd, incorr_fwd))

        total_fwd, incorr_fwd = zip(*results)
        # plt.scatter(total_fwd, incorr_fwd)
        # plt.show()

        return results


    def test(self, chunk_limit = sys.maxsize, lora = None, tag = "", ppl_token = False):
        

        # Hacky: Ignore passed in dataset and load HellaSWAG
        prompts, labels = get_hellaswag(100)

        corr = 0
        incorr = 0
        correct_certs = []
        incorrect_certs = []

        combined_certs = [] 

        i = 0
        # total_time = 0 
        # start_time = 0
        inference_times = []
        for prompt, label in zip(prompts, labels):
            scores = []
            # if i >= 5:
            #     start_time = time.time()
            start_time = time.time()

            for q, a in prompt:
                self._begin()

                input_ids = self._tokenize(q+a)
                answer_ids = self._tokenize(a)

                logits = self._next_logits(input_ids, lora, last_id_only = False)
                log_probs = F.log_softmax(logits, dim=-1)

                # log probs of answers
                relevant_log_probs = log_probs[:, -answer_ids.shape[1]-1:-1]

                # compute answer prob, average over seq length
                seq_prob = 0
                for i in range(answer_ids.shape[-1]):
                    correct_token = int(answer_ids[0, i])
                    seq_prob += float(relevant_log_probs[0, i, correct_token])
                scores.append(seq_prob / float(answer_ids.shape[-1]))

            # compute certainty score, TODO: Try different ones here
            cert, probabilities = self.certainty(scores)

            # print(type(cert))
            combined_certs.append(probabilities.tolist())

            # check if prediction correct
            if np.argmax(scores) == label:
                corr += 1
                correct_certs.append(cert)
            else:
                incorr += 1
                incorrect_certs.append(cert)

            # if i >= 5:
            #     end_time = time.time()
            #     assert start_time != 0
            #     total_time += end_time - start_time
            #     print(end_time - start_time, total_time)
            end_time = time.time()
            inference_times.append(end_time - start_time)

            i += 1
        
        total_time = sum(inference_times)
        total_time_warmup = sum(inference_times[5:])

        print("Total time: ", total_time)
        print("Time per inference: ", total_time /  (len(labels)))
        print("Total time (with warmup): ", total_time_warmup)
        print("Time per inference (with warmup): ", total_time_warmup /  (len(labels) - 5))

        # Compute how many correctly forwarded etc
        results = self.evaluate_thresholding(correct_certs, incorrect_certs)
        print("====================================")
        print("Samples forwarded:", "\tFrac of errors forwarded:")
        for total_f, incorr_f in results:
            print(f"{total_f}\t\t\t{incorr_f}")
        print("====================================")
        print(f"Overall accuracy on task: {corr/(incorr+corr)}")

        if False:
            # write to file certainties
            with open('certainties_13b_1000.json', 'w') as file:
                json.dump(combined_certs, file)

            with open('true_vals_13b_1000.json', 'w') as file:
                json.dump(labels, file)


def add_args(parser):

    parser.add_argument("-ppl", "--perplexity", nargs = '?', const = 'default', metavar = "METHOD", help = "Perplexity benchmark. Optionally specify method: gptq-for-llama, llama.cpp (not yet implemented)")
    parser.add_argument("-ppl_ds", "--perplexity_dataset", metavar = "DATAPATH", type = str, help = "Load dataset for perplexity (JSONL if .jsonl, otherwise parses it as raw text)")
    parser.add_argument("-ppl_cn", "--perplexity_chunk_num", nargs = "?", type = int, help = "Number of chunks for perplexity benchmark", default = 100)
    parser.add_argument("-ppl_cs", "--perplexity_chunk_size", type = int, help = "Size of chunks for perplexity benchmark", default = 2048)
    parser.add_argument("-ppl_ct", "--perplexity_chunk_truncate", type = int, help = "Truncated size of chunks for perplexity benchmark", default = 2048)
    parser.add_argument("-ppl_co", "--perplexity_chunk_overlap", type = int, help = "Chunk overlap", default = 0)
    parser.add_argument("-ppl_cm", "--perplexity_chunk_min", type = int, help = "Minimum chunk length", default = 50)
    parser.add_argument("-ppl_key", "--perplexity_json_key", type = str, help = "Key to extract from JSON dataset, default: 'text'", default = "text")
    parser.add_argument("-ppl_t", "--perplexity_token", action = "store_true", help = "Run perplexity test on individual tokens, for debug purposes (slow)")


def post_parse(args):

    if not args.perplexity: return

    # GPTQ-for-LLaMa equivalent

    if args.perplexity == "gptq-for-llama":
        args.perplexity_dataset = "datasets/wikitext2.txt"
        args.perplexity_chunk_num = 128
        args.perplexity_chunk_size = 2048
        args.perplexity_chunk_truncate = 2048
        args.perplexity_chunk_overlap = 0
        args.perplexity_chunk_min = 0

    # Default dataset for legacy method

    if args.perplexity_dataset is None: args.perplexity_dataset = "datasets/wikitext2_val_sample.jsonl"

    print(f" -- Perplexity:")
    print(f" -- - Dataset: {args.perplexity_dataset}")
    print(f" -- - Chunks: {args.perplexity_chunk_num}")
    print(f" -- - Chunk size: {args.perplexity_chunk_size}" + (f" -> {args.perplexity_chunk_truncate}" if args.perplexity_chunk_truncate is not None else ""))
    print(f" -- - Chunk overlap: {args.perplexity_chunk_overlap}")
    print(f" -- - Min. chunk size: {args.perplexity_chunk_min}")
    print(f" -- - Key: {args.perplexity_json_key}")
    if args.perplexity_token: print("f -- - Per-token mode")
