import warnings
from time import time
from tqdm import tqdm
import pickle
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.think_press import ThinKPress
from transformers import DynamicCache, QuantoQuantizedCache

warnings.filterwarnings("ignore")

device = "cuda:0"
ckpt = "mistralai/Mistral-7B-Instruct-v0.2"  # Use open-source model for Colab
tokenizer = AutoTokenizer.from_pretrained(ckpt)

def get_size_of_cache(cache):
    if isinstance(cache, QuantoQuantizedCache):
        temp_file = "tmp.pickle"
        with open(temp_file, "wb") as f:
            pickle.dump(cache, f)
        size_in_bytes = os.path.getsize(temp_file)
        os.remove(temp_file)
        return size_in_bytes
    elif isinstance(cache, DynamicCache):
        value_cache = cache.value_cache
        key_cache = cache.key_cache
        size_in_bytes = 0
        for value in value_cache:
            size_in_bytes += value.element_size() * value.nelement()
        for key in key_cache:
            size_in_bytes += key.element_size() * key.nelement()
        return size_in_bytes
    else:
        raise NotImplementedError(f"{type(cache)} is not supported yet.")

def get_prefilling_stats(press, n_tokens, cache_implementation="quantized"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    # Quantize model weights to int8 for memory saving
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        device_map="auto",
        load_in_8bit=True,
        attn_implementation="eager"
    )
    initial_peak_memory = torch.cuda.max_memory_allocated()

    inputs = torch.arange(n_tokens).reshape([1, n_tokens]).to(device)
    with torch.no_grad():
        model(inputs[:, :100])
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    with torch.no_grad(), press(model):
        if cache_implementation == "dynamic":
            cache = DynamicCache()
        elif cache_implementation == "quantized":
            cache = QuantoQuantizedCache(config=model.config, nbits=4)
        else:
            raise NotImplementedError(f"Cache {cache_implementation} not yet implemented")

        start = time()
        model(inputs, num_logits_to_keep=1, past_key_values=cache)
        prefilling_time = time() - start

        cache_size = get_size_of_cache(cache)
        del cache

    peak_memory = torch.cuda.max_memory_allocated()
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {
        "Idle Peak memory": idle_peak_memory / 1024**3,
        "Initial Peak memory": initial_peak_memory / 1024**3,
        "Prefilling time": prefilling_time,
        "Peak memory usage": peak_memory / 1024**3,
        "Cache Size": cache_size / 1024**3,
        "Peak memory w/o weights and KV cache (GB)": (peak_memory - cache_size - initial_peak_memory) / 1024**3
    }

def get_generation_stats(press, n_tokens, max_new_tokens=100, cache_implementation="quantized"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    idle_peak_memory = torch.cuda.max_memory_allocated()
    # Quantize model weights to int8 for memory saving
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        device_map="auto",
        load_in_8bit=True,
        attn_implementation="eager"
    )
    initial_peak_memory = torch.cuda.max_memory_allocated()

    # Safer generation config
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    inputs = torch.arange(n_tokens).reshape([1, n_tokens]).to(device)

    with press(model):
        kwargs = dict()
        if cache_implementation == "quantized":
            kwargs = dict(cache_implementation="quantized", cache_config={"backend": "quanto", "nbits": 4})
        start = time()
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            generation_config=model.generation_config,
            **kwargs
        )
        total_time = time() - start
        assert outputs.shape == (1, n_tokens + max_new_tokens), outputs.shape

    peak_memory = torch.cuda.max_memory_allocated()
    model.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return {
        "Idle Peak memory": idle_peak_memory / 1024**3,
        "Initial Peak memory": initial_peak_memory / 1024**3,
        "Total time": total_time,
        "Peak memory usage": peak_memory / 1024**3
    }

def combine_stats(prefilling_stats, generation_stats):
    combined_stats = {}
    for compression_ratio in prefilling_stats:
        combined_stats[compression_ratio] = dict()
        combined_stats[compression_ratio]['Peak memory usage'] = generation_stats[compression_ratio]['Peak memory usage']
        combined_stats[compression_ratio]['Prefilling time'] = prefilling_stats[compression_ratio]['Prefilling time']
        combined_stats[compression_ratio]['Cache Size'] = prefilling_stats[compression_ratio]['Cache Size']
        combined_stats[compression_ratio]['Total time'] = generation_stats[compression_ratio]['Total time']
        combined_stats[compression_ratio]['Generation time'] = generation_stats[compression_ratio]['Total time'] - prefilling_stats[compression_ratio]['Prefilling time']
    return combined_stats

def plot_compression_stats(stats_knorm, stats_think, title_suffix='', max_peak_memory=45, max_cache_size=17.5, max_generation_time=20):
    gree_color = np.array([118, 185, 0]) / 255
    cmap = LinearSegmentedColormap.from_list("apple_green_to_black", [gree_color / 4, gree_color])

    context_lengths = sorted(stats_knorm.keys())
    compression_ratios = sorted(stats_knorm[context_lengths[0]].keys())

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Peak Memory Usage
    for i, context_length in enumerate(context_lengths):
        memory_usages_knorm = [stats_knorm[context_length][ratio]['Peak memory usage'] for ratio in compression_ratios]
        memory_usages_think = [stats_think[context_length][ratio]['Peak memory usage'] for ratio in compression_ratios]
        axes[0].plot(compression_ratios, memory_usages_knorm, '-o', label=f'Knorm {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
        axes[0].plot(compression_ratios, memory_usages_think, '--s', label=f'ThinK {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
    axes[0].set_xlabel('Compression Ratio')
    axes[0].set_ylabel('Peak Memory Usage (GB)')
    axes[0].set_title('Peak Memory Usage vs. Compression Ratio' + title_suffix)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylim(0, max_peak_memory)

    # Cache Size
    for i, context_length in enumerate(context_lengths):
        cache_knorm = [stats_knorm[context_length][ratio]['Cache Size'] for ratio in compression_ratios]
        cache_think = [stats_think[context_length][ratio]['Cache Size'] for ratio in compression_ratios]
        axes[1].plot(compression_ratios, cache_knorm, '-o', label=f'Knorm {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
        axes[1].plot(compression_ratios, cache_think, '--s', label=f'ThinK {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
    axes[1].set_xlabel('Compression Ratio')
    axes[1].set_ylabel('Cache Size (GB)')
    axes[1].set_title('Cache Size vs. Compression Ratio' + title_suffix)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylim(0, max_cache_size)

    # Generation Time
    for i, context_length in enumerate(context_lengths):
        gen_knorm = [stats_knorm[context_length][ratio]['Generation time'] for ratio in compression_ratios]
        gen_think = [stats_think[context_length][ratio]['Generation time'] for ratio in compression_ratios]
        axes[2].plot(compression_ratios, gen_knorm, '-o', label=f'Knorm {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
        axes[2].plot(compression_ratios, gen_think, '--s', label=f'ThinK {context_length} tokens', color=cmap(i/(len(context_lengths)-1)))
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Generation Time (seconds)')
    axes[2].set_title('Generation Time vs. Compression Ratio' + title_suffix)
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_ylim(0, max_generation_time)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compression_ratios = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    context_lengths = [1024, 2048, 4096]

    stats_knorm = {}
    stats_think = {}

    for n_tokens in context_lengths:
        print(f"Running KnormPress for {n_tokens} tokens...")
        prefilling_stats_knorm = {ratio: get_prefilling_stats(KnormPress(ratio), n_tokens) for ratio in tqdm(compression_ratios)}
        generation_stats_knorm = {ratio: get_generation_stats(KnormPress(ratio), n_tokens) for ratio in tqdm(compression_ratios)}
        stats_knorm[n_tokens] = combine_stats(prefilling_stats_knorm, generation_stats_knorm)

        print(f"Running ThinKPress for {n_tokens} tokens...")
        prefilling_stats_think = {ratio: get_prefilling_stats(ThinKPress(ratio), n_tokens) for ratio in tqdm(compression_ratios)}
        generation_stats_think = {ratio: get_generation_stats(ThinKPress(ratio), n_tokens) for ratio in tqdm(compression_ratios)}
        stats_think[n_tokens] = combine_stats(prefilling_stats_think, generation_stats_think)

    plot_compression_stats(stats_knorm, stats_think, title_suffix=' (int8 quantized)', max_peak_memory=45, max_cache_size=17.5)
