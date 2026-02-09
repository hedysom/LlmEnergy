import argparse
import os
import sys
import time
import csv

import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker

from util import print_colored_block, get_processor_name, get_gpu_name 

def read_queries(random=False):
    conversations = pd.read_json(path_or_buf=os.path.join(os.path.dirname(__file__), '../llm_baseline_conversations_puffin.jsonl'), lines=True)
    conversations.set_index('id', inplace=True)
    if random:
        conversations = conversations.sample(frac=1)
    return [conv[0]['value'] for conv in conversations['conversations']]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference benchmarking") 
    # data and model input
    parser.add_argument("--experiment")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument("--nogpu", type=int, default=0)
    parser.add_argument("--seconds", type=int, default= 30 * 60, help="number of seconds to profile model on a subset of the data -- 0 process complete")
    args = parser.parse_args()
    tracker = OfflineEmissionsTracker(log_level='error', country_iso_code="ITA")


    # load data
    queries = read_queries()
    
    """
    models=["deepseek-r1:1.5b",
            "gemma3:4b","gemma3:270m",
            "qwen3:4b","qwen3:0.6b",
            "llama3.2:3b", "phi3:3.8b"]
    """
    models = [args.model]
    # load model
    import ollama
    for model in models:
        print(f"Pulling {model}")
        ollama.pull(model)
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": f"Can you answer questions?"}])

        # run evaluations but watch for time limit
        times, n_samples, tokens = [], 0, {'in': [], 'out': []}

        # evaluate queries
        tracker.start()
        print_colored_block(f'STARTING ENERGY PROFILING FOR   {model.upper()}   temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}')
        # run inference
        for query in queries:
            t0 = time.time()
            resp = ollama.chat(model=model, messages=[{"role": "user", "content": query}], options={"temperature": args.temperature})
            try:
                tokens['in'].append(resp['prompt_eval_count'])
                tokens['out'].append(resp['eval_count'])
            except:
                pass
            times.append(time.time() - t0)
            n_samples += 1
            remaining = args.seconds - sum(times) if args.seconds and len(times) < 5 else args.seconds - (sum(times) + np.average(times)) 
            print(f"\rProcessed queries: {n_samples} | Remaining time: {remaining:.1f}s", end='', flush=True)
            if args.seconds and remaining < 0:
                break
        print_colored_block(f'STOPPING ENERGY PROFILING FOR   {model.upper()}  temperature {args.temperature} on   {"CPU" if args.nogpu else "GPU"}', ok=False)
        tracker.stop()

        # add average amount of tokens if there we any errors:
        tokens['in'] += [np.mean(tokens['in'])] * (n_samples - len(tokens['in']))
        tokens['out'] += [np.mean(tokens['out'])] * (n_samples - len(tokens['in']))

        # assess resource consumption
        emissions = 'emissions.csv'
        emission_data = pd.read_csv('emissions.csv').to_dict()

        results = {
            'model': model,
            'running_time_total': emission_data['duration'][0],
            'power_draw_total': emission_data['energy_consumed'][0] * 3.6e6,
            'n_tokens_in': sum(tokens['in']),
            'n_tokens_out': sum(tokens['out']),
            'running_time':  emission_data['duration'][0] / n_samples,
            'power_draw': emission_data['energy_consumed'][0] * 3.6e6 / n_samples,
            'cpu_energy': emission_data['cpu_energy'][0]  * 3.6e6,
            'gpu_energy': emission_data['gpu_energy'][0]  * 3.6e6,
            'ram_energy': emission_data['ram_energy'][0]  * 3.6e6
        }
        
        #write results to a csv file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}-{model}.csv"
        filepath = os.path.join(os.path.dirname(__file__), f"results/{filename}")
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['model','running_time_total' ,'power_draw_total', 'n_tokens_in', 'n_tokens_out', 'running_time', 
                          'power_draw', 'cpu_energy', 'gpu_energy', 'ram_energy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(results)

        #delete emissions file before running another model
        for f in [emissions]:
            os.remove(f)
        print(results)
        print('n_samples', n_samples)
    sys.exit(0)
