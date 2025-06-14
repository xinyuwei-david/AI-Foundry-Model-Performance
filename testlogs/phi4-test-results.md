**Test result for deploy phi4 on 1*NC48 A100 VM:**

```
(aml_env) root@linuxworkvm:~/AIFperformance# python  press-phi4-0314.py
Please enter the API service URL: https://aml-david-1-nc48.polandcentral.inference.ml.azure.com/score
Please enter the API Key: EhVrLXKhMdlkUvvmgrORZDVP1Ki4z10PaOqdnwx3znxqQ3BHyNyqJQQJ99BCAAAAAAAAAAAAINFRAZML06ur
Please enter the full name of the HuggingFace model for tokenizer loading: microsoft/phi-4
Tokenizer loaded successfully: microsoft/phi-4

Scenario: Text Generation, Concurrency: 1
  Request 1:
    TTFT          : 12.473 s
    Latency       : 12.473 s
    Throughput    : 68.07 tokens/s
    Prompt tokens : 132, Output tokens: 849

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 12.473 s
    Average throughput per req   : 68.07 tokens/s
    Overall throughput (sum)     : 68.07 tokens/s
    Batch duration (wall-clock)  : 12.496 s

Scenario: Text Generation, Concurrency: 2
  Request 1:
    TTFT          : 12.482 s
    Latency       : 12.482 s
    Throughput    : 70.90 tokens/s
    Prompt tokens : 132, Output tokens: 885
  Request 2:
    TTFT          : 26.099 s
    Latency       : 26.099 s
    Throughput    : 40.04 tokens/s
    Prompt tokens : 132, Output tokens: 1045

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 19.291 s
    Average throughput per req   : 55.47 tokens/s
    Overall throughput (sum)     : 110.94 tokens/s
    Batch duration (wall-clock)  : 26.129 s

Scenario: Question Answering, Concurrency: 1
  Request 1:
    TTFT          : 11.914 s
    Latency       : 11.914 s
    Throughput    : 72.10 tokens/s
    Prompt tokens : 114, Output tokens: 859

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 11.914 s
    Average throughput per req   : 72.10 tokens/s
    Overall throughput (sum)     : 72.10 tokens/s
    Batch duration (wall-clock)  : 11.935 s

Scenario: Question Answering, Concurrency: 2
  Request 1:
    TTFT          : 9.169 s
    Latency       : 9.169 s
    Throughput    : 70.02 tokens/s
    Prompt tokens : 114, Output tokens: 642
  Request 2:
    TTFT          : 19.162 s
    Latency       : 19.162 s
    Throughput    : 39.92 tokens/s
    Prompt tokens : 114, Output tokens: 765

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 14.165 s
    Average throughput per req   : 54.97 tokens/s
    Overall throughput (sum)     : 109.94 tokens/s
    Batch duration (wall-clock)  : 19.190 s

Scenario: Translation, Concurrency: 1
  Request 1:
    TTFT          : 2.499 s
    Latency       : 2.499 s
    Throughput    : 47.62 tokens/s
    Prompt tokens : 85, Output tokens: 119

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 2.499 s
    Average throughput per req   : 47.62 tokens/s
    Overall throughput (sum)     : 47.62 tokens/s
    Batch duration (wall-clock)  : 2.517 s

Scenario: Translation, Concurrency: 2
  Request 1:
    TTFT          : 2.501 s
    Latency       : 2.501 s
    Throughput    : 47.98 tokens/s
    Prompt tokens : 85, Output tokens: 120
  Request 2:
    TTFT          : 4.181 s
    Latency       : 4.181 s
    Throughput    : 28.46 tokens/s
    Prompt tokens : 85, Output tokens: 119

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.341 s
    Average throughput per req   : 38.22 tokens/s
    Overall throughput (sum)     : 76.45 tokens/s
    Batch duration (wall-clock)  : 4.206 s

Scenario: Text Summarization, Concurrency: 1
  Request 1:
    TTFT          : 2.811 s
    Latency       : 2.811 s
    Throughput    : 50.16 tokens/s
    Prompt tokens : 90, Output tokens: 141

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 2.811 s
    Average throughput per req   : 50.16 tokens/s
    Overall throughput (sum)     : 50.16 tokens/s
    Batch duration (wall-clock)  : 2.829 s

Scenario: Text Summarization, Concurrency: 2
  Request 1:
    TTFT          : 2.575 s
    Latency       : 2.575 s
    Throughput    : 48.15 tokens/s
    Prompt tokens : 90, Output tokens: 124
  Request 2:
    TTFT          : 4.413 s
    Latency       : 4.413 s
    Throughput    : 29.23 tokens/s
    Prompt tokens : 90, Output tokens: 129

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.494 s
    Average throughput per req   : 38.69 tokens/s
    Overall throughput (sum)     : 77.38 tokens/s
    Batch duration (wall-clock)  : 4.438 s

Scenario: Code Generation, Concurrency: 1
  Request 1:
    TTFT          : 20.441 s
    Latency       : 20.441 s
    Throughput    : 83.12 tokens/s
    Prompt tokens : 79, Output tokens: 1699

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 20.441 s
    Average throughput per req   : 83.12 tokens/s
    Overall throughput (sum)     : 83.12 tokens/s
    Batch duration (wall-clock)  : 20.462 s

Scenario: Code Generation, Concurrency: 2
Attempt 1 failed: The read operation timed out
  Request 1:
    TTFT          : 17.370 s
    Latency       : 17.370 s
    Throughput    : 83.36 tokens/s
    Prompt tokens : 79, Output tokens: 1448
  Request 2:
    TTFT          : 16.017 s
    Latency       : 16.017 s
    Throughput    : 79.36 tokens/s
    Prompt tokens : 79, Output tokens: 1271

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 16.693 s
    Average throughput per req   : 81.36 tokens/s
    Overall throughput (sum)     : 162.72 tokens/s
    Batch duration (wall-clock)  : 47.685 s

Scenario: Chatbot, Concurrency: 1
  Request 1:
    TTFT          : 5.035 s
    Latency       : 5.035 s
    Throughput    : 64.55 tokens/s
    Prompt tokens : 60, Output tokens: 325

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 5.035 s
    Average throughput per req   : 64.55 tokens/s
    Overall throughput (sum)     : 64.55 tokens/s
    Batch duration (wall-clock)  : 5.052 s

Scenario: Chatbot, Concurrency: 2
  Request 1:
    TTFT          : 6.337 s
    Latency       : 6.337 s
    Throughput    : 67.39 tokens/s
    Prompt tokens : 60, Output tokens: 427
  Request 2:
    TTFT          : 11.039 s
    Latency       : 11.039 s
    Throughput    : 32.70 tokens/s
    Prompt tokens : 60, Output tokens: 361

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 8.688 s
    Average throughput per req   : 50.04 tokens/s
    Overall throughput (sum)     : 100.09 tokens/s
    Batch duration (wall-clock)  : 11.065 s

Scenario: Sentiment Analysis / Classification, Concurrency: 1
  Request 1:
    TTFT          : 1.009 s
    Latency       : 1.009 s
    Throughput    : 5.95 tokens/s
    Prompt tokens : 82, Output tokens: 6

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.009 s
    Average throughput per req   : 5.95 tokens/s
    Overall throughput (sum)     : 5.95 tokens/s
    Batch duration (wall-clock)  : 1.026 s

Scenario: Sentiment Analysis / Classification, Concurrency: 2
  Request 1:
    TTFT          : 1.146 s
    Latency       : 1.146 s
    Throughput    : 14.83 tokens/s
    Prompt tokens : 82, Output tokens: 17
  Request 2:
    TTFT          : 1.356 s
    Latency       : 1.356 s
    Throughput    : 5.16 tokens/s
    Prompt tokens : 82, Output tokens: 7

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 1.251 s
    Average throughput per req   : 9.99 tokens/s
    Overall throughput (sum)     : 19.99 tokens/s
    Batch duration (wall-clock)  : 1.382 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 1
  Request 1:
    TTFT          : 13.148 s
    Latency       : 13.148 s
    Throughput    : 76.44 tokens/s
    Prompt tokens : 99, Output tokens: 1005

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 13.148 s
    Average throughput per req   : 76.44 tokens/s
    Overall throughput (sum)     : 76.44 tokens/s
    Batch duration (wall-clock)  : 13.167 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 2
  Request 1:
    TTFT          : 14.184 s
    Latency       : 14.184 s
    Throughput    : 74.31 tokens/s
    Prompt tokens : 99, Output tokens: 1054
  Request 2:
    TTFT          : 26.283 s
    Latency       : 26.283 s
    Throughput    : 36.53 tokens/s
    Prompt tokens : 99, Output tokens: 960

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 20.233 s
    Average throughput per req   : 55.42 tokens/s
    Overall throughput (sum)     : 110.84 tokens/s
    Batch duration (wall-clock)  : 26.310 s
```

**Test result for deploy phi4 on 2*NC24 A100 VM( (When concurrency exceeds 2, a 429 error will occur.):**

```
  (aml_env) root@linuxworkvm:~/AIFperformance# python  press-phi4-0314.py
Please enter the API service URL: https://aml-david-2-nc24.polandcentral.inference.ml.azure.com/score
Please enter the API Key: 4s9oKys5yetlZnmP1hMcYXNUOUj5rIDIl2tJfX1ULebvgxFotfulJQQJ99BCAAAAAAAAAAAAINFRAZML1pQg
Please enter the full name of the HuggingFace model for tokenizer loading: microsoft/phi-4
tokenizer_config.json: 100%|█████████████████████████| 17.7k/17.7k [00:00<00:00, 7.65MB/s]
vocab.json: 100%|████████████████████████████████████| 1.61M/1.61M [00:00<00:00, 3.91MB/s]
merges.txt: 100%|██████████████████████████████████████| 917k/917k [00:00<00:00, 4.51MB/s]
tokenizer.json: 100%|████████████████████████████████| 4.25M/4.25M [00:00<00:00, 6.69MB/s]
added_tokens.json: 100%|█████████████████████████████| 2.50k/2.50k [00:00<00:00, 2.51MB/s]
special_tokens_map.json: 100%|█████████████████████████| 95.0/95.0 [00:00<00:00, 96.1kB/s]
Tokenizer loaded successfully: microsoft/phi-4

Scenario: Text Generation, Concurrency: 1
  Request 1:
    TTFT          : 19.546 s
    Latency       : 19.546 s
    Throughput    : 44.66 tokens/s
    Prompt tokens : 132, Output tokens: 873

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 19.546 s
    Average throughput per req   : 44.66 tokens/s
    Overall throughput (sum)     : 44.66 tokens/s
    Batch duration (wall-clock)  : 19.569 s

Scenario: Text Generation, Concurrency: 2
  Request 1:
    TTFT          : 19.542 s
    Latency       : 19.542 s
    Throughput    : 44.67 tokens/s
    Prompt tokens : 132, Output tokens: 873
  Request 2:
    TTFT          : 20.414 s
    Latency       : 20.414 s
    Throughput    : 45.46 tokens/s
    Prompt tokens : 132, Output tokens: 928

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 19.978 s
    Average throughput per req   : 45.07 tokens/s
    Overall throughput (sum)     : 90.13 tokens/s
    Batch duration (wall-clock)  : 20.444 s

Scenario: Question Answering, Concurrency: 1
  Request 1:
    TTFT          : 15.552 s
    Latency       : 15.552 s
    Throughput    : 44.56 tokens/s
    Prompt tokens : 114, Output tokens: 693

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 15.552 s
    Average throughput per req   : 44.56 tokens/s
    Overall throughput (sum)     : 44.56 tokens/s
    Batch duration (wall-clock)  : 15.573 s

Scenario: Question Answering, Concurrency: 2
Attempt 1 failed: The read operation timed out
  Request 1:
    TTFT          : 15.207 s
    Latency       : 15.207 s
    Throughput    : 45.77 tokens/s
    Prompt tokens : 114, Output tokens: 696
  Request 2:
    TTFT          : 16.606 s
    Latency       : 16.606 s
    Throughput    : 45.10 tokens/s
    Prompt tokens : 114, Output tokens: 749

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 15.906 s
    Average throughput per req   : 45.44 tokens/s
    Overall throughput (sum)     : 90.87 tokens/s
    Batch duration (wall-clock)  : 48.279 s

Scenario: Translation, Concurrency: 1
  Request 1:
    TTFT          : 3.241 s
    Latency       : 3.241 s
    Throughput    : 33.32 tokens/s
    Prompt tokens : 85, Output tokens: 108

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.241 s
    Average throughput per req   : 33.32 tokens/s
    Overall throughput (sum)     : 33.32 tokens/s
    Batch duration (wall-clock)  : 3.258 s

Scenario: Translation, Concurrency: 2
  Request 1:
    TTFT          : 3.170 s
    Latency       : 3.170 s
    Throughput    : 33.12 tokens/s
    Prompt tokens : 85, Output tokens: 105
  Request 2:
    TTFT          : 5.856 s
    Latency       : 5.856 s
    Throughput    : 20.83 tokens/s
    Prompt tokens : 85, Output tokens: 122

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 4.513 s
    Average throughput per req   : 26.98 tokens/s
    Overall throughput (sum)     : 53.95 tokens/s
    Batch duration (wall-clock)  : 5.879 s

Scenario: Text Summarization, Concurrency: 1
  Request 1:
    TTFT          : 4.630 s
    Latency       : 4.630 s
    Throughput    : 37.36 tokens/s
    Prompt tokens : 90, Output tokens: 173

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 4.630 s
    Average throughput per req   : 37.36 tokens/s
    Overall throughput (sum)     : 37.36 tokens/s
    Batch duration (wall-clock)  : 4.647 s

Scenario: Text Summarization, Concurrency: 2
  Request 1:
    TTFT          : 3.650 s
    Latency       : 3.650 s
    Throughput    : 34.80 tokens/s
    Prompt tokens : 90, Output tokens: 127
  Request 2:
    TTFT          : 3.678 s
    Latency       : 3.678 s
    Throughput    : 34.81 tokens/s
    Prompt tokens : 90, Output tokens: 128

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.664 s
    Average throughput per req   : 34.80 tokens/s
    Overall throughput (sum)     : 69.60 tokens/s
    Batch duration (wall-clock)  : 3.700 s

Scenario: Code Generation, Concurrency: 1
  Request 1:
    TTFT          : 27.685 s
    Latency       : 27.685 s
    Throughput    : 51.58 tokens/s
    Prompt tokens : 79, Output tokens: 1428

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 27.685 s
    Average throughput per req   : 51.58 tokens/s
    Overall throughput (sum)     : 51.58 tokens/s
    Batch duration (wall-clock)  : 27.704 s

Scenario: Code Generation, Concurrency: 2
Attempt 1 failed: The read operation timed out
Attempt 2 failed: The read operation timed out
  Request 1:
    TTFT          : 25.714 s
    Latency       : 25.714 s
    Throughput    : 52.19 tokens/s
    Prompt tokens : 79, Output tokens: 1342
  Request 2:
    TTFT          : 26.907 s
    Latency       : 26.907 s
    Throughput    : 52.18 tokens/s
    Prompt tokens : 79, Output tokens: 1404

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 26.310 s
    Average throughput per req   : 52.18 tokens/s
    Overall throughput (sum)     : 104.37 tokens/s
    Batch duration (wall-clock)  : 90.229 s

Scenario: Chatbot, Concurrency: 1
  Request 1:
    TTFT          : 9.349 s
    Latency       : 9.349 s
    Throughput    : 43.96 tokens/s
    Prompt tokens : 60, Output tokens: 411

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 9.349 s
    Average throughput per req   : 43.96 tokens/s
    Overall throughput (sum)     : 43.96 tokens/s
    Batch duration (wall-clock)  : 9.367 s

Scenario: Chatbot, Concurrency: 2
  Request 1:
    TTFT          : 8.554 s
    Latency       : 8.554 s
    Throughput    : 43.37 tokens/s
    Prompt tokens : 60, Output tokens: 371
  Request 2:
    TTFT          : 10.521 s
    Latency       : 10.521 s
    Throughput    : 44.29 tokens/s
    Prompt tokens : 60, Output tokens: 466

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 9.537 s
    Average throughput per req   : 43.83 tokens/s
    Overall throughput (sum)     : 87.67 tokens/s
    Batch duration (wall-clock)  : 10.545 s

Scenario: Sentiment Analysis / Classification, Concurrency: 1
  Request 1:
    TTFT          : 1.235 s
    Latency       : 1.235 s
    Throughput    : 12.96 tokens/s
    Prompt tokens : 82, Output tokens: 16

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.235 s
    Average throughput per req   : 12.96 tokens/s
    Overall throughput (sum)     : 12.96 tokens/s
    Batch duration (wall-clock)  : 1.252 s

Scenario: Sentiment Analysis / Classification, Concurrency: 2
  Request 1:
    TTFT          : 1.045 s
    Latency       : 1.045 s
    Throughput    : 6.70 tokens/s
    Prompt tokens : 82, Output tokens: 7
  Request 2:
    TTFT          : 1.270 s
    Latency       : 1.270 s
    Throughput    : 13.39 tokens/s
    Prompt tokens : 82, Output tokens: 17

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 1.157 s
    Average throughput per req   : 10.04 tokens/s
    Overall throughput (sum)     : 20.09 tokens/s
    Batch duration (wall-clock)  : 1.293 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 1
  Request 1:
    TTFT          : 20.184 s
    Latency       : 20.184 s
    Throughput    : 47.12 tokens/s
    Prompt tokens : 99, Output tokens: 951

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 20.184 s
    Average throughput per req   : 47.12 tokens/s
    Overall throughput (sum)     : 47.12 tokens/s
    Batch duration (wall-clock)  : 20.202 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 2
  Request 1:
    TTFT          : 22.688 s
    Latency       : 22.688 s
    Throughput    : 47.56 tokens/s
    Prompt tokens : 99, Output tokens: 1079
  Request 2:
    TTFT          : 24.621 s
    Latency       : 24.621 s
    Throughput    : 46.91 tokens/s
    Prompt tokens : 99, Output tokens: 1155

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 23.655 s
    Average throughput per req   : 47.23 tokens/s
    Overall throughput (sum)     : 94.47 tokens/s
    Batch duration (wall-clock)  : 24.648 s
```



**Test result for deploy phi4 on 1*NC24 A100 VM( (When concurrency exceeds 2, a 429 error will occur.):**

```
(aml_env) root@linuxworkvm:~/AIFperformance# python  press-phi4-0314.py
Please enter the API service URL: https://aml-david-1-nc24.polandcentral.inference.ml.azure.com/score
Please enter the API Key: 76WXPsoTlX02RIrijwJdUQtDL5K1iIuOqT9vRhOMtC4p2zwRlP9IJQQJ99BCAAAAAAAAAAAAINFRAZMLjTTD
Please enter the full name of the HuggingFace model for tokenizer loading: microsoft/phi-4
Tokenizer loaded successfully: microsoft/phi-4

Scenario: Text Generation, Concurrency: 1
  Request 1:
    TTFT          : 19.497 s
    Latency       : 19.497 s
    Throughput    : 44.78 tokens/s
    Prompt tokens : 132, Output tokens: 873

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 19.497 s
    Average throughput per req   : 44.78 tokens/s
    Overall throughput (sum)     : 44.78 tokens/s
    Batch duration (wall-clock)  : 19.521 s

Scenario: Text Generation, Concurrency: 2
Attempt 1 failed: The read operation timed out
  Request 1:
    TTFT          : 20.401 s
    Latency       : 20.401 s
    Throughput    : 45.49 tokens/s
    Prompt tokens : 132, Output tokens: 928
  Request 2:
    TTFT          : 28.750 s
    Latency       : 28.750 s
    Throughput    : 33.77 tokens/s
    Prompt tokens : 132, Output tokens: 971

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 24.576 s
    Average throughput per req   : 39.63 tokens/s
    Overall throughput (sum)     : 79.26 tokens/s
    Batch duration (wall-clock)  : 60.422 s

Scenario: Question Answering, Concurrency: 1
  Request 1:
    TTFT          : 15.943 s
    Latency       : 15.943 s
    Throughput    : 46.04 tokens/s
    Prompt tokens : 114, Output tokens: 734

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 15.943 s
    Average throughput per req   : 46.04 tokens/s
    Overall throughput (sum)     : 46.04 tokens/s
    Batch duration (wall-clock)  : 15.962 s

Scenario: Question Answering, Concurrency: 2
  Request 1:
    TTFT          : 16.011 s
    Latency       : 16.011 s
    Throughput    : 47.09 tokens/s
    Prompt tokens : 114, Output tokens: 754
  Request 2:
    TTFT          : 27.537 s
    Latency       : 27.537 s
    Throughput    : 19.57 tokens/s
    Prompt tokens : 114, Output tokens: 539

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 21.774 s
    Average throughput per req   : 33.33 tokens/s
    Overall throughput (sum)     : 66.67 tokens/s
    Batch duration (wall-clock)  : 27.563 s

Scenario: Translation, Concurrency: 1
  Request 1:
    TTFT          : 3.411 s
    Latency       : 3.411 s
    Throughput    : 34.59 tokens/s
    Prompt tokens : 85, Output tokens: 118

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.411 s
    Average throughput per req   : 34.59 tokens/s
    Overall throughput (sum)     : 34.59 tokens/s
    Batch duration (wall-clock)  : 3.429 s

Scenario: Translation, Concurrency: 2
  Request 1:
    TTFT          : 6.052 s
    Latency       : 6.052 s
    Throughput    : 39.99 tokens/s
    Prompt tokens : 85, Output tokens: 242
  Request 2:
    TTFT          : 15.796 s
    Latency       : 15.796 s
    Throughput    : 28.55 tokens/s
    Prompt tokens : 85, Output tokens: 451

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 10.924 s
    Average throughput per req   : 34.27 tokens/s
    Overall throughput (sum)     : 68.54 tokens/s
    Batch duration (wall-clock)  : 15.822 s

Scenario: Text Summarization, Concurrency: 1
  Request 1:
    TTFT          : 3.369 s
    Latency       : 3.369 s
    Throughput    : 33.84 tokens/s
    Prompt tokens : 90, Output tokens: 114

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.369 s
    Average throughput per req   : 33.84 tokens/s
    Overall throughput (sum)     : 33.84 tokens/s
    Batch duration (wall-clock)  : 3.387 s

Scenario: Text Summarization, Concurrency: 2
  Request 1:
    TTFT          : 4.376 s
    Latency       : 4.376 s
    Throughput    : 36.56 tokens/s
    Prompt tokens : 90, Output tokens: 160
  Request 2:
    TTFT          : 8.259 s
    Latency       : 8.259 s
    Throughput    : 22.88 tokens/s
    Prompt tokens : 90, Output tokens: 189

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 6.317 s
    Average throughput per req   : 29.72 tokens/s
    Overall throughput (sum)     : 59.45 tokens/s
    Batch duration (wall-clock)  : 8.283 s

Scenario: Code Generation, Concurrency: 1
  Request 1:
    TTFT          : 26.504 s
    Latency       : 26.504 s
    Throughput    : 52.26 tokens/s
    Prompt tokens : 79, Output tokens: 1385

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 26.504 s
    Average throughput per req   : 52.26 tokens/s
    Overall throughput (sum)     : 52.26 tokens/s
    Batch duration (wall-clock)  : 26.523 s

Scenario: Code Generation, Concurrency: 2
Attempt 1 failed: The read operation timed out
Attempt 2 failed: The read operation timed out
Attempt 3 failed: The read operation timed out
  Request 1:
    TTFT          : 27.772 s
    Latency       : 27.772 s
    Throughput    : 53.22 tokens/s
    Prompt tokens : 79, Output tokens: 1478

  Summary for concurrency 2:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 27.772 s
    Average throughput per req   : 53.22 tokens/s
    Overall throughput (sum)     : 53.22 tokens/s
    Batch duration (wall-clock)  : 93.942 s

Scenario: Chatbot, Concurrency: 1
Attempt 1 failed: The read operation timed out
  Request 1:
    TTFT          : 8.366 s
    Latency       : 8.366 s
    Throughput    : 41.24 tokens/s
    Prompt tokens : 60, Output tokens: 345

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.366 s
    Average throughput per req   : 41.24 tokens/s
    Overall throughput (sum)     : 41.24 tokens/s
    Batch duration (wall-clock)  : 40.031 s

Scenario: Chatbot, Concurrency: 2
  Request 1:
    TTFT          : 7.972 s
    Latency       : 7.972 s
    Throughput    : 43.15 tokens/s
    Prompt tokens : 60, Output tokens: 344
  Request 2:
    TTFT          : 16.156 s
    Latency       : 16.156 s
    Throughput    : 24.08 tokens/s
    Prompt tokens : 60, Output tokens: 389

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 12.064 s
    Average throughput per req   : 33.61 tokens/s
    Overall throughput (sum)     : 67.23 tokens/s
    Batch duration (wall-clock)  : 16.182 s

Scenario: Sentiment Analysis / Classification, Concurrency: 1
  Request 1:
    TTFT          : 1.241 s
    Latency       : 1.241 s
    Throughput    : 12.89 tokens/s
    Prompt tokens : 82, Output tokens: 16

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.241 s
    Average throughput per req   : 12.89 tokens/s
    Overall throughput (sum)     : 12.89 tokens/s
    Batch duration (wall-clock)  : 1.258 s

Scenario: Sentiment Analysis / Classification, Concurrency: 2
  Request 1:
    TTFT          : 1.035 s
    Latency       : 1.035 s
    Throughput    : 6.76 tokens/s
    Prompt tokens : 82, Output tokens: 7
  Request 2:
    TTFT          : 1.423 s
    Latency       : 1.423 s
    Throughput    : 9.84 tokens/s
    Prompt tokens : 82, Output tokens: 14

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 1.229 s
    Average throughput per req   : 8.30 tokens/s
    Overall throughput (sum)     : 16.60 tokens/s
    Batch duration (wall-clock)  : 1.447 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 1
  Request 1:
    TTFT          : 19.793 s
    Latency       : 19.793 s
    Throughput    : 47.29 tokens/s
    Prompt tokens : 99, Output tokens: 936

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 19.793 s
    Average throughput per req   : 47.29 tokens/s
    Overall throughput (sum)     : 47.29 tokens/s
    Batch duration (wall-clock)  : 19.812 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 2
Attempt 1 failed: The read operation timed out
  Request 1:
    TTFT          : 19.782 s
    Latency       : 19.782 s
    Throughput    : 46.76 tokens/s
    Prompt tokens : 99, Output tokens: 925
  Request 2:
    TTFT          : 25.978 s
    Latency       : 25.978 s
    Throughput    : 42.04 tokens/s
    Prompt tokens : 99, Output tokens: 1092

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 22.880 s
    Average throughput per req   : 44.40 tokens/s
    Overall throughput (sum)     : 88.79 tokens/s
    Batch duration (wall-clock)  : 57.649 s
```

**on NC-80 VM**

```
Scenario: Text Generation, Concurrency: 1
  Request 1:
    TTFT          : 8.326 s
    Latency       : 8.326 s
    Throughput    : 108.09 tokens/s
    Prompt tokens : 132, Output tokens: 900

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.326 s
    Average throughput per req   : 108.09 tokens/s
    Overall throughput (sum)     : 108.09 tokens/s
    Batch duration (wall-clock)  : 8.351 s

Scenario: Text Generation, Concurrency: 2
  Request 1:
    TTFT          : 9.151 s
    Latency       : 9.151 s
    Throughput    : 112.01 tokens/s
    Prompt tokens : 132, Output tokens: 1025
  Request 2:
    TTFT          : 16.887 s
    Latency       : 16.887 s
    Throughput    : 55.66 tokens/s
    Prompt tokens : 132, Output tokens: 940

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 13.019 s
    Average throughput per req   : 83.84 tokens/s
    Overall throughput (sum)     : 167.68 tokens/s
    Batch duration (wall-clock)  : 16.920 s

Scenario: Text Generation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 8.049 s
    Latency       : 8.049 s
    Throughput    : 108.09 tokens/s
    Prompt tokens : 132, Output tokens: 870
  Request 2:
    TTFT          : 14.843 s
    Latency       : 14.843 s
    Throughput    : 56.46 tokens/s
    Prompt tokens : 132, Output tokens: 838

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 11.446 s
    Average throughput per req   : 82.27 tokens/s
    Overall throughput (sum)     : 164.54 tokens/s
    Batch duration (wall-clock)  : 14.885 s

Scenario: Question Answering, Concurrency: 1
  Request 1:
    TTFT          : 4.424 s
    Latency       : 4.424 s
    Throughput    : 96.51 tokens/s
    Prompt tokens : 114, Output tokens: 427

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 4.424 s
    Average throughput per req   : 96.51 tokens/s
    Overall throughput (sum)     : 96.51 tokens/s
    Batch duration (wall-clock)  : 4.446 s

Scenario: Question Answering, Concurrency: 2
  Request 1:
    TTFT          : 5.892 s
    Latency       : 5.892 s
    Throughput    : 105.56 tokens/s
    Prompt tokens : 114, Output tokens: 622
  Request 2:
    TTFT          : 11.598 s
    Latency       : 11.598 s
    Throughput    : 61.05 tokens/s
    Prompt tokens : 114, Output tokens: 708

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 8.745 s
    Average throughput per req   : 83.30 tokens/s
    Overall throughput (sum)     : 166.61 tokens/s
    Batch duration (wall-clock)  : 11.630 s

Scenario: Question Answering, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 6.059 s
    Latency       : 6.059 s
    Throughput    : 104.98 tokens/s
    Prompt tokens : 114, Output tokens: 636
  Request 2:
    TTFT          : 9.758 s
    Latency       : 9.758 s
    Throughput    : 46.43 tokens/s
    Prompt tokens : 114, Output tokens: 453

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 7.908 s
    Average throughput per req   : 75.70 tokens/s
    Overall throughput (sum)     : 151.40 tokens/s
    Batch duration (wall-clock)  : 9.798 s

Scenario: Translation, Concurrency: 1
  Request 1:
    TTFT          : 3.574 s
    Latency       : 3.574 s
    Throughput    : 89.52 tokens/s
    Prompt tokens : 85, Output tokens: 320

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.574 s
    Average throughput per req   : 89.52 tokens/s
    Overall throughput (sum)     : 89.52 tokens/s
    Batch duration (wall-clock)  : 3.595 s

Scenario: Translation, Concurrency: 2
  Request 1:
    TTFT          : 1.894 s
    Latency       : 1.894 s
    Throughput    : 60.73 tokens/s
    Prompt tokens : 85, Output tokens: 115
  Request 2:
    TTFT          : 4.167 s
    Latency       : 4.167 s
    Throughput    : 63.36 tokens/s
    Prompt tokens : 85, Output tokens: 264

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.030 s
    Average throughput per req   : 62.05 tokens/s
    Overall throughput (sum)     : 124.09 tokens/s
    Batch duration (wall-clock)  : 4.194 s

Scenario: Translation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
  Request 1:
    TTFT          : 3.937 s
    Latency       : 3.937 s
    Throughput    : 92.20 tokens/s
    Prompt tokens : 85, Output tokens: 363
  Request 2:
    TTFT          : 5.795 s
    Latency       : 5.795 s
    Throughput    : 37.10 tokens/s
    Prompt tokens : 85, Output tokens: 215
  Request 3:
    TTFT          : 2.115 s
    Latency       : 2.115 s
    Throughput    : 59.10 tokens/s
    Prompt tokens : 85, Output tokens: 125

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 3.949 s
    Average throughput per req   : 62.80 tokens/s
    Overall throughput (sum)     : 188.40 tokens/s
    Batch duration (wall-clock)  : 6.951 s

Scenario: Text Summarization, Concurrency: 1
  Request 1:
    TTFT          : 1.859 s
    Latency       : 1.859 s
    Throughput    : 57.55 tokens/s
    Prompt tokens : 90, Output tokens: 107

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.859 s
    Average throughput per req   : 57.55 tokens/s
    Overall throughput (sum)     : 57.55 tokens/s
    Batch duration (wall-clock)  : 1.880 s

Scenario: Text Summarization, Concurrency: 2
  Request 1:
    TTFT          : 2.677 s
    Latency       : 2.677 s
    Throughput    : 82.93 tokens/s
    Prompt tokens : 90, Output tokens: 222
  Request 2:
    TTFT          : 3.854 s
    Latency       : 3.854 s
    Throughput    : 33.21 tokens/s
    Prompt tokens : 90, Output tokens: 128

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.266 s
    Average throughput per req   : 58.07 tokens/s
    Overall throughput (sum)     : 116.14 tokens/s
    Batch duration (wall-clock)  : 3.883 s

Scenario: Text Summarization, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
  Request 1:
    TTFT          : 2.062 s
    Latency       : 2.062 s
    Throughput    : 65.00 tokens/s
    Prompt tokens : 90, Output tokens: 134
  Request 2:
    TTFT          : 4.546 s
    Latency       : 4.546 s
    Throughput    : 64.89 tokens/s
    Prompt tokens : 90, Output tokens: 295
  Request 3:
    TTFT          : 3.703 s
    Latency       : 3.703 s
    Throughput    : 30.51 tokens/s
    Prompt tokens : 90, Output tokens: 113

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 3.437 s
    Average throughput per req   : 53.47 tokens/s
    Overall throughput (sum)     : 160.40 tokens/s
    Batch duration (wall-clock)  : 5.625 s

Scenario: Code Generation, Concurrency: 1
  Request 1:
    TTFT          : 13.096 s
    Latency       : 13.096 s
    Throughput    : 130.43 tokens/s
    Prompt tokens : 79, Output tokens: 1708

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 13.096 s
    Average throughput per req   : 130.43 tokens/s
    Overall throughput (sum)     : 130.43 tokens/s
    Batch duration (wall-clock)  : 13.120 s

Scenario: Code Generation, Concurrency: 2
  Request 1:
    TTFT          : 13.197 s
    Latency       : 13.197 s
    Throughput    : 130.18 tokens/s
    Prompt tokens : 79, Output tokens: 1718
  Request 2:
    TTFT          : 20.536 s
    Latency       : 20.536 s
    Throughput    : 49.81 tokens/s
    Prompt tokens : 79, Output tokens: 1023

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 16.867 s
    Average throughput per req   : 90.00 tokens/s
    Overall throughput (sum)     : 179.99 tokens/s
    Batch duration (wall-clock)  : 20.565 s

Scenario: Code Generation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 11.143 s
    Latency       : 11.143 s
    Throughput    : 126.99 tokens/s
    Prompt tokens : 79, Output tokens: 1415
  Request 2:
    TTFT          : 20.682 s
    Latency       : 20.682 s
    Throughput    : 63.34 tokens/s
    Prompt tokens : 79, Output tokens: 1310

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 15.912 s
    Average throughput per req   : 95.17 tokens/s
    Overall throughput (sum)     : 190.33 tokens/s
    Batch duration (wall-clock)  : 20.721 s

Scenario: Chatbot, Concurrency: 1
  Request 1:
    TTFT          : 3.584 s
    Latency       : 3.584 s
    Throughput    : 92.08 tokens/s
    Prompt tokens : 60, Output tokens: 330

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.584 s
    Average throughput per req   : 92.08 tokens/s
    Overall throughput (sum)     : 92.08 tokens/s
    Batch duration (wall-clock)  : 3.604 s

Scenario: Chatbot, Concurrency: 2
  Request 1:
    TTFT          : 4.119 s
    Latency       : 4.119 s
    Throughput    : 97.85 tokens/s
    Prompt tokens : 60, Output tokens: 403
  Request 2:
    TTFT          : 6.767 s
    Latency       : 6.767 s
    Throughput    : 47.88 tokens/s
    Prompt tokens : 60, Output tokens: 324

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 5.443 s
    Average throughput per req   : 72.86 tokens/s
    Overall throughput (sum)     : 145.73 tokens/s
    Batch duration (wall-clock)  : 6.797 s

Scenario: Chatbot, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
  Request 1:
    TTFT          : 5.044 s
    Latency       : 5.044 s
    Throughput    : 101.51 tokens/s
    Prompt tokens : 60, Output tokens: 512
  Request 2:
    TTFT          : 8.897 s
    Latency       : 8.897 s
    Throughput    : 53.17 tokens/s
    Prompt tokens : 60, Output tokens: 473
  Request 3:
    TTFT          : 7.343 s
    Latency       : 7.343 s
    Throughput    : 53.52 tokens/s
    Prompt tokens : 60, Output tokens: 393

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 7.094 s
    Average throughput per req   : 69.40 tokens/s
    Overall throughput (sum)     : 208.20 tokens/s
    Batch duration (wall-clock)  : 12.142 s

Scenario: Sentiment Analysis / Classification, Concurrency: 1
  Request 1:
    TTFT          : 1.056 s
    Latency       : 1.056 s
    Throughput    : 7.58 tokens/s
    Prompt tokens : 82, Output tokens: 8

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.056 s
    Average throughput per req   : 7.58 tokens/s
    Overall throughput (sum)     : 7.58 tokens/s
    Batch duration (wall-clock)  : 1.076 s

Scenario: Sentiment Analysis / Classification, Concurrency: 2
  Request 1:
    TTFT          : 1.127 s
    Latency       : 1.127 s
    Throughput    : 20.41 tokens/s
    Prompt tokens : 82, Output tokens: 23
  Request 2:
    TTFT          : 1.330 s
    Latency       : 1.330 s
    Throughput    : 9.02 tokens/s
    Prompt tokens : 82, Output tokens: 12

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 1.228 s
    Average throughput per req   : 14.72 tokens/s
    Overall throughput (sum)     : 29.44 tokens/s
    Batch duration (wall-clock)  : 1.356 s

Scenario: Sentiment Analysis / Classification, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
  Request 1:
    TTFT          : 1.080 s
    Latency       : 1.080 s
    Throughput    : 13.89 tokens/s
    Prompt tokens : 82, Output tokens: 15
  Request 2:
    TTFT          : 1.268 s
    Latency       : 1.268 s
    Throughput    : 7.88 tokens/s
    Prompt tokens : 82, Output tokens: 10
  Request 3:
    TTFT          : 1.091 s
    Latency       : 1.091 s
    Throughput    : 13.74 tokens/s
    Prompt tokens : 82, Output tokens: 15

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 1.146 s
    Average throughput per req   : 11.84 tokens/s
    Overall throughput (sum)     : 35.52 tokens/s
    Batch duration (wall-clock)  : 3.027 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 1
  Request 1:
    TTFT          : 8.508 s
    Latency       : 8.508 s
    Throughput    : 115.18 tokens/s
    Prompt tokens : 99, Output tokens: 980

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.508 s
    Average throughput per req   : 115.18 tokens/s
    Overall throughput (sum)     : 115.18 tokens/s
    Batch duration (wall-clock)  : 8.530 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 2
  Request 1:
    TTFT          : 7.832 s
    Latency       : 7.832 s
    Throughput    : 112.74 tokens/s
    Prompt tokens : 99, Output tokens: 883
  Request 2:
    TTFT          : 16.772 s
    Latency       : 16.772 s
    Throughput    : 67.26 tokens/s
    Prompt tokens : 99, Output tokens: 1128

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 12.302 s
    Average throughput per req   : 90.00 tokens/s
    Overall throughput (sum)     : 179.99 tokens/s
    Batch duration (wall-clock)  : 16.804 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 10.427 s
    Latency       : 10.427 s
    Throughput    : 121.12 tokens/s
    Prompt tokens : 99, Output tokens: 1263
  Request 2:
    TTFT          : 19.216 s
    Latency       : 19.216 s
    Throughput    : 57.71 tokens/s
    Prompt tokens : 99, Output tokens: 1109

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 14.822 s
    Average throughput per req   : 89.42 tokens/s
    Overall throughput (sum)     : 178.84 tokens/s
    Batch duration (wall-clock)  : 19.253 s
```

**On NC-40 VM**

```
Tokenizer loaded successfully: microsoft/phi-4

Scenario: Text Generation, Concurrency: 1
  Request 1:
    TTFT          : 12.568 s
    Latency       : 12.568 s
    Throughput    : 78.21 tokens/s
    Prompt tokens : 132, Output tokens: 983

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 12.568 s
    Average throughput per req   : 78.21 tokens/s
    Overall throughput (sum)     : 78.21 tokens/s
    Batch duration (wall-clock)  : 12.595 s

Scenario: Text Generation, Concurrency: 2
  Request 1:
    TTFT          : 13.294 s
    Latency       : 13.294 s
    Throughput    : 77.63 tokens/s
    Prompt tokens : 132, Output tokens: 1032
  Request 2:
    TTFT          : 25.377 s
    Latency       : 25.377 s
    Throughput    : 39.56 tokens/s
    Prompt tokens : 132, Output tokens: 1004

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 19.335 s
    Average throughput per req   : 58.60 tokens/s
    Overall throughput (sum)     : 117.19 tokens/s
    Batch duration (wall-clock)  : 25.408 s

Scenario: Text Generation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 11.864 s
    Latency       : 11.864 s
    Throughput    : 76.87 tokens/s
    Prompt tokens : 132, Output tokens: 912
  Request 2:
    TTFT          : 22.267 s
    Latency       : 22.267 s
    Throughput    : 38.49 tokens/s
    Prompt tokens : 132, Output tokens: 857

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 17.066 s
    Average throughput per req   : 57.68 tokens/s
    Overall throughput (sum)     : 115.36 tokens/s
    Batch duration (wall-clock)  : 22.307 s

Scenario: Question Answering, Concurrency: 1
  Request 1:
    TTFT          : 8.698 s
    Latency       : 8.698 s
    Throughput    : 75.30 tokens/s
    Prompt tokens : 114, Output tokens: 655

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.698 s
    Average throughput per req   : 75.30 tokens/s
    Overall throughput (sum)     : 75.30 tokens/s
    Batch duration (wall-clock)  : 8.719 s

Scenario: Question Answering, Concurrency: 2
  Request 1:
    TTFT          : 9.585 s
    Latency       : 9.585 s
    Throughput    : 75.85 tokens/s
    Prompt tokens : 114, Output tokens: 727
  Request 2:
    TTFT          : 15.909 s
    Latency       : 15.909 s
    Throughput    : 32.25 tokens/s
    Prompt tokens : 114, Output tokens: 513

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 12.747 s
    Average throughput per req   : 54.05 tokens/s
    Overall throughput (sum)     : 108.10 tokens/s
    Batch duration (wall-clock)  : 15.939 s

Scenario: Question Answering, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 9.000 s
    Latency       : 9.000 s
    Throughput    : 75.88 tokens/s
    Prompt tokens : 114, Output tokens: 683
  Request 2:
    TTFT          : 18.007 s
    Latency       : 18.007 s
    Throughput    : 40.32 tokens/s
    Prompt tokens : 114, Output tokens: 726

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 13.504 s
    Average throughput per req   : 58.10 tokens/s
    Overall throughput (sum)     : 116.20 tokens/s
    Batch duration (wall-clock)  : 18.046 s

Scenario: Translation, Concurrency: 1
  Request 1:
    TTFT          : 3.858 s
    Latency       : 3.858 s
    Throughput    : 62.20 tokens/s
    Prompt tokens : 85, Output tokens: 240

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 3.858 s
    Average throughput per req   : 62.20 tokens/s
    Overall throughput (sum)     : 62.20 tokens/s
    Batch duration (wall-clock)  : 3.878 s

Scenario: Translation, Concurrency: 2
  Request 1:
    TTFT          : 2.163 s
    Latency       : 2.163 s
    Throughput    : 46.24 tokens/s
    Prompt tokens : 85, Output tokens: 100
  Request 2:
    TTFT          : 4.993 s
    Latency       : 4.993 s
    Throughput    : 45.46 tokens/s
    Prompt tokens : 85, Output tokens: 227

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.578 s
    Average throughput per req   : 45.85 tokens/s
    Overall throughput (sum)     : 91.70 tokens/s
    Batch duration (wall-clock)  : 5.020 s

Scenario: Translation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
  Request 1:
    TTFT          : 2.320 s
    Latency       : 2.320 s
    Throughput    : 48.28 tokens/s
    Prompt tokens : 85, Output tokens: 112
  Request 2:
    TTFT          : 3.853 s
    Latency       : 3.853 s
    Throughput    : 31.14 tokens/s
    Prompt tokens : 85, Output tokens: 120
  Request 3:
    TTFT          : 3.404 s
    Latency       : 3.404 s
    Throughput    : 32.61 tokens/s
    Prompt tokens : 85, Output tokens: 111

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 3.192 s
    Average throughput per req   : 37.34 tokens/s
    Overall throughput (sum)     : 112.03 tokens/s
    Batch duration (wall-clock)  : 5.330 s

Scenario: Text Summarization, Concurrency: 1
  Request 1:
    TTFT          : 5.796 s
    Latency       : 5.796 s
    Throughput    : 70.73 tokens/s
    Prompt tokens : 90, Output tokens: 410

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 5.796 s
    Average throughput per req   : 70.73 tokens/s
    Overall throughput (sum)     : 70.73 tokens/s
    Batch duration (wall-clock)  : 5.816 s

Scenario: Text Summarization, Concurrency: 2
  Request 1:
    TTFT          : 2.320 s
    Latency       : 2.320 s
    Throughput    : 47.84 tokens/s
    Prompt tokens : 90, Output tokens: 111
  Request 2:
    TTFT          : 4.206 s
    Latency       : 4.206 s
    Throughput    : 35.43 tokens/s
    Prompt tokens : 90, Output tokens: 149

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 3.263 s
    Average throughput per req   : 41.63 tokens/s
    Overall throughput (sum)     : 83.27 tokens/s
    Batch duration (wall-clock)  : 4.236 s

Scenario: Text Summarization, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
  Request 1:
    TTFT          : 2.357 s
    Latency       : 2.357 s
    Throughput    : 49.21 tokens/s
    Prompt tokens : 90, Output tokens: 116
  Request 2:
    TTFT          : 3.755 s
    Latency       : 3.755 s
    Throughput    : 28.50 tokens/s
    Prompt tokens : 90, Output tokens: 107
  Request 3:
    TTFT          : 3.777 s
    Latency       : 3.777 s
    Throughput    : 39.71 tokens/s
    Prompt tokens : 90, Output tokens: 150

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 3.296 s
    Average throughput per req   : 39.14 tokens/s
    Overall throughput (sum)     : 117.42 tokens/s
    Batch duration (wall-clock)  : 5.694 s

Scenario: Code Generation, Concurrency: 1
  Request 1:
    TTFT          : 13.112 s
    Latency       : 13.112 s
    Throughput    : 85.19 tokens/s
    Prompt tokens : 79, Output tokens: 1117

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 13.112 s
    Average throughput per req   : 85.19 tokens/s
    Overall throughput (sum)     : 85.19 tokens/s
    Batch duration (wall-clock)  : 13.133 s

Scenario: Code Generation, Concurrency: 2
  Request 1:
    TTFT          : 14.930 s
    Latency       : 14.930 s
    Throughput    : 86.47 tokens/s
    Prompt tokens : 79, Output tokens: 1291
  Request 2:
    TTFT          : 32.439 s
    Latency       : 32.439 s
    Throughput    : 49.23 tokens/s
    Prompt tokens : 79, Output tokens: 1597

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 23.685 s
    Average throughput per req   : 67.85 tokens/s
    Overall throughput (sum)     : 135.70 tokens/s
    Batch duration (wall-clock)  : 32.469 s

Scenario: Code Generation, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 17.627 s
    Latency       : 17.627 s
    Throughput    : 87.31 tokens/s
    Prompt tokens : 79, Output tokens: 1539
  Request 2:
    TTFT          : 31.470 s
    Latency       : 31.470 s
    Throughput    : 39.97 tokens/s
    Prompt tokens : 79, Output tokens: 1258

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 24.549 s
    Average throughput per req   : 63.64 tokens/s
    Overall throughput (sum)     : 127.28 tokens/s
    Batch duration (wall-clock)  : 31.512 s

Scenario: Chatbot, Concurrency: 1
  Request 1:
    TTFT          : 6.060 s
    Latency       : 6.060 s
    Throughput    : 70.30 tokens/s
    Prompt tokens : 60, Output tokens: 426

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 6.060 s
    Average throughput per req   : 70.30 tokens/s
    Overall throughput (sum)     : 70.30 tokens/s
    Batch duration (wall-clock)  : 6.082 s

Scenario: Chatbot, Concurrency: 2
  Request 1:
    TTFT          : 5.218 s
    Latency       : 5.218 s
    Throughput    : 69.18 tokens/s
    Prompt tokens : 60, Output tokens: 361
  Request 2:
    TTFT          : 8.812 s
    Latency       : 8.812 s
    Throughput    : 33.48 tokens/s
    Prompt tokens : 60, Output tokens: 295

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 7.015 s
    Average throughput per req   : 51.33 tokens/s
    Overall throughput (sum)     : 102.66 tokens/s
    Batch duration (wall-clock)  : 8.841 s

Scenario: Chatbot, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
  Request 1:
    TTFT          : 4.813 s
    Latency       : 4.813 s
    Throughput    : 67.73 tokens/s
    Prompt tokens : 60, Output tokens: 326
  Request 2:
    TTFT          : 8.715 s
    Latency       : 8.715 s
    Throughput    : 36.83 tokens/s
    Prompt tokens : 60, Output tokens: 321
  Request 3:
    TTFT          : 8.423 s
    Latency       : 8.423 s
    Throughput    : 43.81 tokens/s
    Prompt tokens : 60, Output tokens: 369

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 7.317 s
    Average throughput per req   : 49.46 tokens/s
    Overall throughput (sum)     : 148.37 tokens/s
    Batch duration (wall-clock)  : 13.227 s

Scenario: Sentiment Analysis / Classification, Concurrency: 1
  Request 1:
    TTFT          : 1.147 s
    Latency       : 1.147 s
    Throughput    : 16.56 tokens/s
    Prompt tokens : 82, Output tokens: 19

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 1.147 s
    Average throughput per req   : 16.56 tokens/s
    Overall throughput (sum)     : 16.56 tokens/s
    Batch duration (wall-clock)  : 1.166 s

Scenario: Sentiment Analysis / Classification, Concurrency: 2
  Request 1:
    TTFT          : 1.057 s
    Latency       : 1.057 s
    Throughput    : 8.51 tokens/s
    Prompt tokens : 82, Output tokens: 9
  Request 2:
    TTFT          : 1.297 s
    Latency       : 1.297 s
    Throughput    : 10.02 tokens/s
    Prompt tokens : 82, Output tokens: 13

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 1.177 s
    Average throughput per req   : 9.27 tokens/s
    Overall throughput (sum)     : 18.54 tokens/s
    Batch duration (wall-clock)  : 1.324 s

Scenario: Sentiment Analysis / Classification, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
  Request 1:
    TTFT          : 1.151 s
    Latency       : 1.151 s
    Throughput    : 14.77 tokens/s
    Prompt tokens : 82, Output tokens: 17
  Request 2:
    TTFT          : 1.417 s
    Latency       : 1.417 s
    Throughput    : 11.29 tokens/s
    Prompt tokens : 82, Output tokens: 16
  Request 3:
    TTFT          : 1.144 s
    Latency       : 1.144 s
    Throughput    : 13.99 tokens/s
    Prompt tokens : 82, Output tokens: 16

  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 1.237 s
    Average throughput per req   : 13.35 tokens/s
    Overall throughput (sum)     : 40.06 tokens/s
    Batch duration (wall-clock)  : 3.064 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 1
  Request 1:
    TTFT          : 15.531 s
    Latency       : 15.531 s
    Throughput    : 80.03 tokens/s
    Prompt tokens : 99, Output tokens: 1243

  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 15.531 s
    Average throughput per req   : 80.03 tokens/s
    Overall throughput (sum)     : 80.03 tokens/s
    Batch duration (wall-clock)  : 15.554 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 2
  Request 1:                                                 
    TTFT          : 12.332 s
    Latency       : 12.332 s
    Throughput    : 80.53 tokens/s
    Prompt tokens : 99, Output tokens: 993
  Request 2:
    TTFT          : 25.583 s
    Latency       : 25.583 s
    Throughput    : 43.00 tokens/s
    Prompt tokens : 99, Output tokens: 1100

  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 18.957 s
    Average throughput per req   : 61.76 tokens/s
    Overall throughput (sum)     : 123.52 tokens/s
    Batch duration (wall-clock)  : 25.615 s

Scenario: Multi-turn Reasoning / Complex Tasks, Concurrency: 3
Attempt 1: Received 429 Too Many Requests. Backing off for 1 seconds.
Attempt 2: Received 429 Too Many Requests. Backing off for 2 seconds.
Attempt 3: Received 429 Too Many Requests. Backing off for 4 seconds.
  Request 1:
    TTFT          : 14.753 s
    Latency       : 14.753 s
    Throughput    : 77.74 tokens/s
    Prompt tokens : 99, Output tokens: 1147
  Request 2:
    TTFT          : 24.854 s
    Latency       : 24.854 s
    Throughput    : 34.80 tokens/s
    Prompt tokens : 99, Output tokens: 865

  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 19.804 s
    Average throughput per req   : 56.27 tokens/s
    Overall throughput (sum)     : 112.55 tokens/s
    Batch duration (wall-clock)  : 24.899 s
```

