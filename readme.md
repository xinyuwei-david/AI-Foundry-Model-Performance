# AI Foundry Model Catalog Model performance

This repository is designed to test the performance of open-source models from the Azure Machine Learning Model Catalog. 



## Deploying models Methods

https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/deployments-overview

| Name                          | Azure OpenAI service                                         | Azure AI model inference                                     | Serverless API                                               | Managed compute                                              |
| :---------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Which models can be deployed? | [Azure OpenAI models](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) | [Azure OpenAI models and Models as a Service](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models) | [Models as a Service](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/model-catalog-overview#content-safety-for-models-deployed-via-serverless-apis) | [Open and custom models](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/model-catalog-overview#availability-of-models-for-deployment-as-managed-compute) |
| Deployment resource           | Azure OpenAI resource                                        | Azure AI services resource                                   | AI project resource                                          | AI project resource                                          |
| Best suited when              | You are planning to use only OpenAI models                   | You are planning to take advantage of the flagship models in Azure AI catalog, including OpenAI. | You are planning to use a single model from a specific provider (excluding OpenAI). | If you plan to use open models and you have enough compute quota available in your subscription. |
| Billing bases                 | Token usage & PTU                                            | Token usage                                                  | Token usage                                                  | Compute core hours                                           |
| Deployment instructions       | [Deploy to Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/deploy-models-openai) | [Deploy to Azure AI model inference](https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/how-to/create-model-deployments) | [Deploy to Serverless API](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/deploy-models-serverless) | [Deploy to Managed compute](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/deploy-models-managed) |

Currently, an increasing number of new flagship models in the Azure AI Foundry model catalog, including OpenAI, will be deployed using the Azure AI model inference method. Models deployed in this way can be accessed via the AI Inference SDK (which now supports stream mode: https://learn.microsoft.com/en-us/python/api/overview/azure/ai-inference-readme?view=azure-python-preview). Open-source models include DeepSeek R1, V3, Phi, Mistral, and more. For a detailed list of models, please refer to:

***https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models***

If you care about the performance data of this method, please skip to the last section of this repo.



## Performance test of AI models in Azure Machine Learning

In this section, we focus on the models deployed on Managed Compute in the Model Catalogue on AML.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/19.png)

Next, we will use a Python script to automate the deployment of the model and use another program to evaluate the model's performance.

### Fast Deploy AI Model on AML Model Catalog via Azure GPU VM

By now, the AML names tested in this repo, their full names on Hugging Face, and the Azure GPU VM SKUs that can be deployed on AML are as follows: 

| **Model Name on AML**                         | **Model on HF**                               | **Azure GPU VM SKU**                             |
| --------------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| Phi-4                                         | microsoft/phi-4                               | NC24/48/96 A100                                  |
| Phi-3.5-vision-instruct                       | microsoft/Phi-3.5-vision-instruct             | NC24/48/96 A100                                  |
| financial-reports-analysis                    |                                               | NC24/48/96 A100                                  |
| Llama-3.2-11B-Vision-Instruct                 | meta-llama/Llama-3.2-11B-Vision-Instruct      | NC24/48/96 A100                                  |
| Phi-3-small-8k-instruct                       | microsoft/Phi-3-small-8k-instruct             | NC24/48/96 A100                                  |
| Phi-3-vision-128k-instruct                    | microsoft/Phi-3-vision-128k-instruct          | NC48 A100 or NC96 A100                           |
| microsoft-swinv2-base-patch4-window12-192-22k | microsoft/swinv2-base-patch4-window12-192-22k | NC24/48/96 A100                                  |
| mistralai-Mixtral-8x7B-Instruct-v01           | mistralai/Mixtral-8x7B-Instruct-v0.1          | NC24/48/96 A100                                  |
| Muse                                          | microsoft/wham                                | NC24/48/96 A100                                  |
| openai-whisper-large                          | openai/whisper-large                          | NC48 A100 or NC96 A100                           |
| snowflake-arctic-base                         | Snowflake/snowflake-arctic-base               | NC24/48/96 A100                                  |
| Nemotron-3-8B-Chat-4k-SteerLM                 | nvidia/nemotron-3-8b-chat-4k-steerlm          | NC24/48/96 A100                                  |
| stabilityai-stable-diffusion-xl-refiner-1-0   | stabilityai/stable-diffusion-xl-refiner-1.0   | Standard_ND96amsr_A100_v4 or Standard_ND96asr_v4 |
| microsoft-Orca-2-7b                           | microsoft/Orca-2-7b                           | NC24/48/96 A100                                  |



**Clone code and prepare shell environment**

First, you need to create an Azure Machine Learning service in the Azure Portal. When selecting the region for the service, you should choose a region under the AML category in your subscription quota that has a GPU VM quota available.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/20.png)

Next, find a shell environment where you can execute `az login` to log in to your Azure subscription.

```
#git clone https://github.com/xinyuwei-david/AI-Foundry-Model-Performance.git
#conda create -n aml_env python=3.9 -y
#conda activate aml_env
#cd AI-Foundry-Model-Performance
#pip install -r requirements.txt  
```

==Backup info for developing========

#cat requirements.txt  

```
azure-ai-ml  
azure-identity  
requests  
pyyaml  
tabulate  
torch
transformers
tiktoken
```

Login to Azure.

```
#curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash  
#az login --use-device
```

**Deploy model Automatically**

Next, you need to execute a script for end-to-end model deployment. This script will: 

- Help you check the GPU VM quota for AML under your subscription
- Prompt you to select the model you want to deploy
- Specify the Azure GPU VM SKU and quantity to be used for deployment. 
- Provide you with the endpoint and key of the successfully deployed model, allowing you to proceed with performance testing. 

```
#python deploymodels-linux.py
```

If you do test on powershell,  you should use:

```
#python deploymodels-powershell.py
```



The deploy process:

```
========== Enter Basic Information ==========
Subscription ID: 53039473-9bbd-499d-90d7-d046d4fa63b6
Resource Group: AIrg1
Workspace Name: aml-david-1

========== Model Name Examples ==========
 - Phi-4
 - Phi-3.5-vision-instruct
 - financial-reports-analysis
 - databricks-dbrx-instruct
 - Llama-3.2-11B-Vision-Instruct
 - Phi-3-small-8k-instruct
 - Phi-3-vision-128k-instruct
 - microsoft-swinv2-base-patch4-window12-192-22k
 - mistralai-Mixtral-8x7B-Instruct-v01
 - Muse
 - openai-whisper-large
 - snowflake-arctic-base
 - Nemotron-3-8B-Chat-4k-SteerLM
 - stabilityai-stable-diffusion-xl-refiner-1-0
 - microsoft-Orca-2-7b
==========================================

Enter the model name to search (e.g., 'Phi-4'): Phi-4

========== Matching Models ==========
Name                       Description    Latest version
-------------------------  -------------  ----------------
Phi-4-multimodal-instruct                 1
Phi-4-mini-instruct                       1
Phi-4                                     7

Note: The above table is for reference only. Enter the exact model name below:
Enter full model name (case-sensitive): Phi-4
Enter model version (e.g., 7): 7
2025-03-13 15:42:02,438 - INFO - User-specified model: name='Phi-4', version='7'

========== GPU Quota (Limit > 1) ==========
Region,ResourceName,LocalizedValue,Usage,Limit
westeurope,standardNCADSH100v5Family,,0,100
polandcentral,standardNCADSA100v4Family,,0,100

========== A100 / H100 SKU Information ==========
SKU Name                            GPU Count  GPU Memory (VRAM)    CPU Cores
----------------------------------- ---------- -------------------- ----------
Standard_NC24ads_A100_v4            1          80 GB                24
Standard_NC48ads_A100_v4            2          1600 GB (2x80 GB)    48
Standard_NC96ads_A100_v4            4          320 GB (4x80 GB)     96
Standard_NC40ads_H100_v5            1          80 GB                40
Standard_NC80ads_H100_v5            2          160 GB (2x80 GB)     80

Available SKUs:
 - Standard_NC24ads_A100_v4
 - Standard_NC48ads_A100_v4
 - Standard_NC96ads_A100_v4
 - Standard_NC40ads_H100_v5
 - Standard_NC80ads_H100_v5

Enter the SKU to use: Standard_NC24ads_A100_v4
Enter the number of instances (integer): 1
2025-03-13 15:52:42,333 - INFO - Model ID: azureml://registries/AzureML/models/Phi-4/versions/7
2025-03-13 15:52:42,333 - INFO - No environment configuration found.
2025-03-13 15:52:42,366 - INFO - ManagedIdentityCredential will use IMDS
2025-03-13 15:52:42,379 - INFO - Creating Endpoint: custom-endpoint-1741852362
2025-03-13 15:52:43,008 - INFO - Request URL: 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=REDACTED&resource=REDACTED'
```

After 3-5 minutes, you will get the final results:

```
----- Deployment Information -----
ENDPOINT_NAME=custom-endpoint-1741863106
SCORING_URI=https://custom-endpoint-1741863106.polandcentral.inference.ml.azure.com/score
PRIMARY_KEY=DRxHMd1jbbSdNoXiYOaWRQ66erYZfejzKhdyDVRuh58v2hXILOcYJQQJ99BCAAAAAAAAAAAAINFRAZML3m1v
SECONDARY_KEY=4dhy3og6WfVzkIijMU7FFUDLpz4WIWEYgIlXMGYUzgwafsW6GPrMJQQJ99BCAAAAAAAAAAAAINFRAZMLxOpO
```



###  Fast Performance Test AI Model on AML Model Catalog

The primary goal of performance testing is to verify tokens/s and TTFT during the inference process. To better simulate real-world scenarios, I have set up several common LLM/SLM use cases in the test script. Additionally, to ensure tokens/s performance, the test script needs to load the corresponding model's tokenizer during execution.



Before officially starting the test, you need to log in to HF on your terminal.

```
huggingface-cli  login
```

#### Phi Text2Text Series 

**Run the test script:**

```
(aml_env) root@pythonvm:~/AIFperformance# python press-phi4-0314.py
Please enter the API service URL: https://david-workspace-westeurop-ldvdq.westeurope.inference.ml.azure.com/score
Please enter the API Key: Ef9DFpATsXs4NiWyoVhEXeR4PWPvFy17xcws5ySCvV2H8uOUfgV4JQQJ99BCAAAAAAAAAAAAINFRAZML3eIO
Please enter the full name of the HuggingFace model for tokenizer loading: microsoft/phi-4
Tokenizer loaded successfully: microsoft/phi-4
```

**Test result analyze：**

**microsoft/phi-4**

Concurrency = 1

| Scenario                 | Concurrency | VM 1 (1-nc48) TTFT (s) | VM 2 (2-nc24) TTFT (s) | VM 3 (1-nc24) TTFT (s) | VM 1 (1-nc48) tokens/s | VM 2 (2-nc24) tokens/s | VM 3 (1-nc24) tokens/s |
| ------------------------ | ----------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| **Text Generation**      | 1           | 12.473                 | 19.546                 | 19.497                 | 68.07                  | 44.66                  | 44.78                  |
| **Question Answering**   | 1           | 11.914                 | 15.552                 | 15.943                 | 72.10                  | 44.56                  | 46.04                  |
| **Translation**          | 1           | 2.499                  | 3.241                  | 3.411                  | 47.62                  | 33.32                  | 34.59                  |
| **Text Summarization**   | 1           | 2.811                  | 4.630                  | 3.369                  | 50.16                  | 37.36                  | 33.84                  |
| **Code Generation**      | 1           | 20.441                 | 27.685                 | 26.504                 | 83.12                  | 51.58                  | 52.26                  |
| **Chatbot**              | 1           | 5.035                  | 9.349                  | 8.366                  | 64.55                  | 43.96                  | 41.24                  |
| **Sentiment Analysis**   | 1           | 1.009                  | 1.235                  | 1.241                  | 5.95                   | 12.96                  | 12.89                  |
| **Multi-turn Reasoning** | 1           | 13.148                 | 20.184                 | 19.793                 | 76.44                  | 47.12                  | 47.29                  |

###### Concurrency = 2

| Scenario                 | VM 1 (1-nc48) Total TTFT (s) | VM 2 (2-nc24) Total TTFT (s) | VM 3 (1-nc24) Total TTFT (s) | VM 1 (1-nc48) Total tokens/s | VM 2 (2-nc24) Total tokens/s | VM 3 (1-nc24) Total tokens/s |
| ------------------------ | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| **Text Generation**      | 19.291                       | 19.978                       | 24.576                       | 110.94                       | 90.13                        | 79.26                        |
| **Question Answering**   | 14.165                       | 15.906                       | 21.774                       | 109.94                       | 90.87                        | 66.67                        |
| **Translation**          | 3.341                        | 4.513                        | 10.924                       | 76.45                        | 53.95                        | 68.54                        |
| **Text Summarization**   | 3.494                        | 3.664                        | 6.317                        | 77.38                        | 69.60                        | 59.45                        |
| **Code Generation**      | 16.693                       | 26.310                       | 27.772                       | 162.72                       | 104.37                       | 53.22                        |
| **Chatbot**              | 8.688                        | 9.537                        | 12.064                       | 100.09                       | 87.67                        | 67.23                        |
| **Sentiment Analysis**   | 1.251                        | 1.157                        | 1.229                        | 19.99                        | 20.09                        | 16.60                        |
| **Multi-turn Reasoning** | 20.233                       | 23.655                       | 22.880                       | 110.84                       | 94.47                        | 88.79                        |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/phi4-test-results.md*

**microsoft/Phi-3-small-8k-instruct**

| Scenario                             | Concurrency | VM 1 (1-nc48) TTFT (s) | VM 2 (2-nc24) TTFT (s) | VM 3 (1-nc24) TTFT (s) | VM 1 (1-nc48) tokens/s | VM 2 (2-nc24) tokens/s | VM 3 (1-nc24) tokens/s |
| ------------------------------------ | ----------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| Text Generation                      | 1           | 9.530                  | 9.070                  | 9.727                  | 68.41                  | 69.79                  | 66.31                  |
| Text Generation                      | 2           | 12.526                 | 13.902                 | 15.290                 | 105.02                 | 101.46                 | 92.11                  |
| Question Answering                   | 1           | 6.460                  | 7.401                  | 6.041                  | 65.64                  | 68.50                  | 65.22                  |
| Question Answering                   | 2           | 8.282                  | 6.851                  | 10.502                 | 89.15                  | 135.39                 | 103.23                 |
| Translation                          | 1           | 6.983                  | 8.552                  | 5.640                  | 67.02                  | 69.57                  | 66.13                  |
| Translation                          | 2           | 3.416                  | 5.951                  | 7.472                  | 73.14                  | 117.58                 | 82.20                  |
| Text Summarization                   | 1           | 2.570                  | 2.690                  | 2.004                  | 44.36                  | 55.39                  | 42.42                  |
| Text Summarization                   | 2           | 3.567                  | 3.197                  | 3.705                  | 75.13                  | 77.44                  | 81.46                  |
| Code Generation                      | 1           | 5.757                  | 1.991                  | 13.481                 | 74.69                  | 42.19                  | 83.15                  |
| Code Generation                      | 2           | 11.920                 | 14.886                 | 23.472                 | 91.85                  | 162.29                 | 115.73                 |
| Chatbot                              | 1           | 3.691                  | 3.160                  | 4.172                  | 54.46                  | 60.13                  | 62.80                  |
| Chatbot                              | 2           | 6.593                  | 3.633                  | 6.296                  | 92.07                  | 116.56                 | 100.43                 |
| Sentiment Analysis / Classification  | 1           | 0.957                  | 0.792                  | 0.783                  | 5.22                   | 6.31                   | 6.38                   |
| Sentiment Analysis / Classification  | 2           | 1.189                  | 1.015                  | 2.102                  | 8.44                   | 9.90                   | 52.12                  |
| Multi-turn Reasoning / Complex Tasks | 1           | 16.343                 | 26.220                 | 11.602                 | 72.45                  | 73.91                  | 72.23                  |
| Multi-turn Reasoning / Complex Tasks | 2           | 16.808                 | 12.774                 | 18.725                 | 149.10                 | 145.65                 | 136.84                 |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/Phi-3-small-8k-instruct-test-results.md*



#### Phi-3.5-vision-instruct Series test

```
#python press-phi3v-20250315.py
```

Test result analyze：

| **Scenario**         | **Concurrency** | **VM 1 (1-NC48) TTFT (s)** | **VM 2 (2-NC24) TTFT (s)** | **VM 3 (1-NC24) TTFT (s)** | **VM 1 (1-NC48) tokens/s per req** | **VM 2 (2-NC24) tokens/s per req** | **VM 3 (1-NC24) tokens/s per req** | **VM 1 (1-NC48) Overall Throughput** | **VM 2 (2-NC24) Overall Throughput** | **VM 3 (1-NC24) Overall Throughput** |
| -------------------- | --------------- | -------------------------- | -------------------------- | -------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| Single Request       | 1               | 5.687                      | 3.963                      | 4.029                      | 40.62                              | 34.82                              | 37.23                              | 40.62                                | 34.82                                | 37.23                                |
| Low Concurrency      | 2               | 6.791                      | 5.303                      | 3.894                      | 30.89                              | 25.57                              | 35.02                              | 61.78                                | 51.13                                | 70.05                                |
| Moderate Concurrency | 3               | 5.873                      | 5.257                      | 5.409                      | 24.58                              | 31.57                              | 24.58                              | 73.74                                | 94.71                                | 73.74                                |
| Higher Concurrency   | 4               | 5.453                      | 5.553                      | 5.823                      | 25.99                              | 27.50                              | 30.01                              | 77.96                                | 110.02                               | 150.03                               |
| Peak Concurrency     | 5               | 5.896                      | 6.466                      | 5.823                      | 28.77                              | 29.06                              | 30.01                              | 86.31                                | 145.30*                              | 150.03                               |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/phi3-v-results.md*



**financial-reports-analysis Series test**

```
#python press-phi3v-20250315.py
```

Test result analyze：

```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-1-nc48.txt |grep -A 7 "Summary for concurrency"
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.347 s
    Average throughput per req   : 74.76 tokens/s
    Overall throughput (sum)     : 74.76 tokens/s
    Batch duration (wall-clock)  : 8.352 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 16.248 s
    Average throughput per req   : 63.78 tokens/s
    Overall throughput (sum)     : 127.56 tokens/s
    Batch duration (wall-clock)  : 21.386 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 13.939 s
    Average throughput per req   : 65.47 tokens/s
    Overall throughput (sum)     : 130.95 tokens/s
    Batch duration (wall-clock)  : 18.746 s

--
  Summary for concurrency 4:
    Successful requests          : 2
    Failed requests              : 2
    Average TTFT per request     : 17.377 s
    Average throughput per req   : 60.21 tokens/s
    Overall throughput (sum)     : 120.42 tokens/s
    Batch duration (wall-clock)  : 22.402 s

--
  Summary for concurrency 5:
    Successful requests          : 2
    Failed requests              : 3
    Average TTFT per request     : 14.266 s
    Average throughput per req   : 65.39 tokens/s
    Overall throughput (sum)     : 130.77 tokens/s
    Batch duration (wall-clock)  : 18.840 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.835 s
    Average throughput per req   : 79.23 tokens/s
    Overall throughput (sum)     : 79.23 tokens/s
    Batch duration (wall-clock)  : 8.839 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 14.554 s
    Average throughput per req   : 62.45 tokens/s
    Overall throughput (sum)     : 124.91 tokens/s
    Batch duration (wall-clock)  : 19.864 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 15.182 s
    Average throughput per req   : 60.29 tokens/s
    Overall throughput (sum)     : 120.58 tokens/s
    Batch duration (wall-clock)  : 19.113 s

--
  Summary for concurrency 4:
    Successful requests          : 2
    Failed requests              : 2
    Average TTFT per request     : 17.206 s
    Average throughput per req   : 62.18 tokens/s
    Overall throughput (sum)     : 124.37 tokens/s
    Batch duration (wall-clock)  : 20.955 s

--
  Summary for concurrency 5:
    Successful requests          : 2
    Failed requests              : 3
    Average TTFT per request     : 15.526 s
    Average throughput per req   : 61.92 tokens/s
    Overall throughput (sum)     : 123.84 tokens/s
    Batch duration (wall-clock)  : 19.806 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 13.329 s
    Average throughput per req   : 86.73 tokens/s
    Overall throughput (sum)     : 86.73 tokens/s
    Batch duration (wall-clock)  : 13.334 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 14.185 s
    Average throughput per req   : 63.47 tokens/s
    Overall throughput (sum)     : 126.93 tokens/s
    Batch duration (wall-clock)  : 19.196 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 15.376 s
    Average throughput per req   : 61.93 tokens/s
    Overall throughput (sum)     : 123.86 tokens/s
    Batch duration (wall-clock)  : 20.004 s

--
  Summary for concurrency 4:
    Successful requests          : 2
    Failed requests              : 2
    Average TTFT per request     : 15.405 s
    Average throughput per req   : 64.14 tokens/s
    Overall throughput (sum)     : 128.29 tokens/s
    Batch duration (wall-clock)  : 20.872 s

--
  Summary for concurrency 5:
    Successful requests          : 2
    Failed requests              : 3
    Average TTFT per request     : 14.909 s
    Average throughput per req   : 63.94 tokens/s
    Overall throughput (sum)     : 127.89 tokens/s
    Batch duration (wall-clock)  : 19.572 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 8.002 s
    Average throughput per req   : 81.48 tokens/s
    Overall throughput (sum)     : 81.48 tokens/s
    Batch duration (wall-clock)  : 8.006 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 16.834 s
    Average throughput per req   : 64.28 tokens/s
    Overall throughput (sum)     : 128.56 tokens/s
    Batch duration (wall-clock)  : 21.731 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 11.225 s
    Average throughput per req   : 60.16 tokens/s
    Overall throughput (sum)     : 120.33 tokens/s
    Batch duration (wall-clock)  : 14.274 s

--
  Summary for concurrency 4:
    Successful requests          : 2
    Failed requests              : 2
    Average TTFT per request     : 13.520 s
    Average throughput per req   : 64.58 tokens/s
    Overall throughput (sum)     : 129.16 tokens/s
    Batch duration (wall-clock)  : 17.599 s

--
  Summary for concurrency 5:
    Successful requests          : 2
    Failed requests              : 3
    Average TTFT per request     : 13.541 s
    Average throughput per req   : 59.00 tokens/s
    Overall throughput (sum)     : 118.00 tokens/s
    Batch duration (wall-clock)  : 16.613 s

```



```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-2-nc24.txt |grep -A 7 "Summary for concurrency"
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 9.659 s
    Average throughput per req   : 62.63 tokens/s
    Overall throughput (sum)     : 62.63 tokens/s
    Batch duration (wall-clock)  : 9.664 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 11.663 s
    Average throughput per req   : 65.23 tokens/s
    Overall throughput (sum)     : 130.46 tokens/s
    Batch duration (wall-clock)  : 13.617 s

--
  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 20.658 s
    Average throughput per req   : 55.25 tokens/s
    Overall throughput (sum)     : 165.74 tokens/s
    Batch duration (wall-clock)  : 28.926 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 16.593 s
    Average throughput per req   : 53.76 tokens/s
    Overall throughput (sum)     : 53.76 tokens/s
    Batch duration (wall-clock)  : 16.597 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 20.202 s
    Average throughput per req   : 50.54 tokens/s
    Overall throughput (sum)     : 101.09 tokens/s
    Batch duration (wall-clock)  : 26.650 s

--
  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 19.131 s
    Average throughput per req   : 58.53 tokens/s
    Overall throughput (sum)     : 175.59 tokens/s
    Batch duration (wall-clock)  : 29.766 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 12.825 s
    Average throughput per req   : 66.27 tokens/s
    Overall throughput (sum)     : 66.27 tokens/s
    Batch duration (wall-clock)  : 12.829 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 12.664 s
    Average throughput per req   : 67.27 tokens/s
    Overall throughput (sum)     : 134.54 tokens/s
    Batch duration (wall-clock)  : 13.328 s

--
  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 17.639 s
    Average throughput per req   : 59.10 tokens/s
    Overall throughput (sum)     : 177.30 tokens/s
    Batch duration (wall-clock)  : 25.248 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 10.546 s
    Average throughput per req   : 68.65 tokens/s
    Overall throughput (sum)     : 68.65 tokens/s
    Batch duration (wall-clock)  : 10.550 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 16.594 s
    Average throughput per req   : 48.65 tokens/s
    Overall throughput (sum)     : 97.31 tokens/s
    Batch duration (wall-clock)  : 20.664 s

--
  Summary for concurrency 3:
    Successful requests          : 3
    Failed requests              : 0
    Average TTFT per request     : 16.779 s
    Average throughput per req   : 56.99 tokens/s
    Overall throughput (sum)     : 170.98 tokens/s
    Batch duration (wall-clock)  : 23.796 s

```



```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-1-nc24.txt |grep -A 7 "Summary for concurrency"
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 13.339 s
    Average throughput per req   : 71.15 tokens/s
    Overall throughput (sum)     : 71.15 tokens/s
    Batch duration (wall-clock)  : 13.344 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 21.675 s
    Average throughput per req   : 49.30 tokens/s
    Overall throughput (sum)     : 98.61 tokens/s
    Batch duration (wall-clock)  : 27.741 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 19.226 s
    Average throughput per req   : 52.44 tokens/s
    Overall throughput (sum)     : 104.88 tokens/s
    Batch duration (wall-clock)  : 26.149 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 14.241 s
    Average throughput per req   : 69.38 tokens/s
    Overall throughput (sum)     : 69.38 tokens/s
    Batch duration (wall-clock)  : 14.245 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 17.212 s
    Average throughput per req   : 51.91 tokens/s
    Overall throughput (sum)     : 103.82 tokens/s
    Batch duration (wall-clock)  : 23.023 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 19.061 s
    Average throughput per req   : 52.79 tokens/s
    Overall throughput (sum)     : 105.58 tokens/s
    Batch duration (wall-clock)  : 25.372 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 10.762 s
    Average throughput per req   : 65.88 tokens/s
    Overall throughput (sum)     : 65.88 tokens/s
    Batch duration (wall-clock)  : 10.765 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 20.992 s
    Average throughput per req   : 52.80 tokens/s
    Overall throughput (sum)     : 105.59 tokens/s
    Batch duration (wall-clock)  : 28.139 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 19.811 s
    Average throughput per req   : 47.85 tokens/s
    Overall throughput (sum)     : 95.71 tokens/s
    Batch duration (wall-clock)  : 24.749 s

--
  Summary for concurrency 1:
    Successful requests          : 1
    Failed requests              : 0
    Average TTFT per request     : 10.182 s
    Average throughput per req   : 66.19 tokens/s
    Overall throughput (sum)     : 66.19 tokens/s
    Batch duration (wall-clock)  : 10.187 s

--
  Summary for concurrency 2:
    Successful requests          : 2
    Failed requests              : 0
    Average TTFT per request     : 18.303 s
    Average throughput per req   : 52.05 tokens/s
    Overall throughput (sum)     : 104.10 tokens/s
    Batch duration (wall-clock)  : 24.445 s

--
  Summary for concurrency 3:
    Successful requests          : 2
    Failed requests              : 1
    Average TTFT per request     : 11.118 s
    Average throughput per req   : 48.83 tokens/s
    Overall throughput (sum)     : 97.65 tokens/s
    Batch duration (wall-clock)  : 14.555 s
```





































































































======================        Under developing part              =======================



## Performance test on Azure AI model inference

Azure AI model inference has a default quota. If you feel that the quota for the model is insufficient, you can apply for an increase separately. 

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/14.png)

***https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/quotas-limits#request-increases-to-the-default-limits***

| Limit name              | Applies to          | Limit value                                                  |
| ----------------------- | ------------------- | ------------------------------------------------------------ |
| Tokens per minute       | Azure OpenAI models | Varies per model and SKU. See [limits for Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits). |
| Requests per minute     | Azure OpenAI models | Varies per model and SKU. See [limits for Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits). |
| **Tokens per minute**   | **DeepSeek models** | **5.000.000**                                                |
| **Requests per minute** | **DeepSeek models** | **5.000**                                                    |
| **Concurrent requests** | **DeepSeek models** | **300**                                                      |
| Tokens per minute       | Rest of models      | 200.000                                                      |
| Requests per minute     | Rest of models      | 1.000                                                        |
| Concurrent requests     | Rest of models      | 300                                                          |

After you have deployed models on Azure AI model inference, you can check their invocation methods：

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/11.png)

Prepare test env:

```
#conda create -n AImodelinference python=3.11 -y
#conda activate AImodelinference
#pip install azure-ai-inference
```

Run test script, after entering the following three variables, the stress test will begin:

```
#python callaiinference.py
Please enter the Azure AI key: 
Please enter the Azure AI endpoint URL:
Please enter the deployment name:
```



### Performance on DS 671B

I will use the test results of DeeSeek R1 on Azure AI model inference  as an example:

  **Max performance:**

• When the concurrency is 300 and the prompt length is 1024, TPS = 2110.77, TTFT = 2.201s.
 • When the concurrency is 300 and the prompt length is 2048, TPS = 1330.94, TTFT = 1.861s.

**Overall performance:** 

The overall throughput averages 735.12 tokens/s, with a P90 of 1184.06 tokens/s, full test result is as following:

| **Concurrency** | **Prompt Length** | **Total Requests** | **Success Count** | **Fail Count** | **Average latency (s)** | **Average TTFT (s)** | **Average token throughput (tokens/s)** | **Overall throughput (tokens/s)** |
| --------------- | ----------------- | ------------------ | ----------------- | -------------- | ----------------------- | -------------------- | --------------------------------------- | --------------------------------- |
| 300             | 1024              | 110                | 110               | 0              | 75.579                  | 2.580                | 22.54                                   | 806.84                            |
| 300             | 1024              | 110                | 110               | 0              | 71.378                  | 71.378               | 24.53                                   | 1028.82                           |
| 300             | 1024              | 110                | 110               | 0              | 76.622                  | 2.507                | 23.24                                   | 979.97                            |
| 300             | 1024              | 120                | 120               | 0              | 68.750                  | 68.750               | 24.91                                   | 540.66                            |
| 300             | 1024              | 120                | 120               | 0              | 72.164                  | 2.389                | 22.71                                   | 1094.90                           |
| 300             | 1024              | 130                | 130               | 0              | 72.245                  | 72.245               | 23.68                                   | 1859.91                           |
| 300             | 1024              | 130                | 130               | 0              | 82.714                  | 2.003                | 20.18                                   | 552.08                            |
| 300             | 1024              | 140                | 140               | 0              | 71.458                  | 71.458               | 23.79                                   | 642.92                            |
| 300             | 1024              | 140                | 140               | 0              | 71.565                  | 2.400                | 22.93                                   | 488.49                            |
| 300             | 1024              | 150                | 150               | 0              | 71.958                  | 71.958               | 24.21                                   | 1269.10                           |
| 300             | 1024              | 150                | 150               | 0              | 73.712                  | 2.201                | 22.35                                   | 2110.77                           |
| 300             | 2048              | 10                 | 10                | 0              | 68.811                  | 68.811               | 24.24                                   | 196.78                            |
| 300             | 2048              | 10                 | 10                | 0              | 70.189                  | 1.021                | 23.18                                   | 172.92                            |
| 300             | 2048              | 20                 | 20                | 0              | 73.138                  | 73.138               | 24.14                                   | 390.96                            |
| 300             | 2048              | 20                 | 20                | 0              | 69.649                  | 1.150                | 24.22                                   | 351.31                            |
| 300             | 2048              | 30                 | 30                | 0              | 66.883                  | 66.883               | 26.13                                   | 556.12                            |
| 300             | 2048              | 30                 | 30                | 0              | 68.918                  | 1.660                | 23.46                                   | 571.63                            |
| 300             | 2048              | 40                 | 40                | 0              | 72.485                  | 72.485               | 23.85                                   | 716.53                            |
| 300             | 2048              | 40                 | 40                | 0              | 65.228                  | 1.484                | 24.87                                   | 625.16                            |
| 300             | 2048              | 50                 | 50                | 0              | 68.223                  | 68.223               | 25.12                                   | 887.64                            |
| 300             | 2048              | 50                 | 50                | 0              | 66.288                  | 1.815                | 24.38                                   | 976.17                            |
| 300             | 2048              | 60                 | 60                | 0              | 66.736                  | 66.736               | 25.85                                   | 547.70                            |
| 300             | 2048              | 60                 | 60                | 0              | 69.355                  | 2.261                | 23.94                                   | 615.81                            |
| 300             | 2048              | 70                 | 70                | 0              | 66.689                  | 66.689               | 25.66                                   | 329.90                            |
| 300             | 2048              | 70                 | 70                | 0              | 67.061                  | 2.128                | 23.89                                   | 1373.11                           |
| 300             | 2048              | 80                 | 80                | 0              | 68.091                  | 68.091               | 25.68                                   | 1516.27                           |
| 300             | 2048              | 80                 | 80                | 0              | 67.413                  | 1.861                | 24.01                                   | 1330.94                           |
| 300             | 2048              | 90                 | 90                | 0              | 66.603                  | 66.603               | 25.51                                   | 418.81                            |
| 300             | 2048              | 90                 | 90                | 0              | 70.072                  | 2.346                | 23.41                                   | 1047.53                           |
| 300             | 2048              | 100                | 100               | 0              | 70.516                  | 70.516               | 24.29                                   | 456.66                            |
| 300             | 2048              | 100                | 100               | 0              | 86.862                  | 2.802                | 20.03                                   | 899.38                            |
| 300             | 2048              | 110                | 110               | 0              | 84.602                  | 84.602               | 21.16                                   | 905.59                            |
| 300             | 2048              | 110                | 110               | 0              | 77.883                  | 2.179                | 21.17                                   | 803.93                            |
| 300             | 2048              | 120                | 120               | 0              | 73.814                  | 73.814               | 23.73                                   | 541.03                            |
| 300             | 2048              | 120                | 120               | 0              | 86.787                  | 4.413                | 20.32                                   | 650.57                            |
| 300             | 2048              | 130                | 130               | 0              | 78.222                  | 78.222               | 22.61                                   | 613.27                            |
| 300             | 2048              | 130                | 130               | 0              | 83.670                  | 2.131                | 20.16                                   | 1463.81                           |
| 300             | 2048              | 140                | 140               | 0              | 77.429                  | 77.429               | 22.74                                   | 1184.06                           |
| 300             | 2048              | 140                | 140               | 0              | 77.234                  | 3.891                | 21.90                                   | 821.34                            |
| 300             | 2048              | 150                | 150               | 0              | 72.753                  | 72.753               | 23.69                                   | 698.50                            |
| 300             | 2048              | 150                | 150               | 0              | 73.674                  | 2.425                | 22.74                                   | 1012.25                           |
| 300             | 4096              | 10                 | 10                | 0              | 83.003                  | 83.003               | 25.52                                   | 221.28                            |
| 300             | 4096              | 10                 | 10                | 0              | 89.713                  | 1.084                | 24.70                                   | 189.29                            |
| 300             | 4096              | 20                 | 20                | 0              | 82.342                  | 82.342               | 26.65                                   | 337.85                            |
| 300             | 4096              | 20                 | 20                | 0              | 84.526                  | 1.450                | 24.81                                   | 376.17                            |
| 300             | 4096              | 30                 | 30                | 0              | 87.979                  | 87.979               | 24.46                                   | 322.62                            |
| 300             | 4096              | 30                 | 30                | 0              | 84.767                  | 1.595                | 24.28                                   | 503.01                            |
| 300             | 4096              | 40                 | 40                | 0              | 85.231                  | 85.231               | 26.03                                   | 733.50                            |
| 300             | 4096              | 40                 | 40                | 0              | 81.514                  | 1.740                | 24.17                                   | 710.79                            |
| 300             | 4096              | 50                 | 50                | 0              | 91.253                  | 91.253               | 24.53                                   | 279.55                            |



### Performance Phi-4

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/12.png)

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/13.png)

```
(AIF) root@pythonvm:~/AIFperformance# python callapinormal.py 
Please enter the Azure AI key: G485wnXwMrAYQKMQPSYpzf7PNLm3sui8qgsXcYFv5Yd3HOmvzZ2GJQQJ99BCACPV0roXJ3w3AAAAACOG9kt1
Please enter the Azure AI endpoint URL: https://xinyu-m7zxv3ow-germanywestcentra.services.ai.azure.com/models
Please enter the deployment name: Phi-4

>>> Testing Concurrency: 200, Prompt Length: 128, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 23.822
  Average TTFT (s): 23.822
  Average token throughput (tokens/s): 27.91
  Overall throughput (tokens/s): 249.26

=== Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 23.547
  Average TTFT (s): 0.419
  Average token throughput (tokens/s): 27.00
  Overall throughput (tokens/s): 222.52


>>> Testing Concurrency: 200, Prompt Length: 128, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 22.429
  Average TTFT (s): 22.429
  Average token throughput (tokens/s): 30.64
  Overall throughput (tokens/s): 463.46

Unable to stream download: Response ended prematurely
Attempt 1 failed: Response ended prematurely
=== Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 21.512
  Average TTFT (s): 0.970
  Average token throughput (tokens/s): 29.54
  Overall throughput (tokens/s): 417.09


>>> Testing Concurrency: 200, Prompt Length: 128, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 23.560
  Average TTFT (s): 23.560
  Average token throughput (tokens/s): 28.28
  Overall throughput (tokens/s): 698.09

=== Stream Mode | Concurrency: 200, Prompt length: 128, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 22.677
  Average TTFT (s): 1.143
  Average token throughput (tokens/s): 27.74
  Overall throughput (tokens/s): 628.50
```

**Max performance:**

• When the concurrency is 300 and the prompt length is 1024, TPS = 1473.44, TTFT = 30.861s (Non-Stream Mode).
• When the concurrency is 300 and the prompt length is 2048, TPS = 849.75, TTFT = 50.730s (Non-Stream Mode).

**Overall performance:**

The overall throughput averages 735.12 tokens/s, with a P90 of 1184.06 tokens/s. Full test results are as follows:

```
>>> Testing Concurrency: 300, Prompt Length: 128, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 42.786
  Average TTFT (s): 42.786
  Average token throughput (tokens/s): 16.25
  Overall throughput (tokens/s): 259.47

=== Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 41.799
  Average TTFT (s): 0.971
  Average token throughput (tokens/s): 15.86
  Overall throughput (tokens/s): 215.46


>>> Testing Concurrency: 300, Prompt Length: 128, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 36.526
  Average TTFT (s): 36.526
  Average token throughput (tokens/s): 18.79
  Overall throughput (tokens/s): 464.05

=== Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 29.335
  Average TTFT (s): 1.016
  Average token throughput (tokens/s): 22.19
  Overall throughput (tokens/s): 404.16


>>> Testing Concurrency: 300, Prompt Length: 128, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 34.573
  Average TTFT (s): 34.573
  Average token throughput (tokens/s): 19.98
  Overall throughput (tokens/s): 635.16

=== Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 37.575
  Average TTFT (s): 1.096
  Average token throughput (tokens/s): 17.29
  Overall throughput (tokens/s): 609.03


>>> Testing Concurrency: 300, Prompt Length: 128, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 25.340
  Average TTFT (s): 25.340
  Average token throughput (tokens/s): 26.43
  Overall throughput (tokens/s): 1092.32

=== Stream Mode | Concurrency: 300, Prompt length: 128, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 54.118
  Average TTFT (s): 1.994
  Average token throughput (tokens/s): 11.59
  Overall throughput (tokens/s): 438.72


>>> Testing Concurrency: 300, Prompt Length: 256, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 31.659
  Average TTFT (s): 31.659
  Average token throughput (tokens/s): 26.99
  Overall throughput (tokens/s): 217.86

=== Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 48.118
  Average TTFT (s): 0.411
  Average token throughput (tokens/s): 18.50
  Overall throughput (tokens/s): 90.95


>>> Testing Concurrency: 300, Prompt Length: 256, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 23.250
  Average TTFT (s): 23.250
  Average token throughput (tokens/s): 34.82
  Overall throughput (tokens/s): 623.39

=== Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 48.669
  Average TTFT (s): 0.887
  Average token throughput (tokens/s): 15.52
  Overall throughput (tokens/s): 259.49


>>> Testing Concurrency: 300, Prompt Length: 256, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 41.130
  Average TTFT (s): 41.130
  Average token throughput (tokens/s): 20.32
  Overall throughput (tokens/s): 456.73

=== Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 57.212
  Average TTFT (s): 1.548
  Average token throughput (tokens/s): 13.65
  Overall throughput (tokens/s): 323.89


>>> Testing Concurrency: 300, Prompt Length: 256, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 57.891
  Average TTFT (s): 57.891
  Average token throughput (tokens/s): 14.17
  Overall throughput (tokens/s): 496.40

=== Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 52.031
  Average TTFT (s): 2.474
  Average token throughput (tokens/s): 14.83
  Overall throughput (tokens/s): 435.96


>>> Testing Concurrency: 300, Prompt Length: 256, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 45.228
  Average TTFT (s): 45.228
  Average token throughput (tokens/s): 17.69
  Overall throughput (tokens/s): 725.04

=== Stream Mode | Concurrency: 300, Prompt length: 256, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 43.595
  Average TTFT (s): 1.257
  Average token throughput (tokens/s): 16.95
  Overall throughput (tokens/s): 712.82


>>> Testing Concurrency: 300, Prompt Length: 512, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 32.092
  Average TTFT (s): 32.092
  Average token throughput (tokens/s): 26.78
  Overall throughput (tokens/s): 242.20

=== Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 25.930
  Average TTFT (s): 0.568
  Average token throughput (tokens/s): 31.35
  Overall throughput (tokens/s): 245.37


>>> Testing Concurrency: 300, Prompt Length: 512, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 34.330
  Average TTFT (s): 34.330
  Average token throughput (tokens/s): 26.04
  Overall throughput (tokens/s): 444.89

=== Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 34.694
  Average TTFT (s): 1.629
  Average token throughput (tokens/s): 23.48
  Overall throughput (tokens/s): 408.55


>>> Testing Concurrency: 300, Prompt Length: 512, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 34.773
  Average TTFT (s): 34.773
  Average token throughput (tokens/s): 25.91
  Overall throughput (tokens/s): 632.48

=== Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 31.973
  Average TTFT (s): 0.970
  Average token throughput (tokens/s): 25.72
  Overall throughput (tokens/s): 632.10


>>> Testing Concurrency: 300, Prompt Length: 512, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 36.616
  Average TTFT (s): 36.616
  Average token throughput (tokens/s): 24.19
  Overall throughput (tokens/s): 851.76

=== Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 34.922
  Average TTFT (s): 1.091
  Average token throughput (tokens/s): 23.83
  Overall throughput (tokens/s): 783.17


>>> Testing Concurrency: 300, Prompt Length: 512, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 36.638
  Average TTFT (s): 36.638
  Average token throughput (tokens/s): 24.40
  Overall throughput (tokens/s): 1003.91

=== Stream Mode | Concurrency: 300, Prompt length: 512, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 34.217
  Average TTFT (s): 1.433
  Average token throughput (tokens/s): 23.82
  Overall throughput (tokens/s): 940.82


>>> Testing Concurrency: 300, Prompt Length: 1024, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 28.029
  Average TTFT (s): 28.029
  Average token throughput (tokens/s): 36.46
  Overall throughput (tokens/s): 305.37

=== Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 30.585
  Average TTFT (s): 0.428
  Average token throughput (tokens/s): 31.08
  Overall throughput (tokens/s): 246.82


>>> Testing Concurrency: 300, Prompt Length: 1024, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 31.945
  Average TTFT (s): 31.945
  Average token throughput (tokens/s): 32.23
  Overall throughput (tokens/s): 559.50

=== Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 24.585
  Average TTFT (s): 0.949
  Average token throughput (tokens/s): 37.25
  Overall throughput (tokens/s): 595.32


>>> Testing Concurrency: 300, Prompt Length: 1024, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 30.950
  Average TTFT (s): 30.950
  Average token throughput (tokens/s): 33.02
  Overall throughput (tokens/s): 852.51

=== Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 25.622
  Average TTFT (s): 1.014
  Average token throughput (tokens/s): 36.02
  Overall throughput (tokens/s): 951.37


>>> Testing Concurrency: 300, Prompt Length: 1024, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 31.642
  Average TTFT (s): 31.642
  Average token throughput (tokens/s): 32.85
  Overall throughput (tokens/s): 1198.05

=== Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 28.190
  Average TTFT (s): 1.099
  Average token throughput (tokens/s): 33.01
  Overall throughput (tokens/s): 1099.36


>>> Testing Concurrency: 300, Prompt Length: 1024, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 30.861
  Average TTFT (s): 30.861
  Average token throughput (tokens/s): 32.97
  Overall throughput (tokens/s): 1473.44

=== Stream Mode | Concurrency: 300, Prompt length: 1024, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 31.885
  Average TTFT (s): 1.121
  Average token throughput (tokens/s): 29.28
  Overall throughput (tokens/s): 1238.09


>>> Testing Concurrency: 300, Prompt Length: 2048, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 27.862
  Average TTFT (s): 27.862
  Average token throughput (tokens/s): 42.47
  Overall throughput (tokens/s): 348.38

=== Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 27.356
  Average TTFT (s): 0.439
  Average token throughput (tokens/s): 36.49
  Overall throughput (tokens/s): 329.59


>>> Testing Concurrency: 300, Prompt Length: 2048, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 29.009
  Average TTFT (s): 29.009
  Average token throughput (tokens/s): 39.40
  Overall throughput (tokens/s): 690.07

=== Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 30.951
  Average TTFT (s): 0.935
  Average token throughput (tokens/s): 33.85
  Overall throughput (tokens/s): 527.14


>>> Testing Concurrency: 300, Prompt Length: 2048, Total Requests: 30 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 54.856
  Average TTFT (s): 54.856
  Average token throughput (tokens/s): 21.02
  Overall throughput (tokens/s): 505.79

=== Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 52.796
  Average TTFT (s): 1.383
  Average token throughput (tokens/s): 20.89
  Overall throughput (tokens/s): 451.47


>>> Testing Concurrency: 300, Prompt Length: 2048, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 49.235
  Average TTFT (s): 49.235
  Average token throughput (tokens/s): 23.15
  Overall throughput (tokens/s): 836.75

=== Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 45.512
  Average TTFT (s): 1.923
  Average token throughput (tokens/s): 23.20
  Overall throughput (tokens/s): 752.37


>>> Testing Concurrency: 300, Prompt Length: 2048, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 50.730
  Average TTFT (s): 50.730
  Average token throughput (tokens/s): 22.45
  Overall throughput (tokens/s): 849.75

=== Stream Mode | Concurrency: 300, Prompt length: 2048, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 52.496
  Average TTFT (s): 1.712
  Average token throughput (tokens/s): 19.05
  Overall throughput (tokens/s): 736.86


>>> Testing Concurrency: 300, Prompt Length: 4096, Total Requests: 10 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 38.206
  Average TTFT (s): 38.206
  Average token throughput (tokens/s): 28.21
  Overall throughput (tokens/s): 225.61

=== Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 10 ===
  Success count: 10, Fail count: 0
  Average latency (s): 34.517
  Average TTFT (s): 0.470
  Average token throughput (tokens/s): 28.74
  Overall throughput (tokens/s): 250.12


>>> Testing Concurrency: 300, Prompt Length: 4096, Total Requests: 20 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 32.739
  Average TTFT (s): 32.739
  Average token throughput (tokens/s): 33.77
  Overall throughput (tokens/s): 556.52

=== Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 20 ===
  Success count: 20, Fail count: 0
  Average latency (s): 32.623
  Average TTFT (s): 1.371
  Average token throughput (tokens/s): 28.39
  Overall throughput (tokens/s): 484.34


>>> Testing Concurrency: 300, Prompt Length: 4096, Total Requests: 30 <<<
Attempt 1 failed: (Timeout) The operation was timeout.
Code: Timeout
Message: The operation was timeout.
=== Non-Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 42.533
  Average TTFT (s): 42.533
  Average token throughput (tokens/s): 26.16
  Overall throughput (tokens/s): 214.83

=== Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 30 ===
  Success count: 30, Fail count: 0
  Average latency (s): 33.837
  Average TTFT (s): 1.250
  Average token throughput (tokens/s): 29.54
  Overall throughput (tokens/s): 609.49


>>> Testing Concurrency: 300, Prompt Length: 4096, Total Requests: 40 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 34.546
  Average TTFT (s): 34.546
  Average token throughput (tokens/s): 33.39
  Overall throughput (tokens/s): 1122.95

=== Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 40 ===
  Success count: 40, Fail count: 0
  Average latency (s): 45.994
  Average TTFT (s): 1.145
  Average token throughput (tokens/s): 22.39
  Overall throughput (tokens/s): 687.95


>>> Testing Concurrency: 300, Prompt Length: 4096, Total Requests: 50 <<<
=== Non-Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 48.059
  Average TTFT (s): 48.059
  Average token throughput (tokens/s): 23.81
  Overall throughput (tokens/s): 733.79

=== Stream Mode | Concurrency: 300, Prompt length: 4096, Total requests: 50 ===
  Success count: 50, Fail count: 0
  Average latency (s): 45.475
  Average TTFT (s): 1.221
  Average token throughput (tokens/s): 21.21
  Overall throughput (tokens/s): 809.73
```



