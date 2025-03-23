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

| **Model Name on AML**                         | **Model on HF** (tokenizers name)             | **Azure GPU VM SKU Support in AML**              |
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



#### **Clone code and prepare shell environment**

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

#### **Deploy model Automatically**

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

The primary goal of performance testing is to verify tokens/s and TTFT during the inference process. To better simulate real-world scenarios, I have set up several common LLM/SLM use cases in the test script. Additionally, to ensure tokens/s performance, the test script needs to load the corresponding model's tokenizer during execution(Refer to upper table of tokenizers name).

Before officially starting the test, you need to log in to HF on your terminal.

```
huggingface-cli  login
```

#### Phi Text2Text Series (Phi-4/Phi-3-small-8k-instruct)

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

| Scenario                 | VM 1 (1-nc48) TTFT (s) | VM 2 (2-nc24) TTFT (s) | VM 3 (1-nc24) TTFT (s) | VM 1 (1-nc48) tokens/s | VM 2 (2-nc24) tokens/s | VM 3 (1-nc24) tokens/s |
| ------------------------ | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| **Text Generation**      | 12.473                 | 19.546                 | 19.497                 | 68.07                  | 44.66                  | 44.78                  |
| **Question Answering**   | 11.914                 | 15.552                 | 15.943                 | 72.10                  | 44.56                  | 46.04                  |
| **Translation**          | 2.499                  | 3.241                  | 3.411                  | 47.62                  | 33.32                  | 34.59                  |
| **Text Summarization**   | 2.811                  | 4.630                  | 3.369                  | 50.16                  | 37.36                  | 33.84                  |
| **Code Generation**      | 20.441                 | 27.685                 | 26.504                 | 83.12                  | 51.58                  | 52.26                  |
| **Chatbot**              | 5.035                  | 9.349                  | 8.366                  | 64.55                  | 43.96                  | 41.24                  |
| **Sentiment Analysis**   | 1.009                  | 1.235                  | 1.241                  | 5.95                   | 12.96                  | 12.89                  |
| **Multi-turn Reasoning** | 13.148                 | 20.184                 | 19.793                 | 76.44                  | 47.12                  | 47.29                  |

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

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/phi4-test-results.md*

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

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/Phi-3-small-8k-instruct-test-results.md*



#### Phi vision series (Phi-3.5-vision-instruct/Phi-3-vision-128k-instruct)

```
# python press-phi35and0v-20250323.py
```

**Phi-3.5-vision-instruct with single image input test result analyze:**

**on NC24 A100 VM:**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 2.117            | 57.17                                 | 57.17                       | 2.126              |
| 2           | 2                   | 0               | 4.348            | 18.85                                 | 37.71                       | 7.722              |
| 3           | 3                   | 0               | 3.389            | 49.50                                 | 148.50                      | 6.354              |
| **4**       | **4**               | **0**           | **2.898**        | **49.22**                             | **196.86**                  | **7.207**          |
| 5           | 4                   | 1               | 2.708            | 41.63                                 | 166.53                      | 8.942              |
| 6           | 5                   | 1               | 2.095            | 32.30                                 | 161.52                      | 8.951              |
| 7           | 5                   | 2               | 2.774            | 48.95                                 | 244.75                      | 8.966              |
| 8           | 4                   | 4               | 2.841            | 48.30                                 | 193.21                      | 8.953              |
| 9           | 4                   | 5               | 2.996            | 41.86                                 | 167.43                      | 8.960              |
| 10          | 4                   | 6               | 2.874            | 45.60                                 | 182.38                      | 8.958              |

**Phi-3-vision-128k-instruct with single image input test result analyze：**

**On NC48 VM：**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 2.124            | 46.13                                 | 46.13                       | 2.130              |
| 2           | 2                   | 0               | 2.828            | 44.21                                 | 88.41                       | 3.858              |
| 3           | 3                   | 0               | 3.432            | 47.35                                 | 142.04                      | 6.437              |
| **4**       | **4**               | **0**           | **2.497**        | **42.99**                             | **171.96**                  | **7.060**          |
| 5           | 4                   | 1               | 3.447            | 47.35                                 | 189.39                      | 8.948              |
| 6           | 5                   | 1               | 2.291            | 38.98                                 | 194.92                      | 8.964              |
| 7           | 4                   | 3               | 3.099            | 41.58                                 | 166.34                      | 8.956              |
| 8           | 4                   | 4               | 2.247            | 34.58                                 | 138.31                      | 8.960              |
| 9           | 5                   | 4               | 2.321            | 36.79                                 | 183.96                      | 8.952              |
| 10          | 5                   | 5               | 2.466            | 36.55                                 | 182.77                      | 8.950              |

#### **financial-reports-analysis Series test**

```
#python press-phi3v-20250315.py
```

**1-nc48**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 8.347            | 74.76                                 | 74.76                       | 8.352              |
| **2**       | **2**               | **0**           | **16.248**       | **63.78**                             | **127.56**                  | **21.386**         |
| 3           | 2                   | 1               | 13.939           | 65.47                                 | 130.95                      | 18.746             |
| 4           | 2                   | 2               | 17.377           | 60.21                                 | 120.42                      | 22.402             |
| 5           | 2                   | 3               | 14.266           | 65.39                                 | 130.77                      | 18.840             |
| 1           | 1                   | 0               | 8.835            | 79.23                                 | 79.23                       | 8.839              |
| 2           | 2                   | 0               | 14.554           | 62.45                                 | 124.91                      | 19.864             |
| 3           | 2                   | 1               | 15.182           | 60.29                                 | 120.58                      | 19.113             |
| 4           | 2                   | 2               | 17.206           | 62.18                                 | 124.37                      | 20.955             |
| 5           | 2                   | 3               | 15.526           | 61.92                                 | 123.84                      | 19.806             |
| 1           | 1                   | 0               | 13.329           | 86.73                                 | 86.73                       | 13.334             |
| 2           | 2                   | 0               | 14.185           | 63.47                                 | 126.93                      | 19.196             |
| 3           | 2                   | 1               | 15.376           | 61.93                                 | 123.86                      | 20.004             |
| 4           | 2                   | 2               | 15.405           | 64.14                                 | 128.29                      | 20.872             |
| 5           | 2                   | 3               | 14.909           | 63.94                                 | 127.89                      | 19.572             |
| 1           | 1                   | 0               | 8.002            | 81.48                                 | 81.48                       | 8.006              |
| 2           | 2                   | 0               | 16.834           | 64.28                                 | 128.56                      | 21.731             |
| 3           | 2                   | 1               | 11.225           | 60.16                                 | 120.33                      | 14.274             |
| 4           | 2                   | 2               | 13.520           | 64.58                                 | 129.16                      | 17.599             |
| 5           | 2                   | 3               | 13.541           | 59.00                                 | 118.00                      | 16.613             |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/output-financial-reports-analysis-1-nc48.txt*

```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-1-nc48.txt |grep -A 7 
```

**2-nc24**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 9.659            | 62.63                                 | 62.63                       | 9.664              |
| 2           | 2                   | 0               | 11.663           | 65.23                                 | 130.46                      | 13.617             |
| 3           | 3                   | 0               | 20.658           | 55.25                                 | 165.74                      | 28.926             |
| 1           | 1                   | 0               | 16.593           | 53.76                                 | 53.76                       | 16.597             |
| 2           | 2                   | 0               | 20.202           | 50.54                                 | 101.09                      | 26.650             |
| **3**       | **3**               | **0**           | **19.131**       | **58.53**                             | **175.59**                  | **29.766**         |
| 1           | 1                   | 0               | 12.825           | 66.27                                 | 66.27                       | 12.829             |
| 2           | 2                   | 0               | 12.664           | 67.27                                 | 134.54                      | 13.328             |
| 3           | 3                   | 0               | 17.639           | 59.10                                 | 177.30                      | 25.248             |
| 1           | 1                   | 0               | 10.546           | 68.65                                 | 68.65                       | 10.550             |
| 2           | 2                   | 0               | 16.594           | 48.65                                 | 97.31                       | 20.664             |
| 3           | 3                   | 0               | 16.779           | 56.99                                 | 170.98                      | 23.796             |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/output-financial-reports-analysis-2-nc24.txt*

```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-2-nc24.txt |grep -A 7 

```

**1-nc24**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 13.339           | 71.15                                 | 71.15                       | 13.344             |
| 2           | 2                   | 0               | 21.675           | 49.30                                 | 98.61                       | 27.741             |
| 3           | 2                   | 1               | 19.226           | 52.44                                 | 104.88                      | 26.149             |
| 1           | 1                   | 0               | 14.241           | 69.38                                 | 69.38                       | 14.245             |
| **2**       | **2**               | **0**           | **17.212**       | **51.91**                             | **103.82**                  | **23.023**         |
| 3           | 2                   | 1               | 19.061           | 52.79                                 | 105.58                      | 25.372             |
| 1           | 1                   | 0               | 10.762           | 65.88                                 | 65.88                       | 10.765             |
| 2           | 2                   | 0               | 20.992           | 52.80                                 | 105.59                      | 28.139             |
| 3           | 2                   | 1               | 19.811           | 47.85                                 | 95.71                       | 24.749             |
| 1           | 1                   | 0               | 10.182           | 66.19                                 | 66.19                       | 10.187             |
| 2           | 2                   | 0               | 18.303           | 52.05                                 | 104.10                      | 24.445             |
| 3           | 2                   | 1               | 11.118           | 48.83                                 | 97.65                       | 14.555             |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/output-financial-reports-analysis-1-nc24.txt*

```
(base) root@linuxworkvm:~/AIFperformance# cat output-financial-reports-analysis-1-nc24.txt |grep -A 7 "Summary for concurrency"
      
```



#### microsoft-swinv2-base-patch4-window12-192-22k Series 

```
#python press-swinv2-20250322.py
```

Test result analyze：

**1-NC48**

| **Concurrency** | **Successful Requests** | **Failed Requests** | **Average TTFT (s)** | **Avg Throughput per Request (tokens/s)** | **Total Throughput (tokens/s)** | **Batch Duration (s)** |
| --------------- | ----------------------- | ------------------- | -------------------- | ----------------------------------------- | ------------------------------- | ---------------------- |
| 1               | 1                       | 0                   | 0.910                | 27.46                                     | 27.46                           | 0.911                  |
| 2               | 2                       | 0                   | 1.055                | 24.12                                     | 48.25                           | 1.198                  |
| 3               | 3                       | 0                   | 1.073                | 23.80                                     | 71.41                           | 2.600                  |
| 4               | 4                       | 0                   | 1.198                | 21.98                                     | 87.93                           | 2.983                  |
| 5               | 5                       | 0                   | 1.031                | 24.69                                     | 123.45                          | 5.209                  |
| 6               | 6                       | 0                   | 1.309                | 20.39                                     | 122.32                          | 5.506                  |
| 7               | 6                       | 1                   | 1.059                | 24.04                                     | 144.25                          | 8.957                  |
| 8               | 6                       | 2                   | 1.110                | 23.16                                     | 138.99                          | 8.965                  |
| 9               | 6                       | 3                   | 1.084                | 23.59                                     | 141.56                          | 8.956                  |
| 10              | 6                       | 4                   | 1.108                | 23.07                                     | 138.40                          | 8.963                  |



**2-NC24**

| **Concurrency** | **Successful Requests** | **Failed Requests** | **Average TTFT (s)** | **Avg Throughput per Request (tokens/s)** | **Total Throughput (tokens/s)** | **Batch Duration (s)** |
| --------------- | ----------------------- | ------------------- | -------------------- | ----------------------------------------- | ------------------------------- | ---------------------- |
| 1               | 1                       | 0                   | 1.002                | 24.94                                     | 24.94                           | 1.004                  |
| 2               | 2                       | 0                   | 1.272                | 19.91                                     | 39.83                           | 1.421                  |
| 3               | 3                       | 0                   | 1.093                | 23.22                                     | 69.65                           | 1.292                  |
| 4               | 4                       | 0                   | 1.151                | 22.22                                     | 88.86                           | 1.357                  |
| 5               | 5                       | 0                   | 1.042                | 24.43                                     | 122.16                          | 2.582                  |
| 6               | 6                       | 0                   | 1.047                | 24.33                                     | 145.98                          | 2.610                  |
| 7               | 7                       | 0                   | 1.067                | 23.90                                     | 167.27                          | 2.859                  |
| 8               | 8                       | 0                   | 1.227                | 21.08                                     | 168.63                          | 2.881                  |
| 9               | 9                       | 0                   | 1.074                | 23.82                                     | 214.39                          | 5.212                  |
| 10              | 10                      | 0                   | 1.234                | 21.25                                     | 212.51                          | 5.506                  |

**1-NC24**

| Concurrency | Successful Requests | Failed Requests | Average TTFT (s) | Avg Throughput per Request (tokens/s) | Total Throughput (tokens/s) | Batch Duration (s) |
| ----------- | ------------------- | --------------- | ---------------- | ------------------------------------- | --------------------------- | ------------------ |
| 1           | 1                   | 0               | 1.015            | 24.64                                 | 24.64                       | 1.016              |
| 2           | 2                   | 0               | 1.068            | 23.88                                 | 47.75                       | 1.220              |
| 3           | 3                   | 0               | 1.074            | 23.73                                 | 71.18                       | 2.602              |
| 4           | 4                   | 0               | 1.105            | 23.08                                 | 92.31                       | 2.872              |
| 5           | 5                   | 0               | 1.096            | 23.29                                 | 116.43                      | 5.226              |
| 6           | 6                   | 0               | 1.130            | 22.79                                 | 136.74                      | 5.571              |
| 7           | 6                   | 1               | 1.100            | 23.19                                 | 139.16                      | 8.958              |
| 8           | 6                   | 2               | 1.101            | 23.16                                 | 138.96                      | 8.951              |
| 9           | 6                   | 3               | 1.079            | 23.63                                 | 141.81                      | 8.951              |
| 10          | 6                   | 4               | 1.075            | 23.71                                 | 142.28                      | 8.946              |

Full original test results are here:

*https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/testlogs/swinv2-base-results.txt*



















































































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
```

**Max performance:**

• When the concurrency is 300 and the prompt length is 1024, TPS = 1473.44, TTFT = 30.861s (Non-Stream Mode).
• When the concurrency is 300 and the prompt length is 2048, TPS = 849.75, TTFT = 50.730s (Non-Stream Mode).

**Overall performance:**

The overall throughput averages 735.12 tokens/s, with a P90 of 1184.06 tokens/s. Full test results are as follows:

| Concurrency | Prompt Length | Total Requests | Mode       | Success Count | Fail Count | Average Latency (s) | Average TTFT (s) | Average Token Throughput (tokens/s) | Overall Throughput (tokens/s) |
| ----------- | ------------- | -------------- | ---------- | ------------- | ---------- | ------------------- | ---------------- | ----------------------------------- | ----------------------------- |
| 300         | 128           | 20             | Non-Stream | 20            | 0          | 42.786              | 42.786           | 16.25                               | 259.47                        |
| 300         | 128           | 20             | Stream     | 20            | 0          | 41.799              | 0.971            | 15.86                               | 215.46                        |
| 300         | 128           | 30             | Non-Stream | 30            | 0          | 36.526              | 36.526           | 18.79                               | 464.05                        |
| 300         | 128           | 30             | Stream     | 30            | 0          | 29.335              | 1.016            | 22.19                               | 404.16                        |
| 300         | 128           | 40             | Non-Stream | 40            | 0          | 34.573              | 34.573           | 19.98                               | 635.16                        |
| 300         | 128           | 40             | Stream     | 40            | 0          | 37.575              | 1.096            | 17.29                               | 609.03                        |
| 300         | 128           | 50             | Non-Stream | 50            | 0          | 25.340              | 25.340           | 26.43                               | 1092.32                       |
| 300         | 128           | 50             | Stream     | 50            | 0          | 54.118              | 1.994            | 11.59                               | 438.72                        |
| 300         | 256           | 10             | Non-Stream | 10            | 0          | 31.659              | 31.659           | 26.99                               | 217.86                        |
| 300         | 256           | 10             | Stream     | 10            | 0          | 48.118              | 0.411            | 18.50                               | 90.95                         |
| 300         | 256           | 20             | Non-Stream | 20            | 0          | 23.250              | 23.250           | 34.82                               | 623.39                        |
| 300         | 256           | 20             | Stream     | 20            | 0          | 48.669              | 0.887            | 15.52                               | 259.49                        |
| 300         | 256           | 30             | Non-Stream | 30            | 0          | 41.130              | 41.130           | 20.32                               | 456.73                        |
| 300         | 256           | 30             | Stream     | 30            | 0          | 57.212              | 1.548            | 13.65                               | 323.89                        |
| 300         | 256           | 40             | Non-Stream | 40            | 0          | 57.891              | 57.891           | 14.17                               | 496.40                        |
| 300         | 256           | 40             | Stream     | 40            | 0          | 52.031              | 2.474            | 14.83                               | 435.96                        |
| 300         | 256           | 50             | Non-Stream | 50            | 0          | 45.228              | 45.228           | 17.69                               | 725.04                        |
| 300         | 256           | 50             | Stream     | 50            | 0          | 43.595              | 1.257            | 16.95                               | 712.82                        |
| 300         | 512           | 10             | Non-Stream | 10            | 0          | 32.092              | 32.092           | 26.78                               | 242.20                        |
| 300         | 512           | 10             | Stream     | 10            | 0          | 25.930              | 0.568            | 31.35                               | 245.37                        |
| 300         | 512           | 20             | Non-Stream | 20            | 0          | 34.330              | 34.330           | 26.04                               | 444.89                        |
| 300         | 512           | 20             | Stream     | 20            | 0          | 34.694              | 1.629            | 23.48                               | 408.55                        |
| 300         | 512           | 30             | Non-Stream | 30            | 0          | 34.773              | 34.773           | 25.91                               | 632.48                        |
| 300         | 512           | 30             | Stream     | 30            | 0          | 31.973              | 0.970            | 25.72                               | 632.10                        |
| 300         | 512           | 40             | Non-Stream | 40            | 0          | 36.616              | 36.616           | 24.19                               | 851.76                        |
| 300         | 512           | 40             | Stream     | 40            | 0          | 34.922              | 1.091            | 23.83                               | 783.17                        |
| 300         | 512           | 50             | Non-Stream | 50            | 0          | 36.638              | 36.638           | 24.40                               | 1003.91                       |
| 300         | 512           | 50             | Stream     | 50            | 0          | 34.217              | 1.433            | 23.82                               | 940.82                        |
| 300         | 1024          | 10             | Non-Stream | 10            | 0          | 28.029              | 28.029           | 36.46                               | 305.37                        |
| 300         | 1024          | 10             | Stream     | 10            | 0          | 30.585              | 0.428            | 31.08                               | 246.82                        |
| 300         | 1024          | 20             | Non-Stream | 20            | 0          | 31.945              | 31.945           | 32.23                               | 559.50                        |
| 300         | 1024          | 20             | Stream     | 20            | 0          | 24.585              | 0.949            | 37.25                               | 595.32                        |
| 300         | 1024          | 30             | Non-Stream | 30            | 0          | 30.950              | 30.950           | 33.02                               | 852.51                        |
| 300         | 1024          | 30             | Stream     | 30            | 0          | 25.622              | 1.014            | 36.02                               | 951.37                        |
| 300         | 1024          | 40             | Non-Stream | 40            | 0          | 31.642              | 31.642           | 32.85                               | 1198.05                       |
| 300         | 1024          | 40             | Stream     | 40            | 0          | 28.190              | 1.099            | 33.01                               | 1099.36                       |
| 300         | 1024          | 50             | Non-Stream | 50            | 0          | 30.861              | 30.861           | 32.97                               | 1473.44                       |
| 300         | 1024          | 50             | Stream     | 50            | 0          | 31.885              | 1.121            | 29.28                               | 1238.09                       |
| 300         | 2048          | 10             | Non-Stream | 10            | 0          | 27.862              | 27.862           | 42.47                               | 348.38                        |
| 300         | 2048          | 10             | Stream     | 10            | 0          | 27.356              | 0.439            | 36.49                               | 329.59                        |





