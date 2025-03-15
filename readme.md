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

======Backup cli for developing=============

```
(aml_env) PS C:\Users\xinyuwei> az ml online-endpoint show --name "custom-endpoint-1741852362" --resource-group "AIrg1" --workspace-name "aml-david-1" --subscription "53039473-9bbd-499d-90d7-d046d4fa63b6" --query "scoring_uri" --output tsv
https://custom-endpoint-1741852362.polandcentral.inference.ml.azure.com/score
(aml_env) PS C:\Users\xinyuwei> az ml online-endpoint get-credentials --name "custom-endpoint-1741852362" --resource-group "AIrg1" --workspace-name "aml-david-1" --subscription "53039473-9bbd-499d-90d7-d046d4fa63b6" --output json
{
  "primaryKey": "5RegBW6MoJ40EPa3FmAqCn2wx7tJnKEimWvoKkATDrGBx1qKcHtYJQQJ99BCAAAAAAAAAAAAINFRAZMLyndR",
  "secondaryKey": "7H3hhLy65SKSikS5hlpsVMxCaTyI40WTTF7sukK5p3OHlBeRAPegJQQJ99BCAAAAAAAAAAAAINFRAZML20M1"
}
(aml_env) PS C:\Users\xinyuwei>
```



###  Fast Performance Test AI Model on AML Model Catalog

The primary goal of performance testing is to verify tokens/s and TTFT during the inference process. To better simulate real-world scenarios, I have set up several common LLM/SLM use cases in the test script. Additionally, to ensure tokens/s performance, the test script needs to load the corresponding model's tokenizer during execution.



Before officially starting the test, you need to log in to HF on your terminal.

```
huggingface-cli  login
```

#### Phi-4 Series test

**Run the test script:**

```
(aml_env) root@pythonvm:~/AIFperformance# python press-phi4-0314.py
Please enter the API service URL: https://david-workspace-westeurop-ldvdq.westeurope.inference.ml.azure.com/score
Please enter the API Key: Ef9DFpATsXs4NiWyoVhEXeR4PWPvFy17xcws5ySCvV2H8uOUfgV4JQQJ99BCAAAAAAAAAAAAINFRAZML3eIO
Please enter the full name of the HuggingFace model for tokenizer loading: microsoft/phi-4
Tokenizer loaded successfully: microsoft/phi-4
```



Test result analyze：

### **Performance Comparison: Single Request Metrics (TTFT and tokens/s)**

 

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



### **Performance Comparison: Overall Throughput at Concurrency = 2**

 

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





======================        Under developing part              =======================

Before deployment, you need to check which region under your subscription has the quota for deploying AML GPU VMs. If your quota is in a specific region, then the workspace and resource group you select below should also be in the same region to ensure a successful deployment. If none of the regions have a quota, you will need to submit a request on the Azure portal. 

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/16.png)



pip install azure-ai-ml azure-identity requests azure-cli































































```
(aml_env) root@pythonvm:~/AIFperformance# python checkgpuquota.py
```

```
Please input RESOURCE_GROUP: A100VM_group
Please inpu WORKSPACE_NAME: david-workspace-westeurope
Region,ResourceName,LocalizedValue,Usage,Limit
eastus,standardNCADSH100v5Family,,0,80
eastus2,standardNCADSH100v5Family,,0,40
```

As shown in the results above, the AML in my subscription has an H100 quota in eastus and eastus2. "40" indicates that NC40ads_H100_v5 can be deployed, and "80" indicates that NC80ads_H100_v5 can be deployed.

| SKU Name                 | GPU Count | GPU Memory (VRAM) | CPU Cores |
| ------------------------ | --------- | ----------------- | --------- |
| Standard_NC24ads_A100_v4 | 1         | 40 GB             | 24        |
| Standard_NC48ads_A100_v4 | 2         | 80 GB (2x40 GB)   | 48        |
| Standard_NC96ads_A100_v4 | 4         | 160 GB (4x40 GB)  | 96        |
| Standard_NC40ads_H100_v5 | 1         | 80 GB             | 40        |
| Standard_NC80ads_H100_v5 | 2         | 160 GB (2x80 GB)  | 80        |

If you have a quota, you can continue to deploy resources.

Check available model first, for example, you want to deploy phi-4 series:

```
(aml_env) root@pythonvm:~/AIFperformance# az ml model list --registry-name AzureML --query "[?contains(name, 'Phi-4')]" --output tableName                       Description    Latest version
-------------------------  -------------  ----------------
Phi-4-multimodal-instruct                 1
Phi-4-mini-instruct                       1
Phi-4                                     7
```

To create a model deployment using a program, you need to specify the model name, subscription ID, resource group name, VM SKU, and the number of VMs.

```
# python deploy_infra.py
2025-02-24 21:39:20,774 - ERROR - Usage: python deploy_infra.py <model_name> <model_version> <subscription_id> <resource_group> <workspace_name> <instance_type> <instance_count>
2025-02-24 21:39:20,775 - ERROR - instance_type options: Standard_NC24ads_A100_v4, Standard_NC48ads_A100_v4, Standard_NC96ads_A100_v4, Standard_NC40ads_H100_v5, Standard_NC80ads_H100_v5
```

Next, deploy the "Phi-3-medium-4k-instruct" deployment using the VM SKU "Standard_NC24ads_A100_v4" with a quantity of 1.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/15.png)

```
# python deploy_infra.py "Phi-4-mini-instruct" "1" "08f95cfd-64fe-4187-99bb-7b3e661c4cde" "A100VM_group" "david-workspace-westeurope" "Standard_NC40ads_H100_v5" 1
```



### Test the performance of the deployment AI model

First, check the usage instructions of the `concurrency_test.py` program.

```
(aml_env) root@davidwei:~/AML_MAAP_benchmark# python concurrency_test.py
usage: concurrency_test.py [-h] --endpoint_url ENDPOINT_URL --api_key API_KEY [--initial_concurrency INITIAL_CONCURRENCY]
                           [--prompt_sizes PROMPT_SIZES [PROMPT_SIZES ...]] [--response_sizes RESPONSE_SIZES [RESPONSE_SIZES ...]] [--max_tests MAX_TESTS]
                           [--output_file OUTPUT_FILE] [--max_concurrency MAX_CONCURRENCY]
concurrency_test.py: error: the following arguments are required: --endpoint_url, --api_key
```



Invoke `concurrency_test.py` to stress test the deployment, configuring parameters such as input and output tokens.

```
#python concurrency_test.py --endpoint_url "https://xinyuwei-9556-jyhjv.westeurope.inference.ml.azure.com/score" --api_key "A2ZdX5yDwbu11ZYKeuznMqoU69GHyRZvU7IbaDPZDkmYH2J1Ia6VJQQJ99BBAAAAAAAAAAAAINFRAZML5E10" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results.csv" --max_concurrency 10


--------------------------------------------------
Testing combination: Concurrency=10, Prompt Size=1024, Response Size=128
Concurrency: 10
Prompt Size: 1024
Response Size: 128
Successful Requests: 10
Failed Requests: 0
Failure Rate: 0.00%
Average Latency (seconds): 3.44
Average TTFT (seconds): 3.44
Throughput (tokens/second): 21.74
Total Execution Time (seconds): 37.21
--------------------------------------------------
Reached maximum concurrency limit of 10.

Best Throughput Achieved:
Concurrency: 7
Prompt Size: 2048
Response Size: 4096
Throughput (tokens/second): 53.20
Average Latency (seconds): 7.97
Average TTFT (seconds): 7.97

Test completed. Results saved to results.csv

```

Check the final test result：

| concurrency | prompt_size | response_size | successful_requests | failed_requests | avg_latency | avg_ttft | throughput | total_execution_time | error_status_codes |
| ----------- | ----------- | ------------- | ------------------- | --------------- | ----------- | -------- | ---------- | -------------------- | ------------------ |
| 1           | 1024        | 2048          | 1                   | 0               | 2.181651    | 2.18167  | 23.37211   | 2.182088375          | {}                 |
| 2           | 1024        | 2048          | 2                   | 0               | 2.659276    | 2.659293 | 28.54849   | 3.082474232          | {}                 |
| 3           | 1024        | 2048          | 3                   | 0               | 4.709896    | 4.709919 | 34.3977    | 7.587716579          | {}                 |
| 4           | 1024        | 2048          | 4                   | 0               | 4.48061     | 4.480646 | 38.37763   | 13.15870714          | {}                 |
| 5           | 1024        | 2048          | 5                   | 0               | 3.949267    | 3.949291 | 33.49872   | 18.8066864           | {}                 |
| 6           | 1024        | 2048          | 6                   | 0               | 6.224106    | 6.224127 | 41.00855   | 25.21425176          | {}                 |
| 7           | 1024        | 2048          | 7                   | 0               | 3.935237    | 3.93527  | 25.43518   | 27.0098393           | {}                 |
| 8           | 1024        | 2048          | 8                   | 0               | 5.495956    | 5.495985 | 36.95024   | 27.90239167          | {}                 |
| 9           | 1024        | 2048          | 9                   | 0               | 3.207933    | 3.207955 | 40.74859   | 15.70606422          | {}                 |
| 10          | 1024        | 2048          | 10                  | 0               | 3.45364     | 3.453665 | 24.30065   | 27.7770319           | {}                 |
| 1           | 2048        | 64            | 1                   | 0               | 2.753683    | 2.753703 | 19.9696    | 2.75418663           | {}                 |
| 2           | 2048        | 64            | 2                   | 0               | 3.243599    | 3.243613 | 27.65017   | 3.905944109          | {}                 |
| 3           | 2048        | 64            | 3                   | 0               | 3.541899    | 3.541919 | 20.01485   | 7.74424839           | {}                 |
| 4           | 2048        | 64            | 4                   | 0               | 3.211296    | 3.211316 | 29.36803   | 7.42303896           | {}                 |
| 5           | 2048        | 64            | 5                   | 0               | 3.1162      | 3.116215 | 19.03373   | 14.29041672          | {}                 |
| 6           | 2048        | 64            | 6                   | 0               | 3.155113    | 3.155133 | 19.44056   | 16.2032392           | {}                 |
| 7           | 2048        | 64            | 7                   | 0               | 2.955534    | 2.955553 | 14.62544   | 24.9565115           | {}                 |
| 8           | 2048        | 64            | 8                   | 0               | 3.374602    | 3.374641 | 15.41315   | 26.53578424          | {}                 |
| 9           | 2048        | 64            | 9                   | 0               | 3.223261    | 3.223279 | 17.0901    | 26.85765004          | {}                 |
| 10          | 2048        | 64            | 10                  | 0               | 3.240726    | 3.240743 | 13.64551   | 37.4482038           | {}                 |
| 1           | 1024        | 1024          | 1                   | 0               | 8.905204    | 8.905224 | 50.97897   | 8.905632257          | {}                 |
| 2           | 1024        | 1024          | 2                   | 0               | 3.82329     | 3.823306 | 20.46571   | 4.299874306          | {}                 |
| 3           | 1024        | 1024          | 3                   | 0               | 4.291468    | 4.29149  | 43.74643   | 8.206383705          | {}                 |
| 4           | 1024        | 1024          | 4                   | 0               | 5.83485     | 5.834883 | 49.99117   | 14.84262228          | {}                 |
| 5           | 1024        | 1024          | 5                   | 0               | 3.832849    | 3.832875 | 37.71236   | 14.00071311          | {}                 |
| 6           | 1024        | 1024          | 6                   | 0               | 3.091236    | 3.09126  | 29.50247   | 15.62581229          | {}                 |
| 7           | 1024        | 1024          | 7                   | 0               | 3.985303    | 3.985327 | 24.42945   | 31.92867732          | {}                 |
| 8           | 1024        | 1024          | 8                   | 0               | 2.955142    | 2.955167 | 17.13504   | 27.37081718          | {}                 |
| 9           | 1024        | 1024          | 9                   | 0               | 3.793313    | 3.793339 | 33.02927   | 26.09806323          | {}                 |
| 10          | 1024        | 1024          | 10                  | 0               | 3.553602    | 3.553631 | 27.23539   | 30.14460588          | {}                 |
| 1           | 2048        | 1024          | 1                   | 0               | 4.298271    | 4.298286 | 35.82447   | 4.298737764          | {}                 |
| 2           | 2048        | 1024          | 2                   | 0               | 3.961102    | 3.961128 | 33.69973   | 4.836833477          | {}                 |
| 3           | 2048        | 1024          | 3                   | 0               | 8.210444    | 8.210467 | 52.85939   | 17.02630162          | {}                 |
| 4           | 2048        | 1024          | 4                   | 0               | 9.815956    | 9.815979 | 44.56057   | 27.53555703          | {}                 |
| 5           | 2048        | 1024          | 5                   | 0               | 6.560005    | 6.560024 | 40.88134   | 20.40050721          | {}                 |
| 6           | 2048        | 1024          | 6                   | 0               | 5.739161    | 5.739182 | 37.97911   | 26.43558216          | {}                 |
| 7           | 2048        | 1024          | 7                   | 0               | 6.897866    | 6.897887 | 46.87257   | 30.01755333          | {}                 |
| 8           | 2048        | 1024          | 8                   | 0               | 7.597272    | 7.597297 | 38.04828   | 37.24215508          | {}                 |
| 9           | 2048        | 1024          | 9                   | 0               | 6.396977    | 6.396999 | 46.64886   | 42.05890155          | {}                 |
| 10          | 2048        | 1024          | 10                  | 0               | 7.478377    | 7.478397 | 38.48196   | 51.29676986          | {}                 |
| 1           | 64          | 1024          | 1                   | 0               | 5.192025    | 5.192043 | 50.8426    | 5.192496538          | {}                 |
| 2           | 64          | 1024          | 2                   | 0               | 3.015155    | 3.015171 | 37.99386   | 3.605846167          | {}                 |
| 3           | 64          | 1024          | 3                   | 0               | 2.770903    | 2.770927 | 32.94883   | 5.827218294          | {}                 |
| 4           | 64          | 1024          | 4                   | 0               | 3.669078    | 3.669098 | 35.94048   | 11.35210299          | {}                 |
| 5           | 64          | 1024          | 5                   | 0               | 4.519947    | 4.519993 | 36.70499   | 16.45552635          | {}                 |
| 6           | 64          | 1024          | 6                   | 0               | 5.526716    | 5.52674  | 45.81613   | 20.40766168          | {}                 |
| 7           | 64          | 1024          | 7                   | 0               | 2.854245    | 2.854279 | 20.08975   | 24.44031954          | {}                 |
| 8           | 64          | 1024          | 8                   | 0               | 4.422138    | 4.422167 | 21.32232   | 38.03525805          | {}                 |
| 9           | 64          | 1024          | 9                   | 0               | 4.831736    | 4.831768 | 36.06449   | 36.21291065          | {}                 |
| 10          | 64          | 1024          | 10                  | 0               | 4.365419    | 4.365444 | 27.69334   | 37.40971231          | {}                 |
| 1           | 64          | 2048          | 1                   | 0               | 3.783458    | 3.783475 | 39.64163   | 3.783900738          | {}                 |
| 2           | 64          | 2048          | 2                   | 0               | 2.917082    | 2.917104 | 37.96118   | 3.529922009          | {}                 |
| 3           | 64          | 2048          | 3                   | 0               | 3.69923     | 3.699253 | 41.81922   | 8.967169285          | {}                 |
| 4           | 64          | 2048          | 4                   | 0               | 5.208706    | 5.208734 | 40.68387   | 15.33777308          | {}                 |
| 5           | 64          | 2048          | 5                   | 0               | 3.047798    | 3.04783  | 27.12727   | 13.67627382          | {}                 |
| 6           | 64          | 2048          | 6                   | 0               | 4.871371    | 4.871394 | 28.578     | 27.32871723          | {}                 |
| 7           | 64          | 2048          | 7                   | 0               | 3.790402    | 3.790425 | 28.72959   | 27.32374406          | {}                 |
| 8           | 64          | 2048          | 8                   | 0               | 6.158503    | 6.158533 | 30.64515   | 50.05685687          | {}                 |
| 9           | 64          | 2048          | 9                   | 0               | 3.78984     | 3.789892 | 21.48182   | 37.79939628          | {}                 |
| 10          | 64          | 2048          | 10                  | 0               | 4.94908     | 4.949111 | 38.50745   | 39.70660496          | {}                 |
| 1           | 64          | 64            | 1                   | 0               | 3.321778    | 3.321802 | 15.04972   | 3.3223207            | {}                 |
| 2           | 64          | 64            | 2                   | 0               | 2.895008    | 2.895078 | 32.98165   | 3.365507841          | {}                 |
| 3           | 64          | 64            | 3                   | 0               | 2.60884     | 2.608863 | 25.18071   | 6.433495283          | {}                 |
| 4           | 64          | 64            | 4                   | 0               | 2.639774    | 2.639804 | 30.15736   | 7.261909246          | {}                 |
| 5           | 64          | 64            | 5                   | 0               | 2.542551    | 2.542575 | 22.89394   | 12.09927177          | {}                 |
| 6           | 64          | 64            | 6                   | 0               | 3.070914    | 3.070936 | 21.29448   | 15.77873421          | {}                 |
| 7           | 64          | 64            | 7                   | 0               | 2.627481    | 2.627503 | 26.9233    | 14.22559643          | {}                 |
| 8           | 64          | 64            | 8                   | 0               | 2.674476    | 2.674501 | 16.97603   | 25.8599925           | {}                 |
| 9           | 64          | 64            | 9                   | 0               | 2.706192    | 2.706214 | 13.36653   | 37.25723648          | {}                 |
| 10          | 64          | 64            | 10                  | 0               | 2.633372    | 2.633397 | 14.85681   | 37.22199941          | {}                 |
| 1           | 2048        | 4096          | 1                   | 0               | 4.666986    | 4.667016 | 35.13607   | 4.667567492          | {}                 |
| 2           | 2048        | 4096          | 2                   | 0               | 4.750616    | 4.750636 | 19.22022   | 5.410969257          | {}                 |
| 3           | 2048        | 4096          | 3                   | 0               | 8.137676    | 8.137692 | 39.69637   | 17.70942569          | {}                 |
| 4           | 2048        | 4096          | 4                   | 0               | 5.468486    | 5.468511 | 40.08852   | 15.46577311          | {}                 |
| 5           | 2048        | 4096          | 5                   | 0               | 5.470666    | 5.470684 | 39.45249   | 18.12306428          | {}                 |
| 6           | 2048        | 4096          | 6                   | 0               | 5.921795    | 5.921815 | 30.75044   | 29.49551463          | {}                 |
| 7           | 2048        | 4096          | 7                   | 0               | 7.974852    | 7.974875 | 53.19904   | 29.75617766          | {}                 |
| 8           | 2048        | 4096          | 8                   | 0               | 6.627603    | 6.627624 | 37.50555   | 46.07317686          | {}                 |
| 9           | 2048        | 4096          | 9                   | 0               | 5.164328    | 5.164353 | 26.40785   | 41.91935372          | {}                 |
| 10          | 2048        | 4096          | 10                  | 0               | 7.520934    | 7.520956 | 40.60979   | 53.48464084          | {}                 |
| 1           | 64          | 4096          | 1                   | 0               | 3.687285    | 3.6873   | 42.57328   | 3.687758923          | {}                 |
| 2           | 64          | 4096          | 2                   | 0               | 5.182035    | 5.182054 | 42.92568   | 6.220052481          | {}                 |
| 3           | 64          | 4096          | 3                   | 0               | 3.335014    | 3.335035 | 28.9469    | 7.807399035          | {}                 |
| 4           | 64          | 4096          | 4                   | 0               | 3.189085    | 3.189113 | 33.31      | 12.36865664          | {}                 |
| 5           | 64          | 4096          | 5                   | 0               | 4.2263      | 4.226328 | 48.76822   | 12.63117766          | {}                 |
| 6           | 64          | 4096          | 6                   | 0               | 3.679967    | 3.679999 | 33.43512   | 16.15068221          | {}                 |
| 7           | 64          | 4096          | 7                   | 0               | 4.724797    | 4.724826 | 32.22914   | 28.42148399          | {}                 |
| 8           | 64          | 4096          | 8                   | 0               | 4.929248    | 4.929276 | 47.36196   | 27.0259068           | {}                 |
| 9           | 64          | 4096          | 9                   | 0               | 4.212567    | 4.212596 | 25.17346   | 39.84354281          | {}                 |
| 10          | 64          | 4096          | 10                  | 0               | 5.931706    | 5.931734 | 38.23398   | 51.1586802           | {}                 |
| 1           | 1024        | 128           | 1                   | 0               | 2.374724    | 2.374737 | 22.73488   | 2.375204802          | {}                 |
| 2           | 1024        | 128           | 2                   | 0               | 3.090174    | 3.09019  | 36.67964   | 4.116725683          | {}                 |
| 3           | 1024        | 128           | 3                   | 0               | 3.479532    | 3.479552 | 24.23492   | 7.262248993          | {}                 |
| 4           | 1024        | 128           | 4                   | 0               | 2.793226    | 2.793246 | 32.55438   | 6.635052681          | {}                 |
| 5           | 1024        | 128           | 5                   | 0               | 3.323107    | 3.323134 | 27.36045   | 13.99830818          | {}                 |
| 6           | 1024        | 128           | 6                   | 0               | 3.702325    | 3.702346 | 30.32033   | 15.86394095          | {}                 |
| 7           | 1024        | 128           | 7                   | 0               | 2.912894    | 2.912918 | 17.71729   | 24.72160792          | {}                 |
| 8           | 1024        | 128           | 8                   | 0               | 3.403699    | 3.403722 | 19.62896   | 27.4084816           | {}                 |
| 9           | 1024        | 128           | 9                   | 0               | 3.305229    | 3.305251 | 16.09414   | 36.72143459          | {}                 |
| 10          | 1024        | 128           | 10                  | 0               | 3.437237    | 3.437261 | 21.74256   | 37.20812988          | {}                 |



View the source code of the program.:

```
(aml_env) root@davidwei:~/AML_MAAP_benchmark# cat concurrency_test.py
import os
import json
import ssl
import requests
import threading
import argparse
import csv
import random
from time import time, sleep

# Allow self-signed HTTPS certificates (if required)
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Function to invoke the endpoint for a single request with retry mechanism and jitter
def invoke_endpoint(url, api_key, input_string, max_new_tokens, results_list, lock, max_retries=5, initial_delay=1, max_delay=10):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "input_data": {
            "input_string": input_string,
            "parameters": {
                "temperature": 0.7,
                "top_p": 1,
                "max_new_tokens": max_new_tokens
            }
        }
    }
    retries = 0
    delay = initial_delay

    while retries <= max_retries:
        try:
            start_time = time()
            response = requests.post(url, json=data, headers=headers, timeout=60)
            latency = time() - start_time  # Total latency
            if response.status_code == 200:
                result = response.json()
                first_token_time = time()  # Assuming we get the full response at once
                ttft = first_token_time - start_time
                output_content = result.get('output', '')
                output_tokens = len(output_content.split())
                with lock:
                    results_list.append({
                        "success": True,
                        "latency": latency,
                        "ttft": ttft,
                        "output_tokens": output_tokens
                    })
                return
            elif response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    with lock:
                        results_list.append({
                            "success": False,
                            "status_code": response.status_code,
                            "error": response.reason
                        })
                    return
                else:
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        delay = max(float(retry_after), delay)
                    else:
                        jitter = random.uniform(0, 1)
                        delay = min(delay * 2 + jitter, max_delay)
                    sleep(delay)
            else:
                with lock:
                    results_list.append({
                        "success": False,
                        "status_code": response.status_code,
                        "error": response.reason
                    })
                return
        except Exception as e:
            with lock:
                results_list.append({
                    "success": False,
                    "error": str(e)
                })
            return

# Function to test a specific combination of concurrency, prompt_size, and response_size
def test_combination(endpoint_url, api_key, concurrency, prompt_size, response_size):
    # Generate input prompts with specified size
    base_prompt = "Sample input prompt with token size."
    repeat_times = max(1, int(prompt_size / len(base_prompt.split())))
    prompt_content = " ".join([base_prompt] * repeat_times)
    input_prompts = [
        {"role": "user", "content": prompt_content}
    ] * concurrency  # Duplicate the prompt for testing concurrency

    results_list = []
    lock = threading.Lock()
    threads = []

    total_start_time = time()

    for i in range(concurrency):
        t = threading.Thread(target=invoke_endpoint, args=(
            endpoint_url,
            api_key,
            [input_prompts[i]],
            response_size,
            results_list,
            lock
        ))
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    total_execution_time = time() - total_start_time

    # Aggregate statistics
    total_latency = 0
    total_ttft = 0
    total_tokens = 0
    successful_requests = 0
    failed_requests = 0
    error_status_codes = {}

    for result in results_list:
        if result["success"]:
            total_latency += result["latency"]
            total_ttft += result["ttft"]
            total_tokens += result["output_tokens"]
            successful_requests += 1
        else:
            failed_requests += 1
            status_code = result.get("status_code", "Unknown")
            error_status_codes[status_code] = error_status_codes.get(status_code, 0) + 1

    avg_latency = total_latency / successful_requests if successful_requests > 0 else 0
    avg_ttft = total_ttft / successful_requests if successful_requests > 0 else 0
    throughput = total_tokens / total_execution_time if total_execution_time > 0 else 0

    return {
        "concurrency": concurrency,
        "prompt_size": prompt_size,
        "response_size": response_size,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "avg_latency": avg_latency,
        "avg_ttft": avg_ttft,
        "throughput": throughput,
        "total_execution_time": total_execution_time,
        "error_status_codes": error_status_codes
    }

# Main function to adaptively adjust concurrency
def main(endpoint_url, api_key, initial_concurrency, prompt_sizes, response_sizes, max_tests, output_file, max_concurrency):
    results = []
    test_count = 0

    print("Starting concurrency testing...\n")

    # Generate all possible prompt and response size combinations
    pr_combinations = [
        (prompt_size, response_size)
        for prompt_size in prompt_sizes
        for response_size in response_sizes
    ]

    # Randomly shuffle the combinations to avoid systematic biases
    random.shuffle(pr_combinations)

    for prompt_size, response_size in pr_combinations:
        concurrency = initial_concurrency
        min_concurrency = 1
        # Use the max_concurrency passed from the arguments
        while test_count < max_tests and concurrency <= max_concurrency:
            print(f"Testing combination: Concurrency={concurrency}, Prompt Size={prompt_size}, Response Size={response_size}")
            result = test_combination(endpoint_url, api_key, concurrency, prompt_size, response_size)
            results.append(result)
            test_count += 1

            # Print results for this combination
            total_requests = result['successful_requests'] + result['failed_requests']
            failure_rate = result['failed_requests'] / total_requests if total_requests > 0 else 0

            print(f"Concurrency: {result['concurrency']}")
            print(f"Prompt Size: {result['prompt_size']}")
            print(f"Response Size: {result['response_size']}")
            print(f"Successful Requests: {result['successful_requests']}")
            print(f"Failed Requests: {result['failed_requests']}")
            print(f"Failure Rate: {failure_rate*100:.2f}%")
            print(f"Average Latency (seconds): {result['avg_latency']:.2f}")
            print(f"Average TTFT (seconds): {result['avg_ttft']:.2f}")
            print(f"Throughput (tokens/second): {result['throughput']:.2f}")
            print(f"Total Execution Time (seconds): {result['total_execution_time']:.2f}")
            if result["failed_requests"] > 0:
                print(f"Error Status Codes: {result['error_status_codes']}")
            print("-" * 50)

            # Adaptive concurrency adjustment
            if failure_rate > 0.2:
                # Reduce concurrency if failure rate is high
                concurrency = max(concurrency - 1, min_concurrency)
                if concurrency == min_concurrency:
                    print("Concurrency reduced to minimum due to high failure rate.")
                    break
            else:
                # Increase concurrency to test higher loads
                concurrency = concurrency + 1

            # Limit the concurrency to max_concurrency
            if concurrency > max_concurrency:
                print(f"Reached maximum concurrency limit of {max_concurrency}.")
                break

    # Find the combination with the maximum throughput
    if results:
        best_throughput_result = max(results, key=lambda x: x['throughput'])

        print("\nBest Throughput Achieved:")
        print(f"Concurrency: {best_throughput_result['concurrency']}")
        print(f"Prompt Size: {best_throughput_result['prompt_size']}")
        print(f"Response Size: {best_throughput_result['response_size']}")
        print(f"Throughput (tokens/second): {best_throughput_result['throughput']:.2f}")
        print(f"Average Latency (seconds): {best_throughput_result['avg_latency']:.2f}")
        print(f"Average TTFT (seconds): {best_throughput_result['avg_ttft']:.2f}")
    else:
        print("No successful test results to report.")

    # Save results to CSV
    with open(output_file, mode='w', newline='') as file:
        fieldnames = [
            "concurrency", "prompt_size", "response_size",
            "successful_requests", "failed_requests", "avg_latency",
            "avg_ttft", "throughput", "total_execution_time", "error_status_codes"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Convert error_status_codes dict to string for CSV
            result['error_status_codes'] = json.dumps(result['error_status_codes'])
            writer.writerow(result)

    print(f"\nTest completed. Results saved to {output_file}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Concurrency and throughput testing of Azure ML Endpoint using threading")
    parser.add_argument('--endpoint_url', type=str, required=True, help="URL of the Azure ML Endpoint")
    parser.add_argument('--api_key', type=str, required=True, help="API key for the Azure ML Endpoint")
    parser.add_argument('--initial_concurrency', type=int, default=1, help="Initial concurrency level to start testing")
    parser.add_argument('--prompt_sizes', type=int, nargs='+', default=[64, 128, 256], help="List of input prompt sizes in tokens")
    parser.add_argument('--response_sizes', type=int, nargs='+', default=[64, 128, 256], help="List of output response sizes in tokens")
    parser.add_argument('--max_tests', type=int, default=30, help="Maximum number of tests to perform")
    parser.add_argument('--output_file', type=str, default="concurrency_test_final_results.csv", help="Output CSV file")
    parser.add_argument('--max_concurrency', type=int, default=50, help="Maximum concurrency level to test") 
    args = parser.parse_args()

    # Run the main function
    main(
        endpoint_url=args.endpoint_url,
        api_key=args.api_key,
        initial_concurrency=args.initial_concurrency,
        prompt_sizes=args.prompt_sizes,
        response_sizes=args.response_sizes,
        max_tests=args.max_tests,
        output_file=args.output_file,
        max_concurrency=args.max_concurrency  # Pass the maximum concurrency parameter
    )
```

### Extra testing cli

Phi-3-small-8k-instruct(7.39B) on Standard_NC24ads_A100_v4, results refer to : ***results-NC24-phi3.csv***

```
python concurrency_test.py --endpoint_url "https://admin-0046-kslbq-48.eastus2.inference.ml.azure.com/score" --api_key "ENsUl1bg6BBj4ZxixddaQK1bz9ytFOhhnvqwfk2on9KzOGkLc4arJQQJ99BBAAAAAAAAAAAAINFRAZML4CVw" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results.csv" --max_concurrency 10

```

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/5.png)

Phi-3-small-8k-instruct(7.39B) on  on Standard_NC24ads_A100_v4, results refer to : ***results-NC48-phi3.csv***

```
python concurrency_test.py --endpoint_url "https://admin-0046-tlgxw.eastus2.inference.ml.azure.com/score" --api_key "6onqC7rYjmAI95zBymMPJTPFk3NtbdCqjav6S96WsxSWWDN0nLZqJQQJ99BBAAAAAAAAAAAAINFRAZML2X1J" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results-24.csv" --max_concurrency 10
```

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/6.png)

Phi-4(14.7B ) on  on Standard_NC24ads_A100_v4, results refer to : ***results-NC24-phi4.csv***

```
python concurrency_test.py --endpoint_url "https://admin-0046-jerzt-24.eastus2.inference.ml.azure.com/score" --api_key "3hD2mSgz2LpriF9ZI4MhiCjjDlEihyFvLwvJZuugIGln2fz19KxhJQQJ99BBAAAAAAAAAAAAINFRAZML1bl3" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results-24.csv" --max_concurrency 10

```

Phi-4(14.7B ) on  on Standard_NC48ads_A100_v4, results refer to : ***results-NC48-phi4.csv***

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/10.png)

```
python concurrency_test.py --endpoint_url "https://admin-0046-tvznu-48.eastus2.inference.ml.azure.com/score" --api_key "FfQh320Ggp8KuLhHiurDzRZhXcP6zLBsdl53ajQAPtbxFJMeIV6LJQQJ99BBAAAAAAAAAAAAINFRAZMLabJg" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results-48-phi4.csv" --max_concurrency 10

```









=====================

### Performance test on Azure AI model inference

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





## L
