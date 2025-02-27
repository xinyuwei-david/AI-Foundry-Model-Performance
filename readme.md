# AI Foundry Model Catalog Endpoint performance

This repository is designed to test the performance of open-source models from the Azure AI Foundry Model Catalog when deployed using Managed Compute across different VBM SKUs. The scripts in this repository consist of two parts: deploying the Managed Compute endpoint and testing the performance of the Managed Compute endpoint. After completing the tests, you should delete it promptly to avoid incurring additional costs.

## Overall conclusion

In the AI Foundry model catalog, many models have published performance metrics, which determine the inference performance of a single deployment instance of such models, whether serverless or managed compute.This is because the factor that determines peak throughput is not the underlying GPU resources of Managed Compute, but the TPM of the model in the Model Catalog.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/2.png)

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/3.png)

Regarding the issue of exposing the `max_concurrent_requests_per_instance` parameter of Managed Compute to users, PG has not yet provided a response. I speculate that this feature might not be supported at this stage, as the entire product is still in the Preview phase.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/4.png)

### How to Fast Deploy Model on AI Foundry Model Catalog

***Refer to：***

*https://learn.microsoft.com/en-us/cli/azure/ml/registry?view=azure-cli-latest*

Before creating a Managed Compute and serverless API, you need to create a resource group in Azure and then visit ai.azure.com to create a Hub and a Project. These common steps will not be elaborated upon in this repository.

![images](https://github.com/xinyuwei-david/AI-Foundry-Model-Performance/blob/main/images/1.png)

Next, prepare the Python environment for running the program. You need to install the required Python packages in this environment and log in to Azure through it. 

```
#conda create -n aml_env python=3.9 -y
#conda activate aml_env
#pip install azure-ai-ml azure-identity requests python-dotenv pyyaml humanfriendly numpy aiohttp  
#apt-get install -y jq  
```

Next, log in to Azure.

```
#az login
```

We know that the Azure AI Foundry Model Catalog allows the deployment of over 1,700 AI models. When deploying, you can choose either the Serverless mode or the Managed Compute mode.

| Features                          | Managed compute                                              | Serverless API (pay-per-token)                               |
| :-------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Deployment experience and billing | Model weights are deployed to dedicated virtual machines with managed compute. A managed compute, which can have one or more deployments, makes available a REST API for inference. You're billed for the virtual machine core hours that the deployments use. | Access to models is through a deployment that provisions an API to access the model. The API provides access to the model that Microsoft hosts and manages, for inference. You're billed for inputs and outputs to the APIs, typically in tokens. Pricing information is provided before you deploy. |
| API authentication                | Keys and Microsoft Entra authentication.                     | Keys only.                                                   |
| Content safety                    | Use Azure AI Content Safety service APIs.                    | Azure AI Content Safety filters are available integrated with inference APIs. Azure AI Content Safety filters are billed separately. |
| Network isolation                 | [Configure managed networks for Azure AI Foundry hubs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/configure-managed-network). | Managed compute follow your hub's public network access (PNA) flag setting. For more information, see the [Network isolation for models deployed via Serverless APIs](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview#network-isolation-for-models-deployed-via-serverless-apis) section later in this article. |

When using the AI Foundry Model Catalog, you can create your own registry or use the default AML..

### Create your own Model Registry

You can **skip this step** if you want to use the model's AML registry. 

```
#cat registry.yml
name: xinyuwei-registry1
tags:
  description: Basic registry with one primary region and to additional regions
  foo: bar
location: eastus
replication_locations:
  - location: eastus
  - location: eastus2
  - location: westus
```



```
(aml_env) root@davidwei:~/AML_MAAP_benchmark# az ml registry create --resource-group rg-admin-2776_ai --file registry.yml
```

```
Class RegistryRegionDetailsSchema: This is an experimental class, and may change at any time. Please see https://aka.ms/azuremlexperimental for more information.
{
  "containerRegistry": null,
  "description": null,
  "discoveryUrl": "https://eastus.api.azureml.ms/registrymanagement/v1.0/registries/xinyuwei-registry1/discovery",
  "identity": {
    "principalId": "4d455e6e-22c9-4281-b1c7-d5f9c7641797",
    "tenantId": "9812d5f8-3c48-49c9-aada-e7174b336629",
    "type": "SystemAssigned",
    "userAssignedIdentities": null
  },
  "intellectualProperty": null,
  "location": "eastus",
  "managedResourceGroup": {
    "resourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418"
  },
  "mlflowRegistryUri": "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/rg-admin-2776_ai/providers/Microsoft.MachineLearningServices/registries/xinyuwei-registry1",
  "name": "xinyuwei-registry1",
  "properties": {},
  "publicNetworkAccess": "Enabled",
  "replicationLocations": [
    {
      "acrConfig": [
        {
          "acrAccountSku": "Premium",
          "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.ContainerRegistry/registries/de35df4bd6e"
        }
      ],
      "location": "eastus",
      "storageConfig": {
        "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.Storage/storageAccounts/471837b7de9",
        "replicatedIds": null,
        "replicationCount": 1,
        "storageAccountHns": false,
        "storageAccountType": "standard_lrs"
      }
    },
    {
      "acrConfig": [
        {
          "acrAccountSku": "Premium",
          "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.ContainerRegistry/registries/de35df4bd6e"
        }
      ],
      "location": "eastus2",
      "storageConfig": {
        "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.Storage/storageAccounts/7a7c0937565",
        "replicatedIds": null,
        "replicationCount": 1,
        "storageAccountHns": false,
        "storageAccountType": "standard_lrs"
      }
    },
    {
      "acrConfig": [
        {
          "acrAccountSku": "Premium",
          "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.ContainerRegistry/registries/de35df4bd6e"
        }
      ],
      "location": "westus",
      "storageConfig": {
        "armResourceId": "/subscriptions/08f95cfd-64fe-4187-99bb-7b3e661c4cde/resourceGroups/azureml-rg-xinyuwei-registry1_fc12d3b8-b1e4-44e1-a461-cf28bb2fc418/providers/Microsoft.Storage/storageAccounts/33ee5e64e33",
        "replicatedIds": null,
        "replicationCount": 1,
        "storageAccountHns": false,
        "storageAccountType": "standard_lrs"
      }
    }
  ],
  "tags": {
    "description": "Basic registry with one primary region and to additional regions",
    "foo": "bar"
  }
}
```



```
# az ml model list --registry-name  xinyuwei-registry1 --resource-group rg-admin-2776_ai
```

```
  {
    "creation_context": {
      "created_at": "2024-12-13T00:56:50.995337+00:00",
      "created_by": "azureml",
      "created_by_type": "User",
      "last_modified_at": "0001-01-01T00:00:00+00:00"
    },
    "description": "",
    "id": "azureml://registries/AzureML/models/Phi-4",
    "latest version": "3",
    "name": "Phi-4",
    "properties": {},
    "stage": null,
    "tags": {}
  },
  {
    "creation_context": {
      "created_at": "2024-12-06T17:14:18.513744+00:00",
      "created_by": "azureml",
      "created_by_type": "User",
      "last_modified_at": "0001-01-01T00:00:00+00:00"
    },
    "description": "",
    "id": "azureml://registries/AzureML/models/supply-chain-trade-regulations",
    "latest version": "2",
    "name": "supply-chain-trade-regulations",
    "properties": {},
    "stage": null,
    "tags": {}
  },
```

### Create models from the AML registry

View the models in the AML registry

```
(aml_env) root@davidwei:~/AML_MAAP_benchmark# az ml model list --registry-name AzureML --query "[?contains(name, 'Phi-3')]" --output table
Name                        Description    Latest version
--------------------------  -------------  ----------------
Phi-3.5-vision-instruct                    2
Phi-3.5-mini-instruct                      6
Phi-3.5-MoE-instruct                       5
Phi-3-vision-128k-instruct                 2
Phi-3-small-8k-instruct                    5
Phi-3-small-128k-instruct                  5
Phi-3-medium-4k-instruct                   6
Phi-3-medium-128k-instruct                 7
Phi-3-mini-4k-instruct                     15
Phi-3-mini-128k-instruct                   13
```

To create a model deployment using a program, you need to specify the model name, subscription ID, resource group name, VM SKU, and the number of VMs.

Check usage first

```
# python deploy_infra.py
2025-02-24 21:39:20,774 - ERROR - Usage: python deploy_infra.py <model_name> <model_version> <subscription_id> <resource_group> <workspace_name> <instance_type> <instance_count>
2025-02-24 21:39:20,775 - ERROR - instance_type options: Standard_NC24ads_A100_v4, Standard_NC48ads_A100_v4, Standard_NC96ads_A100_v4, Standard_NC40ads_H100_v5, Standard_NC80ads_H100_v5
```

Next, deploy the "Phi-3-medium-4k-instruct" deployment using the VM SKU "Standard_NC24ads_A100_v4" with a quantity of 1.

```
# python deploy_infra.py "Phi-3-medium-4k-instruct" "6" "08f95cfd-64fe-4187-99bb-7b3e661c4cde" "rg-admin-2776_ai" "admin-0046" "Standard_NC24ads_A100_v4" 1

```

View the source code of the program.:

```

(aml_env) root@davidwei:~/AML_MAAP_benchmark# cat deploy_infra.py
import os
import sys
import time
import json
import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

# Setting up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 可用的实例类型
INSTANCE_TYPES = [
    "Standard_NC24ads_A100_v4",
    "Standard_NC48ads_A100_v4",
    "Standard_NC96ads_A100_v4",
    "Standard_NC40ads_H100_v5",
    "Standard_NC80ads_H100_v5"
]

if len(sys.argv) != 8:
    logger.error(f"Usage: python {sys.argv[0]} <model_name> <model_version> <subscription_id> <resource_group> <workspace_name> <instance_type> <instance_count>")
    logger.error(f"instance_type options: {', '.join(INSTANCE_TYPES)}")
    sys.exit(1)

model_name = sys.argv[1]           # "Phi-3-medium-4k-instruct"，eg
model_version = sys.argv[2]        # "6", eg.
subscription_id = sys.argv[3]
resource_group = sys.argv[4]
workspace_name = sys.argv[5]
instance_type = sys.argv[6]
instance_count = int(sys.argv[7])

if instance_type not in INSTANCE_TYPES:
    logger.error(f"Invalid instance_type '{instance_type}'. Choose from: {', '.join(INSTANCE_TYPES)}")
    sys.exit(1)

# 构建 model_id
model_id = f"azureml://registries/AzureML/models/{model_name}/versions/{model_version}"

# 认证并创建 MLClient
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# 创建唯一的在线终结点
endpoint_name = f"custom-endpoint-{int(time.time())}"
endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key", description="Custom Model Endpoint")
logger.info(f"Creating endpoint: {endpoint_name}")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 部署模型
deployment_name = "custom-deployment"
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=model_id,
    instance_type=instance_type,
    instance_count=instance_count,
)
logger.info(f"Deploying model {model_name} at endpoint {endpoint_name}")

ml_client.online_deployments.begin_create_or_update(deployment).result()

# 路由流量
endpoint.traffic = {deployment_name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 获取终结点详细信息
endpoint = ml_client.online_endpoints.get(endpoint_name)
scoring_uri = endpoint.scoring_uri
keys = ml_client.online_endpoints.list_keys(endpoint_name)
primary_key = keys.primary_key

logger.info(f"Endpoint {endpoint_name} is deployed successfully.")
logger.info(f"Scoring URI: {scoring_uri}")
logger.info(f"Primary Key: {primary_key}")

# 提供调用终结点的示例代码
logger.info("You can invoke the endpoint using the following example code:")
example_code = f'''
import requests

headers = {{
    "Authorization": "Bearer {primary_key}",
    "Content-Type": "application/json"
}}

data = {{
    "input_data": {{
        "input_string": [
            {{
                "role": "user",
                "content": "Your prompt here"
            }}
        ],
        "parameters": {{
            "max_new_tokens": 50
        }}
    }}
}}

response = requests.post("{scoring_uri}", headers=headers, json=data)
print(response.json())
'''

logger.info(example_code)

logger.info("Deployment completed.")
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

