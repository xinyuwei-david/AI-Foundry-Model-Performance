# AI Foundry performance

This repository is designed to test the performance of open-source models from the Azure AI Foundry Model Catalog when deployed using Managed Compute across different VBM SKUs. The scripts in this repository consist of two parts: deploying the Managed Compute endpoint and testing the performance of the Managed Compute endpoint. After completing the tests, you should delete it promptly to avoid incurring additional costs.

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



```
# python deploy_infra.py "Phi-3-medium-4k-instruct" "6" "08f95cfd-64fe-4187-99bb-7b3e661c4cde" "rg-admin-2776_ai" "admin-0046" "Standard_NC24ads_A100_v4" 1

```

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

# 设置日志记录
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

model_name = sys.argv[1]           # 例如："Phi-3-medium-4k-instruct"
model_version = sys.argv[2]        # 例如："6"
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



```
python concurrency_test.py --endpoint_url "https://xinyuwei-9556-jyhjv.westeurope.inference.ml.azure.com/score" --api_key "A2ZdX5yDwbu11ZYKeuznMqoU69GHyRZvU7IbaDPZDkmYH2J1Ia6VJQQJ99BBAAAAAAAAAAAAINFRAZML5E10" --initial_concurrency 1 --prompt_sizes 64 128 1024 2048 4096 --response_sizes 64 128 1024 2048 4096  --max_tests 100 --output_file "results.csv" --max_concurrency 10


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



```
(aml_env) root@davidwei:~/AML_MAAP_benchmark# cat concurrency_test_final1.py
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
    parser.add_argument('--max_concurrency', type=int, default=50, help="Maximum concurrency level to test")  # 新增的参数
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
        max_concurrency=args.max_concurrency  # 传入最大并发参数
    )
```

