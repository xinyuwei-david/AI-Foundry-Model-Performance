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
