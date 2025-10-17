import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import ListSortOrder

os.environ["AZURE_CLIENT_ID"] = 
os.environ["AZURE_TENANT_ID"] = 
CERT_PATH_RELATIVE = "./final_cert_for_azure.pem"
os.environ["AZURE_CLIENT_CERTIFICATE_PATH"] = os.path.abspath(CERT_PATH_RELATIVE)

ENDPOINT = 
AGENT_ID = 

credential = DefaultAzureCredential()
project = AIProjectClient(
    credential=credential,
    endpoint=ENDPOINT
)

thread = project.agents.threads.create()
print(f"✅ Thread 생성 완료, ID: {thread.id}")

message = project.agents.messages.create(
    thread_id=thread.id,
    role="user",
    content="안녕하세요"
)
print(f"✅ 메시지 전송 완료, ID: {message.id}")

run = project.agents.runs.create_and_process(
    thread_id=thread.id,
    agent_id=AGENT_ID
)

if run.status == "failed":
    print(f"❌ Run 실패: {run.last_error}")
else:
    print(f"✅ Run 완료, 상태: {run.status}")

    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    for msg in messages:
        if msg.text_messages:
            print(f"{msg.role}: {msg.text_messages[-1].text.value}")
