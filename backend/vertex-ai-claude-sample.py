# TODO(developer): Vertex AI SDK - uncomment below & run
# pip3 install --upgrade --user google-cloud-aiplatform
# gcloud auth application-default login
# pip3 install -U 'anthropic[vertex]'

# TODO(developer): Update and un-comment below line
PROJECT_ID = ""

from anthropic import AnthropicVertex

client = AnthropicVertex(project_id=PROJECT_ID, region="us-east5")
message = client.messages.create(
    model="claude-3-5-sonnet-v2@20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Send me a recipe for banana bread.",
        }
    ],
)
print(message.model_dump_json(indent=2))
# Example response:
# {
#   "id": "msg_vrtx_0162rhgehxa9rvJM5BSVLZ9j",
#   "content": [
#     {
#       "text": "Here's a simple recipe for delicious banana bread:\n\nIngredients:\n- 2-3 ripe bananas...
#   ...