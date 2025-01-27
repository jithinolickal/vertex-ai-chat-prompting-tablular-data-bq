from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from google.cloud import bigquery
import os
from dotenv import load_dotenv
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware


from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate

import os
from typing import Sequence
from typing_extensions import Annotated, TypedDict


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VisualizationOption(BaseModel):
    type: str
    label: str
    supported: bool


class NarrativeResponse(BaseModel):
    title: str
    summary: str
    key_points: List[str]


class ResponseModel(BaseModel):
    narrative: NarrativeResponse = None
    sql: str = None
    chart_code: str = None
    response_type: str = None


class FinOpsState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sql: str
    sql_result: str


class FinOpsAnalyzer:
    def __init__(self, data_model: str, ai_model: str):
        self.data_model = data_model
        self.ai_model = ai_model

        self.bq_client = bigquery.Client()

        self.config = {"configurable": {"thread_id": "abc123"}}
        # Data model configurations
        self.data_models = {
            "cloud_forecast": {
                "table_name": "``",
                "schema": """
             Important columns:
             - last_30day_cost: FLOAT   actual billing cost for last 30 days
             - actual_ytd_cost: FLOAT   year-to-date cost
             - jan through dec: FLOAT   forecast data entered by users
             - year: INTEGER
             - cloud_vendor: STRING
             - fo_initiative_group: STRING
             """,
                "context": """
             Important notes:
             - Monthly forecast columns (jan-dec) contain FORECAST data, not actuals
             - last_30day_cost contains actual billing for last 30 days
             - actual_ytd_cost contains year-to-date actual cost
             - Monthly actuals are not available
             """
            },
            "recharge_azure": {
                "table_name": "``",
                "schema": """
             Columns:
                    uuid		STRING	Generated uuid with azure date.
                    subs_res_group_id		STRING	Hash of subscription id and resource_group
                    billing_subscription_id		STRING	Unique identifier for the Azure subscription.
                    billing_subscription_name		STRING	Name of the Azure subscription.
                    billing_resource_group		STRING	Name of the resource group the resource is in.
                    resource_location		STRING	Datacenter location where the resource is running.
                    azure_date		DATE	The usage or purchase date of the charge.
                    department_name		STRING	Name of the EA department or MCA invoice section.
                    pricing_model		STRING	Identifier that indicates how the meter is priced (On Demand, Reservation, and Spot).
                    charge_type		STRING	Indicates whether the charge represents usage, a purchase, or a refund.
                    publisher_type		STRING	Type of publisher (Azure, AWS, Marketplace).
                    publisher_name		STRING	Publisher for Marketplace services.
                    plan_name		STRING	Marketplace plan name.
                    additional_info		STRING	Service-specific metadata. For example, an image type for a virtual machine.
                    reservation_name		STRING	Name of the purchased reservation instance.
                    product_order_id		STRING	Unique identifier for the product order.
                    product_order_name		STRING	Unique name for the product order.
                    frequency		STRING	Indicates whether a charge is expected to repeat. Charges can either happen once (OneTime), repeat on a monthly or yearly basis (Recurring), or be based on usage (UsageBased).
                    quantity		FLOAT	The number of units purchased or consumed.
                    cost_center		STRING	The cost center defined for the subscription for tracking costs.
                    account_name		STRING	Display name of the EA enrollment account or pay-as-you-go billing account.
                    account_owner_id		STRING	Unique identifier for the EA enrollment account or pay-as-you-go billing account.
                    product_name		STRING	Name of the product.
                    resource_id		STRING	Unique identifier of the Azure Resource Manager resource.
                    resource_name		STRING	Name of the resource.
                    benefit_id		STRING	Unique identifier for the benefit.
                    benefit_name		STRING	Name of the benefit.
                    tags		STRING	Tags assigned to the resource. Doesn't include resource group tags. Can be used to group or distribute costs for internal chargeback.
                    bm_cartesis_code		STRING	An identifier for the country and entity being invoiced/billed/charged for Subscription account.
                    bill_to		STRING	An identifier for the relevant Op Company/Agency managing the recharge.
                    cost_category		STRING	 custom logic to identify cost by category (Reservation, SavingsPlan, Amortized, Marketplace, and Azure Subcriptions).
                    list_price		FLOAT	On Demand price for the resource with EDP
                    resource_ri_sp_consumed		FLOAT	Actual Resveration / SavingsPlan consumption at resource level, only displays for Amortized cost category
                    resource_ri_sp_wastage		FLOAT	For RI/SP purchased centrally - central adjustment portion assigned back to resource level based on usage ratio proportionally, Otherwise unused component charged to RI/SP owner itself. Only displays for Amortized cost category
                    total_cost		FLOAT	Cost of the charge in the billing currency before credits or taxes also including ri_sp_recharge.
                    ri_sp_savings		FLOAT	Savings made using RI/SP compared with On-Demand list price, only displays for Amortized cost category
                    period		STRING	Period of the usage or purchase date of the charge.
                    inserted_at		TIMESTAMP	The time of record insertion.
             """,
                "context": """
             Important notes:
                    - table has azure billing data. It also has business mapping data that is used to map the billing data to the business.
                    - use azure_date for all dates

                    return ONLY the SQL query. Dont return anything else. Just the BQ SQL query that can be executed directly.
             """
            },
            "cloud_consumption": {
                "table_name": "``",
                "schema": """
             Columns:
                    subscription_id		STRING	Cloud account ID - AWS Account ID / GCP Project ID / Azure Subscription ID
                    subscription_name		STRING	Cloud account name - AWS Account name / GCP Project name / Azure Subscription name
                    master_account_id		STRING	Cloud master payer account / billing account ID
                    master_account_name		STRING	Cloud master payer account / billing account name
                    fo_department		STRING	[Business Mapping] Department
                    fo_maconomy_job_code_name		STRING	[Business Mapping] Maconomy Job code name
                    fo_maconomy_job_code		STRING	[Business Mapping] Maconomy Job code number
                    fo_maconomy_department		STRING	[Business Mapping] Maconomy Department
                    cloud_vendor		STRING	Cloud Vendor (e.g. aws, gcp, azure, ibm)
                    fo_msp		STRING	[Business Mapping] Managed Service Provider
                    fo_opgroup		STRING	[Business Mapping] OpGroup
                    fo_opco		STRING	[Business Mapping] OpCo
                    fo_bill_to		STRING	[Business Mapping] Bill To
                    fo_archetype		STRING	[Business Mapping] Archetype
                    fo_agency		STRING	[Business Mapping] Agency
                    fo_initiative		STRING	[Business Mapping] Initiative
                    fo_initiative_group		STRING	
                    fo_financial_owner_master_account		STRING	[Business Mapping] Financial Owner of Master Account
                    fo_financial_owner_subscription		STRING	[Business Mapping] Financial Owner of Subscription
                    fo_technical_owner_organisation		STRING	[Business Mapping] Technical Owner of Cloud Organisation
                    fo_technical_owner_subscription		STRING	[Business Mapping] Technical Owner of Subscription
                    fo_business_owner		STRING	[Business Mapping] Business Owner
                    fo_environment		STRING	[Business Mapping] Environment
                    fo_region		STRING	[Business Mapping] Region
                    fo_city		STRING	[Business Mapping] City
                    fo_application_name		STRING	[Business Mapping] Application Name
                    fo_product		STRING	[Business Mapping] Product
                    fo_suite		STRING	[Business Mapping] Suite
                    azure_account		STRING	Azure Account name from Azure Billing
                    azure_cost_center		STRING	Azure cost center from Azure Billing
                    azure_department		STRING	Azure department from Azure Billing
                    fo_notes		STRING	[Business Mapping] Notes
                    fo_budget		FLOAT	[Business Mapping] Budget
                    fo_cartesis_code		STRING	[Business Mapping] Cartesis Code
                    cost		FLOAT	Cloud resource cost (2 d.p)
                    invoice_date		DATE	Invoice date in YYYY-MM-DD format
                    service_name		STRING	Cloud Service name e.g. AWS EC2
                    product_name		STRING	Cloud product / Marketplace product name
                    sku_name		STRING	
                    seller		STRING	Marketplace seller name
                    is_marketplace		STRING	Specifies whether the cost is incurred from a Marketplace purchase [yes / No]
                    payment_type		STRING	Payment type [Own Use / Recharged to OpCo / Paid by OpCo]
             """,
                "context": """
             Important notes:
                    - table has cloud data for azure, gcp and aws (cloud vendors) such as project/subscription and their mapping data.
                        It also has business mapping data that is used to map the billing data to the business.
                    - use invoice_date for all dates

                    return ONLY the SQL query. Dont return anything else. Just the BQ SQL query that can be executed directly.
             """
            }
        }

        # Initialize LLM
        if self.ai_model == "gemini":
            self.llm = ChatVertexAI(
                model_name="gemini-pro",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_REGION', 'europe-west2'),
                temperature=0.1
            )
        elif self.ai_model == "claude":
            self.llm = ChatAnthropicVertex(
                model_name="claude-3-5-sonnet-v2@20241022",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location='us-east5',
                temperature=0.1
            )

        self.current_model = self.data_models[self.data_model]

        # Initialize state
        self.state = {
            "messages": [
                # {
                #     "type": "human",
                #     "step": "question",
                #     "question": "What is the total cost for last month?",
                # }, {
                #     "type": "ai",
                #     "step": "generate_sql",
                #     "summary": "",
                #     "sql": "",
                #     "sql_result": ""
                # },
                # {
                #     "type": "ai",
                #     "step": "generate_narrative",
                #     "summary": "",
                #     "sql": "",
                #     "sql_result": ""
                # }
            ],
        }

    def _clean_sql(self, sql_text: str) -> str:
        # Remove markdown and comments
        sql = sql_text.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()

        # Remove comment lines
        sql_lines = [
            line.strip()
            for line in sql.split('\n')
            if line.strip() and
            not line.strip().startswith('--') and
            not line.strip().startswith('/*') and
            not line.strip().startswith('*') and
            not line.strip().startswith('**')
        ]

        return '\n'.join(sql_lines)

    def get_latest_message(self, messages, prompt_type):
        # Get the latest message
        for message in reversed(messages):
            if message.get("type") == prompt_type:
                return message
        return None

    async def prompt_analyzer(self, input):
        print("\n=== Analyzing Prompt ===")

        prompt = PromptTemplate.from_template("""
              Given the user's question, determine the type of response needed.
    
                Question: {input}
                
                Select ONE of these options:
                - sql_query: If user needs new data, not asking for visualization
                - chart: If user wants some visualization
                
                Output your response as: {{"prompt_response_type": "selected_option"}}                                
        """)
        response = await self.llm.ainvoke(prompt.format(input=input))

        print("response generated:", response)

        response = json.loads(response.content)

        return response

    # Define SQL generation function
    async def generate_sql(self, input):
        try:
            print("\n=== Generating SQL ===")
            # Get current data model config
            model_config = self.current_model

            # Create system message with context
            prompt = PromptTemplate.from_template("""
            You are a SQL expert. Given the following question about FinOps data, generate a BigQuery SQL query.

            Question: {question}

            Table: {table_name}

            Schema:
            {schema}

            Context:
            {context}
                                                  
            previous messages: {previous_messages}

            If necessary refer conversation context from previous messages:

            Rules:
            - Dont make the sql query that retunrs many rows. Make sure 12 rows is the maximum rows returnd. minimum can vary based on the question.  
            - Return ONLY the SQL query. Not even backquotes like ```sql```.
            - Use standard SQL syntax
            - Use only columns mentioned in schema
            - Don't include comments or explanations
            """,
                                                  )

            response = await self.llm.ainvoke(
                prompt.format(question=input["question"],
                              table_name=model_config["table_name"],
                              schema=model_config["schema"],
                              context=model_config["context"],
                              previous_messages=self.state["messages"]
                              ))

            # Update message history with response:
            # sql = {"sql": [response]}
            print("response generated:", response)
            print("SQL generated:", response.content)

            sql_query = response.content

            # sql_query = self._clean_sql(sql_query)
            df = self.bq_client.query(sql_query).to_dataframe()
            print(f"\nQuery executed, got {len(df)} rows")

            # append to self.state messages list with ai message
            self.state["messages"].append({
                "type": "ai",
                "step": "generate_sql",
                "summary": "",
                "sql": sql_query,
                "sql_result": df.to_string()
            })

            return sql_query
        except Exception as e:
            print(f"Error generating SQL: {e}")
            raise

    async def generate_narrative(self, input):
        print("\n=== Generating Narrative ===")
        try:
            current_question = input["question"]
            latest_ai_message = self.get_latest_message(
                self.state["messages"], "ai")
            print("=====current_question=====", current_question)
            print("=====latest_message=====", latest_ai_message)

            # Create system message with context
            prompt = PromptTemplate.from_template("""
            You are a financial analyst. Based on the following data and conversation context,
            provide analysis in the exact JSON structure shown below.


            Current Question: {current_question}
            Data: {sql_result}

            Return only this JSON structure, nothing else:
            The result returned be just a json, dont add anything extra like text or comments.
            JSON response object will have 3 keys,
                1. title: Brief title
                2. summary: One clear sentence about the data
                3. key_points: Array containing points
            """)

            response = await self.llm.ainvoke(
                prompt.format(current_question=current_question,
                              sql_result={latest_ai_message["sql_result"]}
                              ))

            print("\nNarrative generated:", response)
            response = json.loads(response.content)
            print("\nNarrative generated:", response)

            self.state["messages"].append({
                "type": "ai",
                "step": "generate_narrative",
                "summary": response["summary"],
            })

            return response

        except Exception as e:
            print(f"\n!!! Error in generate_narrative: {e}")
            return {
                "title": "Error in Analysis",
                "summary": "Error generating analysis",
                "key_points": [str(e)]
            }

    async def generate_chart(self, input):
        print("\n=== Generating Chart ===")
        try:
            current_question = input["question"]
            latest_ai_message = self.get_latest_message(
                self.state["messages"], "ai")
            
            print("=====current_question=====", current_question)
            print("=====latest_message=====", latest_ai_message)

            # Create system message with context
            prompt = PromptTemplate.from_template("""
            Generate simple React code using Recharts library based on the data and question.
            Give me without markdown formatting. I want to copy the code directly into my React app or Sandpack and run it.
            DONT use backticks or markdown formatting. i need to put this directly to sandpack-react component in my code as a string.
            use <ResponsiveContainer width="100%" height={{300}}> for the chart container.


            Question: {current_question}
            Data: {sql_result}

            Return only react code, nothing else
            """)

            print("=======", prompt.format(current_question=current_question,
                              sql_result={latest_ai_message["sql_result"]}
                              ))

            response = await self.llm.ainvoke(
                prompt.format(current_question=current_question,
                              sql_result={latest_ai_message["sql_result"]}
                              ))

            print("\nChart generated:", response.content)

            self.state["messages"].append({
                "type": "ai",
                "step": "generate_chart",
                "summary": response.content,
            })

            return response.content

        except Exception as e:
            print(f"\n!!! Error in generate_chart: {e}")
            return {
                "title": "Error in Analysis",
                "summary": "Error generating analysis",
                "key_points": [str(e)]
            }

    async def process_query(self, question: str, thread_id: str = "default") -> ResponseModel:
        try:

            prompt_response_type = await self.prompt_analyzer(question)
            print("prompt_response_type:", prompt_response_type)
            input_state = {
                "type": "human",
                "step": "question",
                "question": question,
            }
            self.state["messages"].append(input_state)

            if prompt_response_type.get("prompt_response_type") == "sql_query":


                sql = await self.generate_sql(input_state)

                print("output:\n", sql)

                print("===state:\n", json.dumps(self.state, indent=2))

                # Generate narrative
                narrative = await self.generate_narrative(input_state)

                return ResponseModel(
                    narrative=narrative,
                    sql=sql,
                    response_type="narrative",
                )
            
            if prompt_response_type.get("prompt_response_type") == "chart":
                sql = await self.generate_sql(input_state)
                chart_code = await self.generate_chart(input_state)

                return ResponseModel(
                    chart_code=chart_code,
                    sql=sql,
                    response_type="chart",
                )


        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during process_query: {str(e)}"
            )


analyzer = FinOpsAnalyzer(data_model="cloud_consumption", ai_model="claude")


@app.post("/analyze")
async def analyze_question(question: str, data_model: str = "cloud_consumption", ai_model: str = "claude"):

    # To reinitialize when user changes data model or AI model
    global analyzer
    if analyzer.data_model != data_model or analyzer.ai_model != ai_model:
        analyzer = FinOpsAnalyzer(data_model=data_model, ai_model=ai_model)

    return await analyzer.process_query(question)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
