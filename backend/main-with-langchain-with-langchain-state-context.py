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
    narrative: NarrativeResponse
    sql: str = None


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
                "table_name": "`wpp-it-fo-platform-dev-sbx.test_jithin.cloud_forecast_sample`",
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
                "table_name": "`wpp-it-fo-platform-dev-sbx.recharge_analytics.fct_azure_recharge_breakdown`",
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
                    cost_category		STRING	WPP custom logic to identify cost by category (Reservation, SavingsPlan, Amortized, Marketplace, and Azure Subcriptions).
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

        # Initialize StateGraph
        self.workflow = StateGraph(state_schema=FinOpsState)

        # Add memory
        self.memory = MemorySaver()

        # Setup workflow
        self.setup_workflow()

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

    def get_latest_question(self, messages):
        # Get the last human message
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return None

    # Define SQL generation function
    def generate_sql(self, state: FinOpsState):
        print("\n=== Generating SQL ===")
        # Get current data model config
        model_config = self.current_model

        # Create system message with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
        You are a SQL expert. Generate a BigQuery SQL query.

        Table: {model_config['table_name']}

        Schema:
        {model_config['schema']}

        Context:
        {model_config['context']}

        If necessary refer conversation context from previous messages:

        Rules:
        - Return ONLY the SQL query. Not even backquotes like ```sql```.
        - Use standard SQL syntax
        - Use only columns mentioned in schema
        - Don't include comments or explanations
        """),
            MessagesPlaceholder(variable_name="messages"),
        ])

        runnable = prompt | self.llm

        # Generate SQL
        response = runnable.invoke(state)

        # Update message history with response:
        # sql = {"sql": [response]}
        print("SQL generated:", response.content)

        sql_query = response.content

        # sql_query = self._clean_sql(sql_query)
        df = self.bq_client.query(sql_query).to_dataframe()
        print(f"\nQuery executed, got {len(df)} rows")

        return {
            "messages": [response],
            "sql": sql_query,
            "sql_result": df.to_string(),  # Convert DataFrame to string
        }

    def generate_narrative(self, state: FinOpsState) -> Dict:
        print("\n=== Generating Narrative ===")
        try:
            current_question = self.get_latest_question(state["messages"])
            print("=====STATE=====", state)

            # if state["sql_result"]:
            #     print("No data found")
            #     return {
            #         "title": "No Data Found",
            #         "summary": "No data found for the given query.",
            #         "key_points": ["No records found for the specified criteria."]
            #     }

            # Create system message with context
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
            You are a financial analyst. Based on the following data and conversation context,
            provide analysis in the exact JSON structure shown below.


            Current Question: {current_question}
            Data: {state["sql_result"]}

            Return only this JSON structure, nothing else:
            The result returned be just a json, dont add anything extra like text or comments.
            JSON response object will have 3 keys,
                1. title: Brief title
                2. summary: One clear sentence about the data
                3. key_point: Array containing points
            """),
                MessagesPlaceholder(variable_name="messages"),
            ])

            runnable = prompt | self.llm

            # Generate SQL
            response = runnable.invoke(state)

            print("\nNarrative generated:", response.content)

            return {
                "messages": [response],
            }

        except Exception as e:
            print(f"\n!!! Error in generate_narrative: {e}")
            return {
                "title": "Error in Analysis",
                "summary": "Error generating analysis",
                "key_points": [str(e)]
            }

    def setup_workflow(self):
        # Add nodes
        self.workflow.add_node("generate_sql", self.generate_sql)
        self.workflow.add_node("generate_narrative", self.generate_narrative)

        # Add edges for flow: START -> generate -> generate_narrative
        self.workflow.add_edge(START, "generate_sql")
        self.workflow.add_edge("generate_sql", "generate_narrative")

    async def process_query(self, question: str, thread_id: str = "default") -> ResponseModel:
        try:
            self.app = self.workflow.compile(checkpointer=self.memory)

            config = {"configurable": {"thread_id": thread_id}}

            # # Print history before processing new question
            # print("\n=== Current Conversation History ===")
            # try:
            #     # Using app.get_state similar to docs
            #     current_state = self.app.get_state(self.config).values
            #     print("current_state:", current_state)
            #     if current_state and "messages" in current_state:
            #         print("Previous messages:")
            #         for message in current_state["messages"]:
            #             message.pretty_print()
            #         print("---")
            #     else:
            #         print("No previous history")
            # except Exception as e:
            #     print(f"Error loading history: {e}")
            #     print("No previous history")

            input_state = {
                "messages": [HumanMessage(question)],
                "sql": "",
                "sql_result": "",
            }
            # Run workflow
            output = self.app.invoke(input_state, self.config)

            print("output:\n", output)

            print("\noutput[messages][-1]", json.loads(output["messages"][-1].content))

            # Execute SQL
            # df = self.bq_client.query(sql).to_dataframe()
            # print(f"\nQuery executed, got {len(df)} rows")

            # Generate narrative
            # narrative = self.generate_narrative(question, df)

            return ResponseModel(
                narrative=json.loads(output["messages"][-1].content),
                sql=output["sql"]
            )

        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during process_query: {str(e)}"
            )


analyzer = FinOpsAnalyzer(data_model="recharge_azure", ai_model="claude")

@app.post("/analyze")
async def analyze_question(question: str, data_model: str = "recharge_azure", ai_model: str = "claude"):
    
    # To reinitialize when user changes data model or AI model
    global analyzer
    if analyzer.data_model != data_model or analyzer.ai_model != ai_model:
        analyzer = FinOpsAnalyzer(data_model=data_model, ai_model=ai_model)

    return await analyzer.process_query(question)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
