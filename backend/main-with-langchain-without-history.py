from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from google.cloud import bigquery
import os
from dotenv import load_dotenv
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware

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
  available_visualizations: List[VisualizationOption]
  has_visualizations: bool
  raw_data: Union[Dict[str, Any], List[Dict[str, Any]]]
  sql: str = None

class FinOpsAnalyzer:
    def __init__(self,data_model: str, ai_model: str):
        self.data_models  = {
            "cloud_forecast": {
                "prompt": """
                You are a SQL expert. Given the following question about FinOps data, generate a BigQuery SQL query.

                Question: {question}

                Table: `wpp-it-fo-platform-dev-sbx.test_jithin.cloud_forecast_sample`

                Important notes:
                - Monthly forecast columns:
                - Each month is a separate column: jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
                - DO NOT use CASE statements with month names
                - DO NOT treat month as a separate column
                - Each month column contains forecast value for that month
                - If monthly cost is asked, mention that monthly actuals are not available only forecast cost is available.
                - For actual costs:
                - last_30day_cost: Contains actual billing cost for last 30 days
                - actual_ytd_cost: Contains actual year-to-date cost
                - For monthly data:
                - jan through dec columns contain FORECAST data entered by users, NOT actuals
                - These are future predictions/forecasts only
                - Monthly actuals are not available. Only monthly forecasts given by users are available.
                - The table contains forecast costs for cloud services per month.
                - Only actuals available are for the actual_ytd_cost and last_30day_cost.
                - if asked data for this year, use EXTRACT(YEAR FROM CURRENT_DATE()) function to get the current year.

                Important columns:
                - last_30day_cost: Contains actual billing cost for last 30 days from todays date for each year
                - actual_ytd_cost: Year-to-date cost
                - jan through dec (monthly columns): forecast data entered by users
                - year: INTEGER
                - cloud_vendor: STRING
                - fo_initiative_group: STRING


                return ONLY the SQL query. Dont return anything else. Just the BQ SQL query that can be executed directly.
                """
            },
            "recharge_azure": {
                "prompt": """
                You are a SQL expert. Given the following question about FinOps data, generate a BigQuery SQL query on top of this azure cloud billing data.

                Question: {question}

                Table: wpp-it-fo-platform-dev-sbx.recharge_analytics.fct_azure_recharge_breakdown

                Important notes:
                    - table has azure billing data. It also has business mapping data that is used to map the billing data to the business.
                    - use azure_date for all dates
                
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
                
                    return ONLY the SQL query. Dont return anything else. Just the BQ SQL query that can be executed directly.
                """
            }
        }

        if data_model not in self.data_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_model. Available options: {list(self.data_models.keys())}"
            )

        self.current_model = self.data_models[data_model]
        
        if ai_model == "gemini":
            # Initialize Vertex AI with Gemini
            self.llm = ChatVertexAI(
                model_name="gemini-pro",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_REGION', 'europe-west2'),
                temperature=0.1
            )
        elif ai_model == "claude":
            # Initialize Vertex AI with Claude 
            self.llm = ChatAnthropicVertex(
                model_name="claude-3-5-sonnet-v2@20241022",
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location='us-east5',
                temperature=0.1
            )
        
        
        self.bq_client = bigquery.Client()

        self.prompt = PromptTemplate(
            template=self.current_model["prompt"],
            input_variables=["question"]
        )
   

    async def generate_sql(self, question: str) -> str:
        try:
            # Create completion with prompt
            response = await self.llm.ainvoke(
                self.prompt.format(question=question)
            )
            sql = response.content.strip()

            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # Remove any lines that start with comments or explanations
            sql_lines = [
                line.strip() 
                for line in sql.split('\n') 
                if line.strip() and 
                not line.strip().startswith('--') and 
                not line.strip().startswith('/*') and 
                not line.strip().startswith('*') and 
                not line.strip().startswith('**')
            ]
            
            # Rejoin the lines
            sql = '\n'.join(sql_lines)
            print(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            print(f"Error generating SQL: {e}")
            raise

    def generate_narrative(self, question: str, data: pd.DataFrame) -> Dict:
        try:
            if data.empty:
                return NarrativeResponse(
                    title="No Data Found",
                    summary="No data found for the given query.",
                    key_points=["No records found for the specified criteria."]
                )

            prompt = f"""
            You are a financial analyst. Based on the following data, provide analysis in the exact JSON structure shown below.
            Do not include markdown formatting, code blocks, or any additional text.

            Return only this JSON structure, nothing else:
            {{
                "title": "Brief title",
                "summary": "One clear sentence about the data",
                "key_points": [
                    "Point 1 about the cost",
                    "Point 2 about the implications",
                    "Point 3 about recommendations"
                ]
            }}

            Question: {question}
            Data: {data.to_string()}

            Return only the JSON, nothing else.
            """
            
            response = self.llm.invoke(prompt)
            
            # Clean up the response
            cleaned_response = response.content.strip()
            cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            
            return json.loads(cleaned_response)

        except Exception as e:
            return NarrativeResponse(
                title="Error in Analysis",
                summary="Error generating analysis",
                key_points=[str(e)]
            )

    def check_visualization_potential(self, df: pd.DataFrame) -> List[VisualizationOption]:
        visualizations = []
        
        if not df.empty:
            # For single total value, don't suggest pie chart
            if len(df.columns) == 1:
                return []  # No visualizations for single total

            # For time series data (monthly data)
            if any(col in df.columns for col in ['jan', 'feb', 'mar']):
                visualizations.append(VisualizationOption(
                    type="line",
                    label="Monthly Trend",
                    supported=True
                ))
            
            # For comparisons
            if len(df) > 1:
                visualizations.append(VisualizationOption(
                    type="bar",
                    label="Comparison",
                    supported=True
                ))
            
            # For distribution
            if len(df) <= 10:
                visualizations.append(VisualizationOption(
                    type="pie",
                    label="Distribution",
                    supported=True
                ))
                
        return visualizations

    async def analyze(self, question: str) -> ResponseModel:
        try:
            # Generate and execute SQL
            sql = await self.generate_sql(question)
            df = self.bq_client.query(sql).to_dataframe()
            print(f"\nQuery executed, got {len(df)} rows")
            # print("df: ", df)
            
            # Generate narrative
            narrative = self.generate_narrative(question, df)
            print(f"\nNarrative generated: {narrative}")
            
            # Check visualization options
            visualizations = self.check_visualization_potential(df)
            
            return ResponseModel(
                narrative=narrative,
                available_visualizations=visualizations,
                has_visualizations=len(visualizations) > 0,
                raw_data=df.to_dict(orient='records'),
                sql=sql
            )
            
        except Exception as e:
            print(f"Error in analyze: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during analysis: {str(e)}"
            )

@app.post("/analyze")
async def analyze_question(question: str, data_model: str = "cloud_forecast", ai_model: str = "gemini-pro"):
    analyzer = FinOpsAnalyzer(data_model, ai_model)
    return await analyzer.analyze(question)

@app.post("/generate_chart")
async def generate_chart(question: str, chart_type: str):
    analyzer = FinOpsAnalyzer()
    response = await analyzer.analyze(question)

    if not response.has_visualizations:
        raise HTTPException(
            status_code=400,
            detail="No visualization available for this data"
        )

    if not any(v.type == chart_type and v.supported 
            for v in response.available_visualizations):
        raise HTTPException(
            status_code=400,
            detail=f"Chart type {chart_type} not supported for this data"
        )

    return {
        "chart_type": chart_type,
        "data": response.raw_data,
        "config": {
            "title": f"{chart_type.capitalize()} chart for {question}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)