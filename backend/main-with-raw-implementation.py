from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel
import vertexai  # Add this import
import pandas as pd
import json

app = FastAPI()


class VisualizationOption(BaseModel):
    type: str
    label: str
    supported: bool


class ResponseModel(BaseModel):
    narrative: Dict[str, Any]
    available_visualizations: List[VisualizationOption]
    has_visualizations: bool
    raw_data: Union[Dict[str, Any], List[Dict[str, Any]]]  # Modified to handle both empty and non-empty cases


class FinOpsAnalyzer:
    def __init__(self):
        print("Loading model and initializing BigQuery client...")
        # Initialize Vertex AI with region
        vertexai.init(project='',
                      location='europe-west2')
        self.model = GenerativeModel("gemini-pro")
        self.bq_client = bigquery.Client()

        # You can modify this based on your actual table
        self.table_schema = """
     Table: 
     Columns:
        - key		STRING
        - year		INTEGER
        - fo_initiative_group		STRING  unique identifier for the initiative group
        - fo_bill_to		STRING
        - fo_opco		STRING
        - fo_cartesis		STRING
        - cloud_vendor		STRING
        - forecast_type		STRING
        - fo_departments		STRING
        - last_30day_cost		FLOAT   actual billing cost for the last 30 days of an initiative group
        - actual_ytd_cost		FLOAT   actual billing cost for the year-to-date of an initiative group
        - jan		FLOAT   forecast data entered by users
        - feb		FLOAT   forecast data entered by users
        - mar		FLOAT   forecast data entered by users
        - apr		FLOAT   forecast data entered by users
        - may		FLOAT   forecast data entered by users
        - jun		FLOAT   forecast data entered by users
        - jul		FLOAT   forecast data entered by users
        - aug		FLOAT   forecast data entered by users
        - sep		FLOAT   forecast data entered by users
        - oct		FLOAT   forecast data entered by users
        - nov		FLOAT   forecast data entered by users
        - dec		FLOAT   forecast data entered by users
        - last_updated_at		DATETIME    timestamp when the record was last updated by user for updating month forecast
        - last_updated_by		STRING  user who last updated the record
     """

    def generate_sql(self, question: str) -> str:

        try:
            prompt = f"""
            You are a SQL expert. Generate a BigQuery SQL query based on this schema:
            {self.table_schema}

            Important Column Meanings:
            - last_30day_cost: Already contains the cost for the last 30 days
            - actual_ytd_cost: Contains the year-to-date actual cost
            - jan through dec: Contains monthly costs for each month
            - last_updated_at: Just indicates when the record was last updated (don't use for cost filtering)

            Question: {question}

            Rules:
            - Use only the columns mentioned in the schema
            - Use standard SQL syntax
            - Return ONLY the raw SQL query
            - For last 30 days cost, just use last_30day_cost directly
            - For monthly costs, use the specific month columns (jan, feb, etc.)
            - Don't use last_updated_at for cost filtering

            Do not include:
            - No explanations
            - No comments (--comments or /* comments */)
            - No markdown formatting
            - No notes or additional text
            Just the raw SQL query.

            """
            # Example queries:
            # 1. For last 30 days cost total:
            # SELECT SUM(last_30day_cost) as total_cost
            # FROM ``
            # WHERE year = EXTRACT(YEAR FROM CURRENT_DATE())

            # 2. For specific month's cost in current year:
            # SELECT SUM(aug) as august_cost
            # FROM ``
            # WHERE year = EXTRACT(YEAR FROM CURRENT_DATE())
            
            response = self.model.generate_content(prompt)
            sql = response.text.strip()
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
            
            if not sql or not sql.lower().startswith('select'):
                raise ValueError("Invalid SQL generated")
                
            return sql
            
        except Exception as e:
            print(f"Error in generate_sql: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating SQL: {str(e)}"
            )


    def generate_narrative(self, question: str, data: pd.DataFrame) -> Dict:
        try:
            if data.empty:
                return {
                    "summary": "No data found for the given query.",
                    "key_points": [
                        "No records found for the specified criteria.",
                        "This might be due to no data for the selected time period.",
                        "Try adjusting your query parameters if needed."
                    ]
                }

            prompt = f"""
            Question: {question}
            Data: {data.to_string()}
            
            Provide a clear analysis with:
            1. A brief summary
            2. 3-4 key points about the data
            
            Format as JSON with 'summary' and 'key_points' keys.
            """
            
            response = self.model.generate_content(prompt)
            
            # Clean up the response and ensure it's proper JSON
            cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            try:
                # Try to parse as JSON
                parsed_response = json.loads(cleaned_response)
                return parsed_response
            except json.JSONDecodeError:
                # If JSON parsing fails, return a formatted dict
                return {
                    "summary": cleaned_response,
                    "key_points": []
                }
                
        except Exception as e:
            print(f"Error in generate_narrative: {str(e)}")
            return {
                "summary": "Error generating narrative",
                "key_points": [f"Error: {str(e)}"]
            }

    def check_visualization_potential(self, data: pd.DataFrame) -> List[VisualizationOption]:
        visualizations = []

        # Check for time series potential (line chart)
        if 'date' in data.columns and len(data) > 1:
            visualizations.append(VisualizationOption(
                type="line",
                label="Trend over time",
                supported=True
            ))

        # Check for categorical comparison potential (bar chart)
        if len(data) > 1 and data.select_dtypes(include=['number']).columns.any():
            visualizations.append(VisualizationOption(
                type="bar",
                label="Comparison view",
                supported=True
            ))

        # Check for distribution potential (pie chart)
        if len(data) <= 10 and data.select_dtypes(include=['number']).columns.any():
            visualizations.append(VisualizationOption(
                type="pie",
                label="Distribution view",
                supported=True
            ))

        return visualizations

    def analyze(self, question: str) -> ResponseModel:
        try:
            print(f"Analyzing question: {question}")
            
            print("\n---------------GENERATING SQL----------------")
            # Generate and execute SQL
            sql = self.generate_sql(question)
            
            df = self.bq_client.query(sql).to_dataframe()
            print(f"Query executed, got {len(df)} rows")
            
            # Handle empty results
            if df.empty:
                return ResponseModel(
                    narrative={
                        "summary": "No data found for the given query.",
                        "key_points": [
                            "No records found for the specified criteria.",
                            "This might be due to no data for the selected time period.",
                            "Try adjusting your query parameters if needed."
                        ]
                    },
                    available_visualizations=[],
                    has_visualizations=False,
                    raw_data={}  # Empty dict instead of empty list
                )
            
            print("\n---------------GENERATING NARRATIVE----------------")
            # If we have data, proceed as normal
            narrative = self.generate_narrative(question, df)
            print(f"Narrative generated: {narrative}")

            print("\n---------------CHECKING VISUALIZATION POTENTIAL----------------")
            visualizations = self.check_visualization_potential(df)
            
            return ResponseModel(
                narrative=narrative,
                available_visualizations=visualizations,
                has_visualizations=len(visualizations) > 0,
                raw_data=df.to_dict(orient='records')
            )
            
        except Exception as e:
            print(f"Error in analyze: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during analysis: {str(e)}"
            )

@app.post("/analyze")
async def analyze_question(question: str) -> ResponseModel:
    analyzer = FinOpsAnalyzer()
    return analyzer.analyze(question)


@app.post("/generate_chart")
async def generate_chart(question: str, chart_type: str):
    analyzer = FinOpsAnalyzer()
    # First get the data
    response = analyzer.analyze(question)

    if not response.has_visualizations:
        raise HTTPException(
            status_code=400, detail="No visualization available for this data")

    # Check if requested chart type is supported
    if not any(v.type == chart_type and v.supported for v in response.available_visualizations):
        raise HTTPException(status_code=400, detail=f"Chart type {chart_type} not supported for this data")

    # Return data formatted for the requested chart type
    # Frontend will handle the actual chart rendering
    return {
        "chart_type": chart_type,
        "data": response.raw_data,
        "config": {
            "title": f"{chart_type.capitalize()} chart for {question}",
            # Add more chart-specific configuration as needed
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
