from typing import Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    run_id: str = Field(default="", description="Unique identifier for this pipeline run.")
    status: str = Field(default="initialized", description="Current status of the pipeline run.")

    uploaded_files: list[str] = Field(
        default_factory=list,
        description="Paths to uploaded CSV files.",
    )
    business_intent: str = Field(
        default="",
        description="The user's business question in plain English.",
    )

    profile_path: str = Field(
        default="",
        description="Path to the data profile JSON file.",
    )

    sttm_bronze_path: str = Field(
        description="Path to the Bronze layer source-to-target mapping rules.",
    )
    sttm_silver_path: str = Field(
        description="Path to the Silver layer source-to-target mapping rules.",
    )
    sttm_gold_path: str = Field(
        description="Path to the Gold layer source-to-target mapping rules.",
    )
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human has approved the transformation rules.",
    )

    bronze_output_paths: list[str] = Field(
        description="Paths to Bronze Parquet output files.",
    )
    silver_output_paths: list[str] = Field(
        description="Paths to Silver Parquet output files.",
    )
    gold_output_paths: list[str] = Field(
        description="Paths to Gold Parquet output files.",
    )

    report_path: str = Field(
        default="",
        description="Path to the final HTML report.",
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message captured during pipeline execution, if any.",
    )