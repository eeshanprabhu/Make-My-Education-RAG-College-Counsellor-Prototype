"""Build the vector store from the college dataset.

Run once before answering:
    python ingest.py

Reads sample_colleges.csv (preferred) or sample_colleges.xlsx, embeds each
college with Gemini, and upserts the vectors + metadata into a persisted Chroma
collection. Safe to re-run (upsert, not add) so a clean checkout can rebuild.
"""

import math
import os

import chromadb
import pandas as pd
from google import genai
from google.genai import types

from rag import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_client,
)

CSV_PATH = "sample_colleges.csv"
XLSX_PATH = "sample_colleges.xlsx"


def load_dataframe() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    if os.path.exists(XLSX_PATH):
        return pd.read_excel(XLSX_PATH)
    raise FileNotFoundError(
        f"Could not find {CSV_PATH} or {XLSX_PATH} in the project folder."
    )


def build_text(row) -> str:
    """One natural-language description per college (retrieval-friendly)."""
    return (
        f"The unique identifier of the college is {row['college_id']}. "
        f"Name of the college is {row['name']}. "
        f"{row['name']} is located in this city: {row['city']}. "
        f"{row['name']} is located in this state: {row['state']}. "
        f"Type of the {row['name']} college is: {row['type']}. "
        f"These are the courses offered by {row['name']}: {row['courses_offered']}. "
        f"The total annual fees (per year) of {row['name']} is {row['annual_fees_inr']}. "
        f"The last-year cutoff for {row['name']} is: {row['last_year_cutoff_pct']}. "
        f"The total seats in {row['name']} is: {row['total_seats']}. "
        f"Hostel availability for {row['name']} is: {row['hostel_available']}. "
        f"The NAAC Grade for {row['name']} is: {row['naac_grade']}. "
        f"The average placement (LPA) for {row['name']} is: {row['avg_placement_lpa']}. "
        f"{row['name']} was established in the year: {row['established_year']}. "
        f"Additional information about {row['name']}: {row['about']}"
    )


def clean_metadata(row) -> dict:
    """Coerce a dataframe row into Chroma-safe primitives (no numpy types, no NaN)."""
    out = {}
    for key, value in row.items():
        if isinstance(value, float) and math.isnan(value):
            out[key] = ""
        elif isinstance(value, (pd.Int64Dtype,)):
            out[key] = int(value)
        else:
            # Convert numpy scalars to native python via .item() when available.
            item = getattr(value, "item", None)
            out[key] = item() if callable(item) else value
    return out


def main():
    df = load_dataframe()
    client = get_client()

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []

    for _, row in df.iterrows():
        text = build_text(row)
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        ids.append(str(row["college_id"]))
        embeddings.append(result.embeddings[0].values)
        documents.append(text)
        metadatas.append(clean_metadata(row))

    # upsert => safe to re-run without duplicate-id errors.
    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    print(f"Ingested {collection.count()} colleges into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
