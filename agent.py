import anthropic
import base64
import json
import sqlite3
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads API key from .env file

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

FIELDS = [
    {
        "id": "psma_tac_precursor",
        "label": "SAP Batch — PSMA-TAC Precursor",
        "page": 5,
        "prompt": """Find the Materials Receipt table.
Locate the row for the BAY3546827-225Ac-PSMA-TAC precursor vial.
Extract the handwritten SAP Batch Number for that row.
Return JSON only — no preamble, no markdown:
{"sap_batch_number": "...", "confidence": "high|medium|low"}""",
    },
    {
        "id": "ac225_stock",
        "label": "SAP Batch — 225-Ac Stock Solution",
        "page": 6,
        "prompt": """Find Section 5 — 225-Ac Stock Solutions.
Extract the SAP batch number for the row that is NOT marked N/A.
Return JSON only:
{"sap_batch_number": "...", "confidence": "high|medium|low"}""",
    },
    {
        "id": "order_reference",
        "label": "Order Reference Number",
        "page": 7,
        "prompt": """Find Section 6 — Batch Details.
Extract only the Order Reference Number (labeled 'Order reference no.').
Return JSON only:
{"order_reference_number": "...", "confidence": "high|medium|low"}""",
    },
    {
        "id": "vial_masses",
        "label": "Vial Masses",
        "page": 11,
        "prompt": """Find Section 9 — Preparation DP Vials.
There is a table with rows for each vial. Extract every handwritten weight value.
For each row return the vial label exactly as written and its weight in grams to 2 decimal places.
Use null for any weight that is blank or marked N/A.
Return JSON only:
{"vials": [{"label": "...", "weight_g": "..."}], "confidence": "high|medium|low"}""",
    },
]

# Core functions


def rasterize_page(pdf_path: str, page_num: int) -> str:
    """Convert a single PDF page to base64 JPEG for vision."""
    output_prefix = "/tmp/mbr_page"

    # Remove any leftover files from previous runs
    for f in Path("/tmp").glob("mbr_page-*.jpg"):
        f.unlink()

    subprocess.run(
        [
            "pdftoppm",
            "-jpeg",
            "-r",
            "200",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            pdf_path,
            output_prefix,
        ],
        check=True,
        capture_output=True,
    )

    output_files = sorted(Path("/tmp").glob("mbr_page-*.jpg"))
    if not output_files:
        raise FileNotFoundError(f"pdftoppm produced no output for page {page_num}")

    return base64.b64encode(output_files[0].read_bytes()).decode()


def extract_field(image_b64: str, field: dict) -> dict:
    """Send a page image to Claude and get structured JSON back."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="""You are a pharmaceutical manufacturing data extraction assistant.
You are reading scanned pages from a controlled Batch Manufacturing Record (MBR).
Handwritten values may appear in blue or black ink.
Extract ONLY what is explicitly written — never infer or assume values.
Always return valid JSON with no markdown fences, no preamble.""",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": field["prompt"]},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if model includes them despite instructions
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def save_to_db(records: list, db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mbr_extractions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_document    TEXT NOT NULL,
            field           TEXT NOT NULL,
            page            INTEGER,
            value           TEXT,
            confidence      TEXT,
            extraction_date TEXT,
            model_version   TEXT
        )
    """)

    from datetime import date

    today = date.today().isoformat()

    for r in records:
        conn.execute(
            """
            INSERT INTO mbr_extractions
            (pdf_document, field, page, value, confidence, extraction_date, model_version)
            VALUES (?,?,?,?,?,?,?)
        """,
            (
                r["pdf_document"],
                r["field"],
                r["page"],
                r["value"],
                r.get("confidence"),
                today,
                "claude-sonnet-4-20250514",
            ),
        )

    conn.commit()
    conn.close()
    print(f"  Saved {len(records)} records to {db_path}")


def flatten_result(field: dict, result: dict, batch_num: str) -> list:
    rows = []

    if field["id"] == "psma_tac_precursor":
        rows.append(
            {
                "pdf_document": batch_num,
                "field": "SAP_Batch_PSMA_TAC_Precursor",
                "page": field["page"],
                "value": result.get("sap_batch_number"),
                "confidence": result.get("confidence"),
            }
        )

    elif field["id"] == "ac225_stock":
        rows.append(
            {
                "pdf_document": batch_num,
                "field": "SAP_Batch_225Ac_Stock",
                "page": field["page"],
                "value": result.get("sap_batch_number"),
                "confidence": result.get("confidence"),
            }
        )

    elif field["id"] == "order_reference":
        rows.append(
            {
                "pdf_document": batch_num,
                "field": "Order_Reference_Number",
                "page": field["page"],
                "value": result.get("order_reference_number"),
                "confidence": result.get("confidence"),
            }
        )

    elif field["id"] == "vial_masses":
        for vial in result.get("vials", []):
            rows.append(
                {
                    "pdf_document": batch_num,
                    "field": vial.get("label"),
                    "page": field["page"],
                    "value": vial.get("weight_g"),
                    "confidence": result.get("confidence"),
                }
            )

    return rows


# Main entry point


def process_pdf(pdf_path: str, output_db: str):
    """Run the full extraction pipeline on one MBR PDF."""
    pdf_path = str(Path(pdf_path).resolve())
    batch_num = Path(pdf_path).stem  # e.g. "batch_202124"

    print(f"\nProcessing: {pdf_path}")
    print(f"Batch:      {batch_num}")
    print(f"Output DB:  {output_db}\n")

    all_rows = []

    for field in FIELDS:
        print(f"  → Extracting: {field['label']} (page {field['page']})...")
        try:
            image_b64 = rasterize_page(pdf_path, field["page"])
            result = extract_field(image_b64, field)
            rows = flatten_result(field, result, batch_num)
            all_rows.extend(rows)
            confidence = result.get("confidence", "?")
            print(f"    ✓ Done  [{confidence} confidence]")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            all_rows.append(
                {
                    "pdf_document": batch_num,
                    "field": field["label"],
                    "page": field["page"],
                    "sap_item_number": field["sap_item"],
                    "value": f"ERROR: {e}",
                    "units": None,
                    "confidence": "failed",
                    "notes": None,
                }
            )

    save_to_db(all_rows, output_db)
    print(f"\nDone. {len(all_rows)} records extracted.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <path-to-pdf> [output.db]")
        sys.exit(1)

    pdf_input = sys.argv[1]
    db_output = sys.argv[2] if len(sys.argv) > 2 else "extractions.db"
    process_pdf(pdf_input, db_output)
