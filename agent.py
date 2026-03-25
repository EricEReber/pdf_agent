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

# Field definitions
FIELDS = [
    {
        "id": "psma_tac_precursor",
        "label": "SAP Batch — PSMA-TAC Precursor",
        "page": 5,
        "sap_item": "110810",
        "prompt": """Find SAP Item 110810 in the Materials Receipt table.
Extract the handwritten SAP Batch Number in the SAP Batch Number column.
Return JSON only — no preamble, no markdown:
{"sap_batch_number": "...", "expiration_date": "...", "confidence": "high|medium|low", "notes": "..."}""",
    },
    {
        "id": "ac225_stock",
        "label": "SAP Batch — 225-Ac Stock Solution",
        "page": 6,
        "sap_item": "110860",
        "prompt": """Find Section 5 — 225-Ac Stock Solutions.
Extract the SAP batch number and expiration date/time for the row that is NOT marked N/A.
Return JSON only:
{"sap_batch_number": "...", "manufacturer": "...", "expiration_date": "...", "expiration_time": "...", "confidence": "high|medium|low"}""",
    },
    {
        "id": "order_reference",
        "label": "Order Reference Number",
        "page": 7,
        "sap_item": "N/A",
        "prompt": """Find Section 6 — Batch Details.
Extract all handwritten values.
Return JSON only:
{"order_reference_number": "...", "manufacturing_date": "...", "cohort_number": "...", "number_of_doses": "1|2|3", "target_activity_mbq": "...", "confidence": "high|medium|low"}""",
    },
    {
        "id": "vial_masses",
        "label": "Vial Masses",
        "page": 11,
        "sap_item": "109417",
        "prompt": """Find Section 9 — Preparation DP Vials.
Extract each handwritten weight value in grams (2 decimal places). Use null if N/A or blank.
Return JSON only:
{"W1_DP01_g": "...", "W2_DP02_g": null, "W3_DP03_g": null, "W4_Sterility_g": "...", "W5_QC_g": "...", "W6_Retain_g": "...", "confidence": "high|medium|low"}""",
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
    """Write extracted records to SQLite with full audit trail."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mbr_extractions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_number    TEXT NOT NULL,
            field           TEXT NOT NULL,
            page            INTEGER,
            sap_item_number TEXT,
            value           TEXT,
            units           TEXT,
            confidence      TEXT,
            notes           TEXT,
            extraction_date TEXT,
            model_version   TEXT,
            source_document TEXT
        )
    """)

    from datetime import date

    today = date.today().isoformat()

    for r in records:
        conn.execute(
            """
            INSERT INTO mbr_extractions
            (batch_number, field, page, sap_item_number, value, units,
             confidence, notes, extraction_date, model_version, source_document)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                r["batch_number"],
                r["field"],
                r["page"],
                r["sap_item_number"],
                r["value"],
                r.get("units"),
                r.get("confidence"),
                r.get("notes"),
                today,
                "claude-sonnet-4-20250514",
                "MBR-00497 v4.0",
            ),
        )

    conn.commit()
    conn.close()
    print(f"  Saved {len(records)} records to {db_path}")


def flatten_result(field: dict, result: dict, batch_num: str) -> list:
    """Normalize a field extraction result into database rows."""
    rows = []

    if field["id"] == "psma_tac_precursor":
        rows.append(
            {
                "batch_number": batch_num,
                "field": "SAP_Batch_PSMA_TAC_Precursor",
                "page": field["page"],
                "sap_item_number": field["sap_item"],
                "value": result.get("sap_batch_number"),
                "units": None,
                "confidence": result.get("confidence"),
                "notes": f"Exp: {result.get('expiration_date')}",
            }
        )

    elif field["id"] == "ac225_stock":
        rows.append(
            {
                "batch_number": batch_num,
                "field": "SAP_Batch_225Ac_Stock",
                "page": field["page"],
                "sap_item_number": field["sap_item"],
                "value": result.get("sap_batch_number"),
                "units": None,
                "confidence": result.get("confidence"),
                "notes": f"Mfr: {result.get('manufacturer')}, Exp: {result.get('expiration_date')} {result.get('expiration_time')}",
            }
        )

    elif field["id"] == "order_reference":
        for fname, fval, funits in [
            ("Order_Reference_Number", result.get("order_reference_number"), None),
            ("Cohort_Number", result.get("cohort_number"), None),
            ("Number_Of_Doses", result.get("number_of_doses"), "doses"),
            ("Target_Activity_MBq", result.get("target_activity_mbq"), "MBq"),
        ]:
            rows.append(
                {
                    "batch_number": batch_num,
                    "field": fname,
                    "page": field["page"],
                    "sap_item_number": "N/A",
                    "value": fval,
                    "units": funits,
                    "confidence": result.get("confidence"),
                    "notes": None,
                }
            )

    elif field["id"] == "vial_masses":
        mass_map = [
            ("W1_DP01_g", "Vial_Mass_DP01_W1"),
            ("W2_DP02_g", "Vial_Mass_DP02_W2"),
            ("W3_DP03_g", "Vial_Mass_DP03_W3"),
            ("W4_Sterility_g", "Vial_Mass_Sterility_W4"),
            ("W5_QC_g", "Vial_Mass_QC_W5"),
            ("W6_Retain_g", "Vial_Mass_Retain_W6"),
        ]
        for key, fname in mass_map:
            rows.append(
                {
                    "batch_number": batch_num,
                    "field": fname,
                    "page": field["page"],
                    "sap_item_number": "109417",
                    "value": result.get(key),
                    "units": "g",
                    "confidence": result.get("confidence"),
                    "notes": result.get("notes"),
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
                    "batch_number": batch_num,
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
