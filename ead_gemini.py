import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

# =====================================================================
# 1. INITIALIZE THE GEMINI CLIENT
# =====================================================================
# Ensure GEMINI_API_KEY is set in your environment variables:
# export GEMINI_API_KEY="your-api-key"
client = genai.Client()
model_id = "gemini-2.5-flash"

print(f"Initialized Gemini Client using model: {model_id}...")

# =====================================================================
# 2. READ THE TARGET EAD XML FILE
# =====================================================================
xml_file_path = sys.argv[1] if len(sys.argv) > 1 else "sample_ead.xml"

# Quick structural safeguard to create test records if missing
if not os.path.exists(xml_file_path):
    print(f"\nCreating sample '{xml_file_path}' for run test...")
    with open(xml_file_path, "w", encoding="utf-8") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<ead>
    <archdesc level="collection">
        <bioghist>
            <p>The collection details the life of Arthur Pendragon (1950-2012) during his tenure at the university. 
            Many letters are addressed to his close colleague, Guinevere Smith. Merlin Ambrosius oversaw the initial funding.</p>
            <p>The Acme Archival Corporation was hired in 1998 to evaluate the rare manuscript fragments.</p>
        </bioghist>
    </archdesc>
</ead>""")

with open(xml_file_path, "r", encoding="utf-8") as file:
    ead_xml_content = file.read()

soup = BeautifulSoup(ead_xml_content, "xml")
prose_blocks = soup.find_all(["bioghist", "scopecontent", "abstract", "p"])
print(f"File loaded. Isolated {len(prose_blocks)} text segments to analyze.\n")

# =====================================================================
# 3. ADVANCED METADATA SCHEMA EXTRACTION LOOP
# =====================================================================
all_extracted_agents = []

# Define standard system prompt framing for Gemini
system_instruction = (
    "You are an expert library metadata engineer. Read the unstructured text narrative. "
    "Discover every individual person and corporate organization mentioned.\n"
    "1. For People: Use 'type': 'Personal Name' and map to '100'. Normalize to Inverted Order (Surname, Given Name).\n"
    "2. For Organizations: Use 'type': 'Corporate Name' and map to '110'. Keep in Natural Order.\n"
    "3. For '670_citation': Quote 4-8 words directly from the text where the agent is discussed."
)

for index, block in enumerate(prose_blocks, start=1):
    raw_text = block.get_text(strip=True)
    if not raw_text or len(raw_text) < 10:
        continue

    print(f"Scanning Text Segment #{index}...")

    # Leveraging Gemini's Structured JSON Output capabilities
    response = client.models.generate_content(
        model=model_id,
        contents=f"Extract all agents (people and corps) from this block:\n{raw_text}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "extracted_agents": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "type": {
                                    "type": "STRING",
                                    "enum": ["Personal Name", "Corporate Name"]
                                },
                                "100": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "a": {"type": "STRING", "description": "Surname, Given Name"},
                                        "c": {"type": "STRING", "description": "Titles or honorifics if any"},
                                        "d": {"type": "STRING", "description": "Dates if any"}
                                    }
                                },
                                "110": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "a": {"type": "STRING", "description": "Full Cleaned Corporate/Organizational Name"},
                                        "b": {"type": "STRING", "description": "Subordinate department or division if any"}
                                    }
                                },
                                "670_citation": {
                                    "type": "STRING",
                                    "description": "Brief 4-8 word snippet from text proving agent exists"
                                }
                            },
                            "required": ["type", "670_citation"]
                        }
                    }
                },
                "required": ["extracted_agents"]
            }
        )
    )

    try:
        parsed_json = json.loads(response.text)
        agents = parsed_json.get("extracted_agents", [])
        print(f" -> Processed segment #{index}. Isolated {len(agents)} item(s).")
        all_extracted_agents.extend(agents)
    except (json.JSONDecodeError, AttributeError):
        print(f" -> Error parsing JSON on segment #{index}. Skipping.")

    print("-" * 70 + "\n")

# =====================================================================
# 4. EXPORT INDIVIDUAL AGENTS TO SEPARATE XML AUTHORITY FILES
# =====================================================================
print(f"Beginning file serialization loop for {len(all_extracted_agents)} authority entries...")

output_dir = "extracted_authority_records"
os.makedirs(output_dir, exist_ok=True)

for record_index, agent in enumerate(all_extracted_agents, start=1):
    agent_type = agent.get("type", "Personal Name")
    citation_text = agent.get("670_citation", "Name mentioned in finding aid narrative text.")

    # Standard MARCXML root container
    root = ET.Element("record", xmlns="http://www.loc.gov/MARC21/slim")

    # Standard Leader tag
    leader = ET.SubElement(root, "leader")
    leader.text = "00000nz  a2200000n  4500"

    base_filename_string = "authority_record"

    # CASE A: Processing a standard Personal Name Heading
    if agent_type == "Personal Name" and "100" in agent:
        subfields_100 = agent.get("100", {})
        name_a = subfields_100.get("a")
        if not name_a:
            continue

        base_filename_string = re.sub(r'[^a-zA-Z0-9_]', '', name_a.replace(" ", "_"))

        # ind1="1" indicates Single Surname formatting
        df_100 = ET.SubElement(root, "datafield", tag="100", ind1="1", ind2=" ")
        ET.SubElement(df_100, "subfield", code="a").text = name_a
        if subfields_100.get("c"):
            ET.SubElement(df_100, "subfield", code="c").text = subfields_100["c"]
        if subfields_100.get("d"):
            ET.SubElement(df_100, "subfield", code="d").text = subfields_100["d"]

    # CASE B: Processing a Corporate Entity Heading
    elif agent_type == "Corporate Name" and "110" in agent:
        subfields_110 = agent.get("110", {})
        corp_a = subfields_110.get("a")
        if not corp_a:
            continue

        base_filename_string = re.sub(r'[^a-zA-Z0-9_]', '', corp_a.replace(" ", "_"))

        # ind1="2" indicates Corporate Name in Direct Order
        df_110 = ET.SubElement(root, "datafield", tag="110", ind1="2", ind2=" ")
        ET.SubElement(df_110, "subfield", code="a").text = corp_a
        if subfields_110.get("b"):
            ET.SubElement(df_110, "subfield", code="b").text = subfields_110["b"]

    else:
        continue

    # GENERATE FIELD 670: Sources Found Note
    df_670 = ET.SubElement(root, "datafield", tag="670", ind1=" ", ind2=" ")
    ET.SubElement(df_670, "subfield", code="a").text = f"EAD Finding Aid source file: {xml_file_path}"
    ET.SubElement(df_670, "subfield", code="b").text = f"({citation_text})"

    # =====================================================================
    # 5. WRITE PRETTY FORMATTED XML INDIVIDUALLY TO DISK
    # =====================================================================
    raw_xml_bytes = ET.tostring(root, encoding="utf-8")
    parsed_dom = minidom.parseString(raw_xml_bytes)
    pretty_xml_string = parsed_dom.toprettyxml(indent="  ")

    safe_filename = f"{record_index:03d}_{base_filename_string[:40]}.xml"
    full_output_path = os.path.join(output_dir, safe_filename)

    with open(full_output_path, "w", encoding="utf-8") as out_file:
        out_file.write(pretty_xml_string)

    print(f" -> Exported individual authority record: {full_output_path}")

print(f"\nProcessing complete! All files generated in folder: ./{output_dir}/")