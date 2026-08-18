import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================
# 1. INITIALIZE THE EXTRACTION ENGINE
# =====================================================================
model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"Loading {model_id} natively via Transformers...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# =====================================================================
# 2. READ THE TARGET EAD XML FILE
# =====================================================================
xml_file_path = sys.argv[1] if len(sys.argv) > 1 else "sample_ead.xml"

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

# Using lxml parser for robustness with XML structures
soup = BeautifulSoup(ead_xml_content, "xml")
prose_blocks = soup.find_all(["bioghist", "scopecontent", "abstract", "p"])
print(f"File loaded. Isolated {len(prose_blocks)} text segments to analyze.\n")

# =====================================================================
# 3. ADVANCED METADATA SCHEMA EXTRACTION LOOP
# =====================================================================
target_schema_example = json.dumps({
    "extracted_agents": [
        {
            "type": "Personal Name",
            "100": {
                "a": "Surname, Given Name",
                "c": "Titles or honorifics if any",
                "d": "Dates if any"
            },
            "670_citation": "Brief word snippet from the text proving this person exists"
        },
        {
            "type": "Corporate Name",
            "110": {
                "a": "Full Cleaned Corporate/Organizational Name",
                "b": "Subordinate department or division if any"
            },
            "670_citation": "Brief word snippet from the text proving this corporation exists"
        }
    ]
}, indent=2)

all_extracted_agents = []

for index, block in enumerate(prose_blocks, start=1):
    raw_text = block.get_text(strip=True)
    if not raw_text or len(raw_text) < 10:
        continue
        
    system_instruction = (
        "You are an expert library metadata engineer. Read the unstructured text narrative. "
        "Discover every individual person and corporate organization mentioned.\n"
        "1. For People: Use 'type': 'Personal Name' and map to '100'. Normalize to Inverted Order.\n"
        "2. For Organizations: Use 'type': 'Corporate Name' and map to '110'. Keep in Natural Order.\n"
        "3. For '670_citation': Quote 4-8 words directly from the text where the agent is discussed.\n"
        f"Format the final output strictly matching this JSON structure:\n{target_schema_example}\n"
        "Output ONLY raw, valid JSON without markdown code blocks or prose."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Extract all agents (people and corps) from this block:\n{raw_text}"}
    ]
    
    print(f"Scanning Text Segment #{index}...")
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=False
        )
    
    input_len = model_inputs.input_ids.shape[1]
    response_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
    
    # Extract JSON robustly using regex matching
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
        
    try:
        parsed_json = json.loads(response_text)
        
        if isinstance(parsed_json, list):
            agents = parsed_json
        elif isinstance(parsed_json, dict):
            agents = parsed_json.get("extracted_agents", [])
        else:
            agents = []

        print(f" -> Processed segment #{index}. Isolated {len(agents)} item(s).")
        all_extracted_agents.extend(agents)
    except json.JSONDecodeError:
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
    
    # Fixed raw URL namespace string
    root = ET.Element("record", xmlns="http://www.loc.gov/MARC21/slim")
    
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
        
        df_110 = ET.SubElement(root, "datafield", tag="110", ind1="2", ind2=" ")
        ET.SubElement(df_110, "subfield", code="a").text = corp_a
        if subfields_110.get("b"):
            ET.SubElement(df_110, "subfield", code="b").text = subfields_110["b"]
            
    else:
        continue

    # Field 670
    df_670 = ET.SubElement(root, "datafield", tag="670", ind1=" ", ind2=" ")
    ET.SubElement(df_670, "subfield", code="a").text = f"EAD Finding Aid source file: {xml_file_path}"
    ET.SubElement(df_670, "subfield", code="b").text = f"({citation_text})"

    # Write formatted XML
    raw_xml_bytes = ET.tostring(root, encoding="utf-8")
    parsed_dom = minidom.parseString(raw_xml_bytes)
    pretty_xml_string = parsed_dom.toprettyxml(indent="  ")
    
    safe_filename = f"{record_index:03d}_{base_filename_string[:40]}.xml"
    full_output_path = os.path.join(output_dir, safe_filename)
    
    with open(full_output_path, "w", encoding="utf-8") as out_file:
        out_file.write(pretty_xml_string)
        
    print(f" -> Exported individual authority record: {full_output_path}")

print(f"\nProcessing complete! All files generated in folder: ./{output_dir}/")