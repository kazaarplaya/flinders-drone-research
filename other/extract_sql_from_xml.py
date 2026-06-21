import base64
import xml.etree.ElementTree as ET
from pathlib import Path

input_file = Path("SQL_tables_AuDroK_20230620.xml")
output_file = Path("SQL_tables_AuDroK_20230620.sql")

tree = ET.parse(input_file)
root = tree.getroot()

binary_data = None

for element in root.iter():
    if element.tag.endswith("binary"):
        binary_data = element.text
        break

if binary_data is None:
    raise ValueError("No Base64 binary data found in the XML file.")

sql_bytes = base64.b64decode(binary_data)

with open(output_file, "wb") as f:
    f.write(sql_bytes)

print(f"Saved SQL dump as: {output_file}")