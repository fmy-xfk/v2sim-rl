import os
from xml.etree import ElementTree as ET
import gzip
ORIGIN = "cases/drl_2cs/drl_2cs.veh.xml.gz"
fh = gzip.open(ORIGIN, "rb")
root = ET.ElementTree(file=fh).getroot()
fh.close()
assert root is not None, "Root element is None"
for node in root.findall("vehicle"):
    for trip in node.findall("trip"):
        if trip.attrib["id"].endswith("1"):
            trip.attrib["route_edges"] = "gneE21 gneE16"
        elif trip.attrib["id"].endswith("2"):
            trip.attrib["route_edges"] = "gneE16 gneE32"
        elif trip.attrib["id"].endswith("3"):
            trip.attrib["route_edges"] = "gneE32 gneE21"
if os.path.exists(ORIGIN + ".bak"):
    os.remove(ORIGIN + ".bak")
os.rename(ORIGIN, ORIGIN + ".bak")
ET.ElementTree(root).write("cases/drl_2cs/drl_2cs.veh.xml", encoding="UTF-8", xml_declaration=True)