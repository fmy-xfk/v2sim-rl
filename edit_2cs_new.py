import os
from xml.etree import ElementTree as ET
import gzip
ORIGIN = "cases/2cs_new/2cs_new.veh.xml.gz"
fh = gzip.open(ORIGIN, "rb")
root = ET.ElementTree(file=fh).getroot()
fh.close()
assert root is not None, "Root element is None"
for node in root.findall("vehicle"):
    for trip in node.findall("trip"):
        if trip.attrib["id"].endswith("1"):
            trip.attrib["route_edges"] = "-E40 E61"
        elif trip.attrib["id"].endswith("2"):
            trip.attrib["route_edges"] = "E61 E31"
        elif trip.attrib["id"].endswith("3"):
            trip.attrib["route_edges"] = "E31 -E40"
if os.path.exists(ORIGIN + ".bak"):
    os.remove(ORIGIN + ".bak")
os.rename(ORIGIN, ORIGIN + ".bak")
ET.ElementTree(root).write("cases/2cs_new/2cs_new.veh.xml", encoding="UTF-8", xml_declaration=True)