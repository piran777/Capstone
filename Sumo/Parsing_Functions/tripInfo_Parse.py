import xml.etree.ElementTree as ET
import csv

# Parse the XML file
tree = ET.parse(r"C:\Users\nicka\School\Capstone\Sumo\RawOutputs\tripinfo_rawOut.xml")
root = tree.getroot()

# Open the CSV file
with open(r"C:\Users\nicka\School\Capstone\Sumo\ParsedOutputs\tripinfo_Parsed.csv", 'w', newline='') as csvfile:
    fieldnames = ['id', 'depart', 'departLane', 'departPos', 'departSpeed', 'departDelay', 'arrival', 'arrivalLane', 'arrivalPos', 'arrivalSpeed', 'duration', 'routeLength', 'waitingTime', 'waitingCount', 'stopTime', 'timeLoss', 'rerouteNo', 'devices', 'vType', 'speedFactor', 'vaporized', 'CO_abs', 'CO2_abs', 'HC_abs', 'PMx_abs', 'NOx_abs', 'fuel_abs', 'electricity_abs']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data
    for tripinfo in root.findall('tripinfo'):
        emissions = tripinfo.find('emissions')
        writer.writerow({**tripinfo.attrib, **emissions.attrib})