import xml.etree.ElementTree as ET
import csv
import os

print(os.getcwd())

tree = ET.parse(r"C:\Users\nicka\School\Capstone\Sumo\RawOutputs\fcd-output_rawOut.xml")
root = tree.getroot()

# Open a file for writing
with open(r"C:\Users\nicka\School\Capstone\Sumo\ParsedOutputs\FCD_Parsed.csv", 'w', newline='') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(["time", "id", "x", "y", "angle", "type", "speed", "pos", "lane", "slope"])

    for timestep in root.findall('timestep'):
        time = timestep.get('time')
        for vehicle in timestep.findall('vehicle'):
            writer.writerow([time] + list(vehicle.attrib.values()))

print("CSV file has been created")