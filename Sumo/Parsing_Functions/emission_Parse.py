import xml.etree.ElementTree as ET
import csv

# Parse the XML file
tree = ET.parse(r"C:\Users\nicka\School\Capstone\Sumo\RawOutputs\emission_rawOut.xml")
root = tree.getroot()

# Open the CSV file
with open(r"C:\Users\nicka\School\Capstone\Sumo\ParsedOutputs\emission_parsed.csv", 'w', newline='') as csvfile:
    fieldnames = ['time', 'id', 'eclass', 'CO2', 'CO', 'HC', 'NOx', 'PMx', 'fuel', 'electricity', 'noise', 'route', 'type', 'waiting', 'lane', 'pos', 'speed', 'angle', 'x', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the data
    for timestep in root.findall('timestep'):
        time = timestep.get('time')
        for vehicle in timestep.findall('vehicle'):
            writer.writerow({'time': time, **vehicle.attrib})