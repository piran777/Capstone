# Capstone
Trips.trips.xml is used by the NetEdit tool RandomTrips.py to generate the routes.rou.xml, which is referenced and used in the configuration file simcfg.sumocfg
Simcfg.sumocfg define the files and resouces used as well as output paths for our data

Delete all files in RawOutputs and ParsedOutputs each time sim is run

Files used for model can be found in parsedOutputs

1. Open Config folder and run sumocfg file
2. Run Sim, raw outputs will be in RawOutputs folder
3. run functions in Parsing_Functions to parse to readable csv
4. Results of sim are al ParsedOutputs folder