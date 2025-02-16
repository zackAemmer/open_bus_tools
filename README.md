# Open Bus Tools
Tools for open bus data collection, travel time prediction, energy consumption and visualization.

This is companion code for a dissertation in Civil and Environmental Engineering at the University of Washington.

If not installed as package, you may need to add [export PYTHONPATH="${PYTHONPATH}:/my/other/path/to/open_bus_tools"] to .bashrc to run the scripts.

### Workflow
There are two folders in the base directory /scripts and /notebooks. These are useful starting points for understanding how different parts of the code work together. The two main modules of the package are:
* drivecycle: Used to estimate energy use based on vehicle and driving profile from a given actual or predicted GPS trajectory.

* traveltime: Used to predict travel times using ML-based geospatial data mining.

The /webscraper folder has code used to gather GTFS and GTFS-RT feeds. These data are used in training the travel time models.

### Motivation
Standardized and open source bus data including static schedules and realtime positions have become widely available in public web APIs. Though these data primarily underly popular map-based trip planning applications, they also enable new analyses in understanding, forecasting and improving bus operations across agencies and cities. Due to their lower resolution and simpler, anonymized features, these data are more challenging to work with than those of the underlying sensors, making them unfavorable for direct operations analysis by transit agencies. However, the wide scale of these data and their open source nature make them a valuable resource for researchers and planners.

This work develops and tests a set of tools for analyzing bus operations using open source data. Central to this endeavor is the ongoing collection of a multi-year dataset from the King County Metro transit network in Seattle, Washington, rapidly approaching one billion tracked bus locations. First, these data are used in basic roadway segment matching and aggregation to visualize the spatiotemporal dynamics of different types of delays in the transit system. The findings are used to identify locations in the network where targeted transit priority treatments can generate the greatest benefit. Then a set of deep learning models and baselines are developed to forecast bus travel times under different data availability scenarios. Their generalizability is tested across different cities and transit networks. Finally, these models are used to estimate the drive cycle energy demands of a system-wide bus electrification project, and are validated using results of a more traditional approach with direct sensor data.

![Bus capacities required for electrification](thumbnail.png?raw=true "Example of Tool Output")

### Data Sample
The full dataset is stored on Amazon S3 and downloaded for local analysis as needed, please feel free to reach out if you would like access.

### Other Libraries and Code You Might Find Useful
* https://github.com/kraina-ai/srai
* https://github.com/NREL/fastsim
* https://github.com/smohiudd/drivecycle
* https://github.com/EricaEgg/Route_Dynamics
* https://github.com/mrcagney/gtfs_kit