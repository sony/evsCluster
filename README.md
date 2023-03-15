# evsCluster

evsCluster is a set of Python scripts to process EVS (Event-based vision sensor) data.
evsCluster is compatible with EVT 3.0 Format data captured by a camera equipped with an [IMX636](https://www.sony-semicon.com/en/products/is/industry/evs.html) sensor.


___
### Installation

evsCluster is a set of Python scripts that can be executed by Command Prompt.
No installation is required for the scripts but the dependency listed below needs to be met.
<br>\* Note that the repository needs to be downloaded as a whole and any content should not be moved or deleted.



### Dependency

evsCluster is dependent on the following language and libraries.
Parentheses show the tested version. You need to install any library absent from your environment.
<br>\* The scripts are tested under a pip environment but may work under a conda environment as well.

- Python (3.8.9)
- NumPy (1.19.5)
- Numba (0.55.0)
- SciPy (1.7.0)
- Matplotlib (3.4.2)
- OpenCV-Python (4.5.2.54)



### Overview

evsCluster consists of four independent GUI applications that are launched by the following scripts.
Other scripts are configuration files or modules to be called by applications.

| Script | Explanation |
| ---- | --- |
| evs2cluster.py | Main application for cluster analysis |
| cluster2radarChart.py | To plot a radar chart based on cluster analysis results |
| learn.py | For machine learning of cluster classification |
| evs2video.py | To quickly visualize EVS data without analysis |



### Configuration
There are two configuration files contained in the top folder.
<br>\* In the Python scripts, "1/0" are used instead of "True/False" for configuration variables.

| Script | Explanation |
| ---- | --- |
| config_by_gui.py | Settings that are supposed to be often changed depending on user's demand and are intended to be edited by a GUI application |
| config.py | All other minor settings that users are expected to edit manually when needed |



### Tutorial and sample data
Refer to "Tutorial.md" in the top folder.



### License
The software may be used only for non-commercial use.
"Non-commercial" means research, education or evaluation purposes only.
If you redistribute the software with modification, you have to provide the source code on request.
For further details, refer to "LICENSE.txt" in the top folder.



### Reference
An example of research works using evsCluster is the following:
[Millisecond-scale behaviours of plankton quantified in situ and in vitro using the Event-based Vision Sensor (EVS)](https://doi.org/10.1101/2023.01.11.523686)



# Notes
Please note that Sony cannot answer or respond to any inquiries regarding the contents of the scripts.
