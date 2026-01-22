# LlmEnergy
A series of scripts that estimate the energy consumption of local inference on popular, open-source SLMs. The scripts use a list of small models available on Ollama.

The run consists of sequential measurements using two python libraries: `PyJoules` and `CodeCarbon`. The `plot.py` script can be used to read the CSV files and create a bar chart comparing the results.

## Run
To run the measurement, execute `bash run.sh S`, where `S` is the number of seconds for the measurement. IMPORTANT: The PyJoules measurement is hardware-dependent and has been adjusted for my machine. It measures consumption for only one CPU and one Nvidia GPU.
