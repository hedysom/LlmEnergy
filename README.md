# LlmEnergy
A series of scripts that estimate the energy consumption of local inference on popular, open-source SLMs. The scripts use a list of small models available on Ollama:
deepseek-r1:1.5b, gemma3:4b, gemma3:270m, qwen3:4b, qwen3:0.6b, llama3.2:3b, phi3:3.8b.

## Run
To run the measurement, execute `bash run.sh S`, where `S` is the number of seconds for the measurement. IMPORTANT: The PyJoules measurement is hardware-dependent and has been adjusted for my machine. It measures consumption for only one CPU and one Nvidia GPU.
