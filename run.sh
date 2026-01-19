#!/usr/bin/env bash

models=(
  "deepseek-r1:1.5b"
  "gemma3:4b"
  "gemma3:270m"
  "qwen3:4b"
  "qwen3:0.6b"
  "llama3.2:3b"
  "phi3:3.8b"
)

# a trap for Ctrl+C in case the while loop gets stuck
# gets trapped after a single model run completes
trap printout SIGINT SIGTERM
printout() {
  echo ""
  echo "trapped"
  exit
}

#chmod to access powercap files
sudo chmod 0444 /sys/class/powercap/*/*/energy_uj

#---------------------
#    PyJoules
#---------------------

#create venv and activate it
python3 -m venv ./PyJoules/.venv
source ./PyJoules/.venv/bin/activate

#install packages
pip install -r ./PyJoules/requirements.txt

#run inference for each model
(
  for model in "${models[@]}"; do
    while true; do
      trap printout SIGINT SIGTERM
      echo "starting a run for $model tracking with PyJoules"
      #timeout just in case something goes wrong
      timeout $(($1 * 2 + 30)) python3 ./PyJoules/main.py --model "$model" --seconds $1
      #try until success (exit code 0)
      if [ $? -eq 0 ]; then
        echo "Run for $model succeded"
        break
      else
        echo "Run for $model failed"
      fi
    done
  done
)

deactivate

#---------------------
#    CodeCarbon
#---------------------

#create venv and activate it
python3 -m venv ./CodeCarbon/.venv
source ./CodeCarbon/.venv/bin/activate

#install packages
pip install -r ./CodeCarbon/requirements.txt

#run inference for each model
(
  for model in "${models[@]}"; do
    while true; do
      trap printout SIGINT SIGTERM
      echo "starting a run for $model tracking with CodeCarbon"
      #timeout just in case something goes wrong
      timeout $(($1 * 2 + 30)) python3 ./CodeCarbon/main.py --model "$model" --seconds $1
      #try until success (exit code 0)
      if [ $? -eq 0 ]; then
        echo "Run for $model succeded"
        break
      else
        echo "Run for $model failed"
      fi
    done
  done
)

deactivate
