
### Run a custom Docker image on a Duckiebot:

- Open a terminal window, `cd` into this folder, and run `rsync -a * duckie@<DUCKIE_NAME>.local:/code/agent/`
- Open new terminal and ssh into the Duckiebot (`ssh duckie@<DUCKIE_NAME>.local`)
- Run `cd ../../code/agent/` and run `docker build -t custom-baseline .` to build the custom image
- Leave Duckiebot ssh
- Open file explorer (Home) and enable viewing of hidden files
- Navigate to `~/.dt-shell/commands-multi/daffy/exercises/test/command.py`
- Open it, go to line 331, and change it to `agent_base_image = "custom-baseline:latest"` (make sure to leave previous version of this line to run exercises without custom images)
