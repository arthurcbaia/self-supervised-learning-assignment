# ---
# args: ["--timeout", 10]
# ---

# ## Overview
#
# Quick snippet showing how to connect to a Jupyter notebook server running inside a Modal container,
# especially useful for exploring the contents of Modal Volumes.
# This uses [Modal Tunnels](https://modal.com/docs/guide/tunnels#tunnels-beta)
# to create a tunnel between the running Jupyter instance and the internet.
#
# If you want to your Jupyter notebook to run _locally_ and execute remote Modal Functions in certain cells, see the `basic.ipynb` example :)

import os
import subprocess
import time

import modal

app = modal.App(
    image=modal.Image.debian_slim(python_version="3.10").pip_install(
        "jupyter", "transformers[torch]", "datasets", "evaluate", "scikit-learn", "nltk", "matplotlib"
    )
)


JUPYTER_TOKEN = "1234"  # Change me to something non-guessable!
import pathlib

volume = modal.Volume.from_name("my-volume", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")

# 3 hours in seconds
TIMEOUT = 6 * 60 * 60

@app.function(concurrency_limit=1, volumes={VOL_MOUNT_PATH: volume}, timeout = TIMEOUT)
def run_jupyter(timeout: int):
    print(f"Running Jupyter Notebook server with timeout {timeout} seconds")
    jupyter_port = 8888
    with modal.forward(jupyter_port) as tunnel:
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "notebook",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout
            while time.time() < end_time:
                time.sleep(5)
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = TIMEOUT):
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout=timeout)


# Doing `modal run jupyter_inside_modal.py` will run a Modal app which starts
# the Juypter server at an address like https://u35iiiyqp5klbs.r3.modal.host.
# Visit this address in your browser, and enter the security token
# you set for `JUPYTER_TOKEN`.