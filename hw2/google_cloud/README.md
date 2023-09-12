## Using Google Cloud Compute

Here we provide 4 scripts to help you run experiments on Google Cloud Compute. Everyone has $50 of free credit to use, which should be enough for the semester if you don't leave machines running.

### 1. Install Google Cloud SDK

Follow the instructions [here](https://cloud.google.com/sdk/docs/install-sdk#installing_the_latest_version) to install the Google Cloud SDK. Then, run `gcloud init` to log in. Select the project in which you have claimed your $50 credit as the default project.

### 2. Create a VM

Run `google_cloud/create_instance.sh` to create a VM instance. **Make sure to run all of the scripts from the `hw2` directory.** This will create an instance called `cs285` which has one NVIDIA T4 GPU, 4 CPUs, and Python 3.10 with PyTorch 1.13 pre-installed. You can edit the command at the top of the script to create a different machine type (see [here](https://cloud.google.com/compute/gpus-pricing#accelerator-optimized) for pricing). However, we have chosen this machine type because the assignments are generally not very compute intensive; if you're only running one job at a time, a GPU usually does not even help. Compute only really becomes a bottleneck if you're running many jobs at once, in which case a T4 GPU should provide more than enough throughput.

The `create_instance.sh` script will also wait for the NVIDIA drivers to be installed, transfer your current directory (which should be `hw2`) to the VM, install all the dependencies, and then shut down the VM. To see your newly created VM, you can run `gcloud compute instances list`.

### 3. Run a job

The main entrypoint to running jobs is `launch.sh`. You use the script by adding it to the beginning of your command: e.g. `google_cloud/launch.sh bash sweep_lambda.sh`. This does the following things:

1. Starts the VM
2. Syncs your current directory (which should be `hw2`) to the VM
3. Starts a new tmux session, which prevents your job from getting killed if you suspend your local machine or otherwise lose connection to the VM

Inside the tmux session, the following things happen:

1. Your command runs until completion
2. The shell sleeps for 5 minutes
3. The VM is shut down

The reason the shell sleeps for 5 minutes is so that if your job immediately crashes, you have time to look at the error message before the VM shuts down.

### 4. Check on your job

After running `google_cloud/launch.sh`, you can connect to the running VM by running `gcloud compute ssh cs285 --zone="us-west1-b"`. Once connected, run `tmux ls` to see the running tmux sessions. There should only be one, but keep in mind that you could accidentally start multiple by running `launch.sh` multiple times. To connect to the tmux session, run `tmux a -t <session_id>` (you can ommit the `-t <session_id>` if there is only one session). To detach from the session, press `Ctrl-b d`.

Once inside the tmux session, you should see your command running. If your command crashed immediately, you can see the error message, and you have 5 minutes until the VM shuts down. To cancel this shutdown, do `Ctrl-b :kill-pane`. **However, you must either then shut down the VM yourself or run `launch.sh` again to launch another job that will automatically shut down after completion. Do not leave your instance running!!!**

### 5. Launch TensorBoard

You can run TensorBoard on the VM by running `google_cloud/launch_tensorboard.sh`. You should then be able to view TensorBoard by going to `http://localhost:6006` on your local machine. This only lasts as long as you keep the script running.

### 6. Sync data to your local machine

After a job has completed, you can sync the experiment output back to your local machine by running `google_cloud/sync_data.sh`. This starts the instance, downloads the remote `data/` directory to your local `data/` directory, and then shuts it down again.


## Important notes

- **First and foremost, do not leave your instance running! You will be charged for every hour that it is running, even if you are not using it. If you leave your instance running and run out of credits, you will be charged for further usage.**
- Remember to run all the scripts from the `hw2` directory.
- You probably only need the instance for big sweeps of hyperparameters. Running a single Python command will be fairly wasteful, and also probably faster on your local machine (even if you don't have a GPU!). Instead, use the instance for running multiple jobs at once that you've defined in a shell script: for example, `google_cloud/launch.sh bash sweep_lambda.sh`. For running multiple jobs at once, you can use subshells (i.e. `&`) or [GNU parallel](https://www.gnu.org/software/parallel/) (which is pre-installed on the VM). If you use subshells, make sure to put `wait` at the end of your shell script or the VM will shut down without waiting for your jobs to finish.