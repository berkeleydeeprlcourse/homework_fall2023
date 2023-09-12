#! /bin/bash

# Creates a new GPU instance, transfers the code to it, and runs some installation steps.

export ZONE="us-west1-b"
export INSTANCE_NAME="cs285"

echo "Creating instance..."

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=pytorch-1-13-cu113-debian-11-py310 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"

echo "-------------------------------------"
echo "Waiting for NVIDIA driver install..."

while true; do
  output=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="nvidia-smi" 2>&1)

  if [ $? -eq 0 ]; then
    echo $output
    break
  else
    sleep 10
  fi
done

echo "-------------------------------------"
echo "Transferring files to instance..."

IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE | grep natIP | cut -d : -f 2)

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine" . $IP:

echo "-------------------------------------"
echo "Running setup..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='
  sudo apt install -y swig python3-dev parallel
  echo "export MUJOCO_GL=egl" >> ~/.bashrc
  /opt/conda/bin/conda init bash
  bash -lic "pip install -r requirements.txt"
  bash -lic "pip install -e ." \
'

echo "-------------------------------------"
echo "Shutting down..."
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE

