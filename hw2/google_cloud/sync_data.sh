#! /bin/bash

# Syncs the `data` directory, which contains the results of your training runs, from the remote instance to your local
# machine.

export ZONE="us-west4-a"
export INSTANCE_NAME="cs285"

echo "Starting instance..."
gcloud compute instances start $INSTANCE_NAME --zone=$ZONE

echo "-------------------------------------"
echo "Waiting for instance to boot..."

while true; do
  output=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="nvidia-smi" 2>&1)

  if [ $? -eq 0 ]; then
    break
  else
    sleep 1
  fi
done

echo "-------------------------------------"
echo "Transferring files..."

IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE | grep natIP | cut -d : -f 2)

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine" $IP:~/data ./data

echo "-------------------------------------"
echo "Shutting down..."
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE