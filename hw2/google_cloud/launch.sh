#! /bin/bash

# Launches a command on the remote instance. Ensures that your code is synchronized to the remote instance beforehand.

export ZONE="us-west1-b"
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
echo "Transferring files to instance..."

IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE | grep natIP | cut -d : -f 2)

rsync -av --progress -e "ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine" . $IP:

echo "-------------------------------------"
echo "Running command..."

CMD="
  tmux new -d ' 
    bash -lic '\''$*'\''
    sleep 5m 
    sudo shutdown now'
"

echo $CMD

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="$CMD"
