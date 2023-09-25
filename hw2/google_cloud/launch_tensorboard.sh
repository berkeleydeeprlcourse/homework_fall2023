#! /bin/bash

# Launches tensorboard on the remote instance, forwarding the port to your local machine.

export ZONE="us-west3-b"
export INSTANCE_NAME="cs285"

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='sudo pkill -f tensorboard'
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='bash -lc "tensorboard --logdir data --port 6006"' --ssh-flag="-L 6006:localhost:6006"
