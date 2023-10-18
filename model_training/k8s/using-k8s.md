# Directions for interacting with k8s cluster

## Prerequisites
### Get AWS CLI
    - download from amazon and install

        aws configure
### Install eksctl
        brew tap weaveworks/tap
        brew install weaveworks/tap/eksctl
        eksctl version


### Creating the Cluster

        eksctl create cluster \
          --name ModelTraining \
          --version 1.28 \
          --region us-east-2 \
          --nodegroup-name linux-nodes \
          --node-type t2.micro \
          --nodes 2


### Delete Cluster When Done

        eksctl delete cluster --name ModelTraining


## Build Docker Image and Add to AWS

### Build Docker Image

        docker build -t model-training .
        docker run -it model-training
        
### Push to AWS
        aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 544064040421.dkr.ecr.us-east-2.amazonaws.com
        docker build -t w210-second-sight-model-training .
        docker tag w210-second-sight-model-training:latest public.ecr.aws/e9o7l9v1/w210-second-sight-model-training:latest
        docker push public.ecr.aws/e9o7l9v1/w210-second-sight-model-training:latest

## Deploy to Kubernetes
