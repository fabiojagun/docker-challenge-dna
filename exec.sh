#!/bin/bash
docker build -t challenge-dna .
docker run --name chocobo challenge-dna
docker cp chocobo:/data/data-model-docker.csv .
docker rm chocobo