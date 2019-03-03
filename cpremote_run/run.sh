#!/usr/bin/bash

sudo apt-get update
sudo apt-get -y install python3-pip
pip3 install -r requirements.txt
sudo apt-get -y install libsm6 libxrender1 libfontconfig1
