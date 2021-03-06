#!/bin/bash

cd ~
# Druid setup
wget http://static.druid.io/artifacts/releases/druid-0.12.0-bin.tar.gz
tar -xzf druid-0.12.0-bin.tar.gz
rm druid-0.12.0-bin.tar.gz
~/druid-0.12.0/bin/init

# Zookeeper setup
wget http://ftp.unicamp.br/pub/apache/zookeeper/current/zookeeper-3.4.10.tar.gz
tar -xzf zookeeper-3.4.10.tar.gz
rm zookeeper-3.4.10.tar.gz
mv zookeeper-3.4.10/conf/zoo_sample.cfg zookeeper-3.4.10/conf/zoo.cfg

# Update Java
sudo yum install java-1.8.0
sudo yum remove java-1.7.0-openjdk

# Superset dependencies
sudo yum upgrade python-setuptools
sudo yum install gcc gcc-c++ libffi-devel python-devel python-pip python-wheel openssl-devel libsasl2-devel openldap-devel
pip install --upgrade setuptools pip

# Install superset
pip install superset==0.23.0

# Create an admin user (you will be prompted to set username, first and last name before setting a password)
fabmanager create-admin --app superset

# Initialize the database
superset db upgrade

# Load some data to play with
superset load_examples

# Create default roles and permissions
superset init

# Start the web server on port 8088, use -p to bind to another port
superset runserver

# To start a development web server, use the -d switch
# superset runserver -d