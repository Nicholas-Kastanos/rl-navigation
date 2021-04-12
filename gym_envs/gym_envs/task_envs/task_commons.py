#!/usr/bin/env python
import rosparam
import os

def LoadYamlFileParams(path_config_file):
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)
        