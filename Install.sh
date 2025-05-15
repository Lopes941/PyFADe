#!/bin/bash


py="/c/Users/Heitor/anaconda3/python"

$py setup.py clean --all
$py ./setup.py build_ext --inplace
#$py setup.py sdist bdist_wheel

pip install .