# 3DiStress
3D Finite Element simulation for geological stress modelling

Please install the required packages specified in "requirements.txt" to run this program.

This program takes the discritised geometry (mesh) of a coal seam, apply appropriate boundary conditions and obtain the stress distribution across the whole domain using Finite Element Analysis. 

In "src" directory you will find sub-directory "data" where the required input data is stored in .txt format, the next sub-directory is called "test" where you can find .py files which are different modules imported in the main program body which is written in "Runner.py" file.

After setting the pythonpath to the directory you have your modules in by " export PYTHONPATH=${PWD} ", you can run "Runner.py" file.
