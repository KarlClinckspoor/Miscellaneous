# Miscellaneous
Miscellaneous small projects I've been developing to make my data treatment life easier.

The scripts contained here are:

## AFM adhesion forces.py

This script reads plaintext force-distance data. First, it reads data that has the X values (distance) in the fourth column (separated by tabs), and y (the force) in the third column. It rewrites the files to a second format that is easier for independent plotting afterwards. Then, it finds the deltaY, related to the adhesive force.

This force appears when the microscopy tip is going backwards (to the right, on the graphs). The first value is at the minimum of the graph, where there is the maximum downwards force. The second value (to calculate DeltaY), is when the tip detaches from the surface of the sample, and is indicated by an inflection point. Afterwards, the curve is perfectly flat. The script finds this second point by doing fits of all the points from the minimum to the last, rightmost point. It starts at the minimum and goes to the right. When the fit gets very good, it has detached, so the program knows this is the second point to be considered. Then, it just calculates the deltaY.

A very simple script that can be easily adapted for other uses and saves a lot of time.

## Electrostatic.py, Hamaker.py

Nigh useless scripts to calculate the Hamaker constant and an unfinished script to calculate electrostatic forces.


