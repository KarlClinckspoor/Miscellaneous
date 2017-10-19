# Miscellaneous
Miscellaneous small projects I've been developing to make my data treatment life easier.

The scripts contained here are:

## AFM adhesion forces.py

This script reads plaintext force-distance data. First, it reads data that has the X values (distance) in the fourth column (separated by tabs), and y (the force) in the third column. It rewrites the files to a second format that is easier for independent plotting afterwards. Then, it finds the deltaY, related to the adhesive force.

This force appears when the microscopy tip is going backwards (to the right, on the graphs). The first value is at the minimum of the graph, where there is the maximum downwards force. The second value (to calculate DeltaY), is when the tip detaches from the surface of the sample, and is indicated by an inflection point. Afterwards, the curve is perfectly flat. The script finds this second point by doing fits of all the points from the minimum to the last, rightmost point. It starts at the minimum and goes to the right. When the fit gets very good, it has detached, so the program knows this is the second point to be considered. Then, it just calculates the deltaY.

A very simple script that can be easily adapted for other uses and saves a lot of time.

## Electrostatic.py, Hamaker.py

Nigh useless scripts to calculate the Hamaker constant and an unfinished script to calculate electrostatic forces.

## RheoFC.py

A script that calculates the zero-shear viscosity of flow curves obtained from a rheometer automatically, and saves the figures of each plot and fit. The data is required to be in plaintext, and needs to be separated by ';'. This script was specifically written for the ASCII exports from the RheoWin software from Thermo Scientific (Haake). The data reading function is specific to this type of file. Two sample curves are included with the script for comparison.

The fitting can be done automatically, or manually. Both are based on fitting an horizontal curve at low shear rates and getting the intercept as the value for the zero shear viscosity.

For the manual fitting, it is recommended to use Spyder, or some ipython interpreter that supports inline plotting of graphs. Manual fitting requires you to input which points are going to be considered for the linear fit. It then saves a figure with the fit and saves the intercept + error value on a file named results.dat.

Automatic fitting chooses two points and tries to fit them. It then varies the chosen points, compares the error from the fits and chooses the one with the smallest error.

The points are chosen like this:
* First point: Point 0 up to point length // 3 (// means floor division, to remove any decimal places)
* Second point: necessarily 4 points to the right of the first (or else it would always fit two adjacent points), and goes to the right up to length // 2 (the middle of the curve).
* The script goes through all the second points available before changing the first point.

These parameters can easily be changed to better suit the data to be treated.
