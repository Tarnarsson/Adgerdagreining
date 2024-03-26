import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum

HEIGHTS = [1,2,3,4,5]
WIDTHS = [2,3,1,4,5]

#HEIGHTS = [1,2,3,1,2,3]
#WIDTHS = [1,1,1,2,2,2]

BOXES = range(len(HEIGHTS))

model = gp.Model("BoxPacking")

BigM = 10000

BxB = [(i,j) for i in BOXES for j in BOXES if i < j]

# Assuming BOXES is a range or list of indices for your rectangles
# And assuming WIDTHS and HEIGHTS are lists of widths and heights for these rectangles

# Define variables for the x and y coordinates of each corner of each rectangle
# Let's redefine corners such that 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left
corners = range(4)  # Redefining corners with 0 as bottom-left
x = model.addVars(BOXES, corners, name="x")
y = model.addVars(BOXES, corners, name="y")

Hmax = model.addVar(name="Hmax")
Wmax = model.addVar(name="Wmax")

#Boundry for box b from the bottom left point.
model.addConstrs((x[b, 1] == x[b, 0] + WIDTHS[b] for b in BOXES), "WidthConstraint")
model.addConstrs((y[b, 3] == y[b, 0] + HEIGHTS[b] for b in BOXES), "HeightConstraint")
model.addConstrs((x[b, 2] == x[b, 3] + WIDTHS[b] for b in BOXES), "TopRightXConstraint")
model.addConstrs((y[b, 2] == y[b, 1] + HEIGHTS[b] for b in BOXES), "TopRightYConstraint")

#making sure each corner is within the boundry of the canvas
model.addConstrs((y[b, 0] >= -x[b, 0] + 6 for b in BOXES), "BottomLeftAboveLine1") 
#model.addConstrs((x[b, 0] >= -y[b, 0] + 6 for b in BOXES), "BottomLeftAboveLine2") 

model.addConstrs((y[b, 1] >= x[b, 1] - 6 for b in BOXES), "BottomRightAboveLine1")
#model.addConstrs((-x[b, 1] >= -y[b, 1] - 6 for b in BOXES), "BottomRightAboveLine2")

model.addConstrs((y[b, 2] <= -x[b, 2] + Hmax for b in BOXES), "TopRightBelowLine1")
#model.addConstrs((x[b, 2] <= -y[b, 2] + Wmax for b in BOXES), "TopRightBelowLine2")

model.addConstrs((y[b, 3] <= x[b, 3] + 6 for b in BOXES), "TopLeftBelowLine1")
#model.addConstrs((-x[b, 3] <= -y[b, 3] + 6 for b in BOXES), "TopLeftBelowLine2")

z = model.addVars(BxB, vtype = GRB.BINARY)
w = model.addVars(BxB, vtype = GRB.BINARY)
v = model.addVars(BxB, vtype = GRB.BINARY)


#####################     CONSTRAINTS       #######################

model.addConstrs(x[i ,0] + WIDTHS[i] <= x[j ,0] + z[i,j] * BigM + BigM * v[i,j] for (i,j) in BxB)
model.addConstrs(x[j ,0] + WIDTHS[j] <= x[i ,0] + (1-z[i,j]) * BigM + BigM * v[i,j] for (i,j) in BxB)

model.addConstrs(y[i ,0] + HEIGHTS[i] <= y[j ,0] + w[i,j] * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)
model.addConstrs(y[j ,0] + HEIGHTS[j] <= y[i ,0] + (1-w[i,j]) * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)


######################        MODEL OBJECTIVE         ##################
model.setObjective(Hmax, GRB.MINIMIZE) # quicksum(y[b,2]+x[b,2] for b in BOXES)
model.optimize()



# Check if the model has a feasible solution
if model.Status == GRB.OPTIMAL:
    # Extracting the final values of x and y variables
    x_values = model.getAttr('X', x)
    y_values = model.getAttr('X', y)
    Hmax_value = Hmax.X
else:
    print("Model did not solve to optimality. The status code is:", model.Status)



plt.figure(figsize=(10, 10))  # Adjusted for square proportions to reflect actual packing dimensions accurately
plt.gca().set_aspect('equal', adjustable='box')  # Ensuring grid axis are on the same scale
for b in BOXES:
    plt.gca().add_patch(plt.Rectangle((x_values[b, 0], y_values[b, 0]), WIDTHS[b], HEIGHTS[b], edgecolor='blue', facecolor='none', linewidth=2))
    if b == BOXES[0]:
        plt.plot([], [], color='blue', label='Packed Items', linewidth=2)  # Add a custom legend entry

################   BORDER LINES    ###################

# Drawing the line y = -x + 6
x0_vals = np.array([0, 6])
y0_vals = -x0_vals + 6
plt.plot(x0_vals, y0_vals, label='y = -x + 6', color='red', linewidth=2)

# Drawing the line y = x - 6
x1_vals = np.array([6, 15])
y1_vals = x1_vals - 6
plt.plot(x1_vals, y1_vals, label='y = x - 6', color='red', linewidth=2)

# Drawing the line y = x + 6
x3_vals = np.array([0, 9])
y3_vals = x3_vals + 6
plt.plot(x3_vals, y3_vals, label='y = x + 6', color='red', linewidth=2)

x2_vals = np.array([0, 9])
y2_vals = -x2_vals + Hmax_value
plt.plot(x2_vals, y2_vals, label='y = -x + Hmax', color='green', linewidth=2)

################   ORIGIN AND AXIS   ###################

# Marking the origin for clarity
plt.scatter([0], [0], color='green', label='Origin (0,0)', zorder=5)

# Drawing x and y axis for better visualization
plt.axhline(0, color='black', linewidth=1)  # Y-axis
plt.axvline(0, color='black', linewidth=1)  # X-axis

###################   AESTEHTICS AND SHOW  ###################

plt.title('Final Positions of Packed Items with Constraint Line')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.show()
