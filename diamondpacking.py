import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

HEIGHTS = [1,2,3,4,5,1,2,3]
WIDTHS = [2,3,1,4,5,3,4,5]

#HEIGHTS = [1,2,3,1,2,3]
#WIDTHS = [1,1,1,2,2,2]

BOXES = range(len(HEIGHTS))

model = gp.Model("BoxPacking")

BigM = 10000

BxB = [(i,j) for i in BOXES for j in BOXES if i < j]

x = model.addVars(BOXES, name="x")
y = model.addVars(BOXES, name="y")

H = model.addVar(name="H", lb = 15)
W = model.addVar(name="W", lb = 15)

"""#Boundry for box b from the bottom left point.
model.addConstrs((x[b] == x[b] + WIDTHS[b] for b in BOXES), "WidthConstraint")
model.addConstrs((y[b, 3] == y[b, 0] + HEIGHTS[b] for b in BOXES), "HeightConstraint")
model.addConstrs((x[b, 2] == x[b, 3] + WIDTHS[b] for b in BOXES), "TopRightXConstraint")
model.addConstrs((y[b, 2] == y[b, 1] + HEIGHTS[b] for b in BOXES), "TopRightYConstraint")"""

#making sure each corner is within the boundry of the canvas
model.addConstrs((y[b] >= -x[b] + W/(2**.5) for b in BOXES), "BottomLeftAboveLine1") #TODO breyta x til þess að representa rétt horn mv línuna sem er takmarkandi.

model.addConstrs((y[b] >= x[b] + WIDTHS[b] - W/(2**.5)  for b in BOXES), "BottomRightAboveLine1") #+ WIDTHS[b]

model.addConstrs((y[b] + HEIGHTS[b] <= -x[b] - WIDTHS[b] + W/(2**.5) + 2*H/(2**.5) for b in BOXES), "TopRightBelowLine1") #+ WIDTHS[b] + HEIGHTS[b]

model.addConstrs((y[b] + HEIGHTS[b] <= x[b] +  W/(2**.5) for b in BOXES), "TopLeftBelowLine1") #+ HEIGHTS[b]


z = model.addVars(BxB, vtype = GRB.BINARY)
w = model.addVars(BxB, vtype = GRB.BINARY)
v = model.addVars(BxB, vtype = GRB.BINARY)



#####################     CONSTRAINTS       ####################### Þarf að breyta til þess að nota rotation #TODO

model.addConstrs(x[i] + WIDTHS[i] <= x[j] + z[i,j] * BigM + BigM * v[i,j] for (i,j) in BxB)
model.addConstrs(x[j] + WIDTHS[j] <= x[i] + (1-z[i,j]) * BigM + BigM * v[i,j] for (i,j) in BxB)

model.addConstrs(y[i] + HEIGHTS[i] <= y[j] + w[i,j] * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)
model.addConstrs(y[j] + HEIGHTS[j] <= y[i] + (1-w[i,j]) * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)


######################        MODEL OBJECTIVE         ##################
model.setObjective(W*H, GRB.MINIMIZE) 
model.optimize()



# Check if the model has a feasible solution
if model.Status == GRB.OPTIMAL:
    # Extracting the final values of x and y variables
    x_values = model.getAttr('X', x)
    y_values = model.getAttr('X', y)
    H_value = H.X
    W_value = W.X
else:
    print("Model did not solve to optimality. The status code is:", model.Status)

#################    PLOTTING BOXES    ###################

plt.figure(figsize=(10, 10))  # Adjusted for square proportions to reflect actual packing dimensions accurately
plt.gca().set_aspect('equal', adjustable='box')  # Ensuring grid axis are on the same scale
for i, (x_val, y_val) in enumerate(zip(x_values.values(), y_values.values())):
    #rotated_width, rotated_height = (HEIGHTS[i], WIDTHS[i]) if r_value[i] == 0 else (WIDTHS[i], HEIGHTS[i])
    plt.gca().add_patch(plt.Rectangle((x_val, y_val), WIDTHS[i], HEIGHTS[i], edgecolor='blue', facecolor='none', linewidth=2))   
    plt.text(x_val + WIDTHS[i]/2, y_val + HEIGHTS[i]/2, f"{i}, {WIDTHS[i]}x{HEIGHTS[i]}", ha='center', va='center', color='red')
    if i == 0:
        plt.plot([], [], color='blue', label='Packed Items', linewidth=2)  # Add a custom legend entry

################   BORDER LINES    ###################

# Drawing the line y = -x + W/sqrt(2)
x0_vals = np.array([0, W_value/(2**.5)])#W_value/(2**.5)
y0_vals = -x0_vals + W_value/(2**.5)
plt.plot(x0_vals, y0_vals, label='y = -x + W/sqrt(2)', color='red', linewidth=2)

# Drawing the line y = x - 6
x1_vals = np.array([W_value/(2**.5) , W_value/(2**.5) + H_value/(2**.5)])
y1_vals = x1_vals - W_value/(2**.5) 
plt.plot(x1_vals, y1_vals, label='y = x - W/sqrt(2) - 2*H/sqrt(2)', color='red', linewidth=2)

x2_vals = np.array([W_value/(2**.5), W_value/(2**.5) + H_value/(2**.5)])
y2_vals = -x2_vals + W_value/(2**.5) + 2*H_value/(2**.5)
plt.plot(x2_vals, y2_vals, label='y = -x + W/sqrt(2) + 2*H/sqrt(2)', color='red', linewidth=2)

# Drawing the line y = x + 6
x3_vals = np.array([0, W_value/(2**.5)])
y3_vals = x3_vals + W_value/(2**.5)
plt.plot(x3_vals, y3_vals, label='y = x + W/sqrt(2)', color='red', linewidth=2)


################   ORIGIN AND AXIS   ###################

# Marking the origin for clarity
plt.scatter([0], [0], color='green', label='Origin (0,0)', zorder=5)

# Drawing x and y axis for better visualization
plt.axhline(0, color='black', linewidth=1)  # Y-axis
plt.axvline(0, color='black', linewidth=1)  # X-axis

###################   AESTEHTICS AND SHOW  ###################
plt.xlim(-2, W_value/(2**.5) + 2*H_value/(2**.5) + 2)
plt.ylim(-2, W_value/(2**.5) + H_value/(2**.5) + 2)
plt.title('Final Positions of Packed Items with Constraint Line')
plt.xlabel('X Values')
plt.ylabel('Y Values')


plt.legend()
plt.grid(True)

plt.savefig("images/dp_v0.jpg")

#plt.show()
