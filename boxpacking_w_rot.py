import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum

#HEIGHTS = [1,2,3,4]
#WIDTHS = [2,3,1,4]

#HEIGHTS = [1,1,1,1,1]
#WIDTHS = [3,3,3,3,3]

HEIGHTS = [1,2,3,4,5,1,2]
WIDTHS = [2,3,1,4,5,3,4]

BOXES = range(len(HEIGHTS))

MAXHEIGHT = 10
MAXWIDTH = 10
MINHEIGHT = 0
MINWIDTH = 0 

BigM = 100000

model = gp.Model("BoxPacking")

BxB = [(i,j) for i in BOXES for j in BOXES if i < j]

####################      VARIABLES        #######################

x = model.addVars(BOXES, ub = [MAXWIDTH - WIDTHS[i] for i in BOXES])
y = model.addVars(BOXES, ub = [MAXHEIGHT - HEIGHTS[i] for i in BOXES])
z = model.addVars(BxB, vtype = GRB.BINARY)
w = model.addVars(BxB, vtype = GRB.BINARY)
v = model.addVars(BxB, vtype = GRB.BINARY)

r = model.addVars(BOXES, vtype = GRB.BINARY)

Hmax = model.addVar()
Wmax = model.addVar()

#####################     CONSTRAINTS       #######################


model.addConstrs(x[i] + (WIDTHS[i]*r[i] + HEIGHTS[i]*(1-r[i]))<= x[j] + z[i,j] * BigM + BigM * v[i,j] for (i,j) in BxB)
model.addConstrs(x[j] + (WIDTHS[j]*r[j] + HEIGHTS[j]*(1-r[j])) <= x[i] + (1-z[i,j]) * BigM + BigM * v[i,j] for (i,j) in BxB)

model.addConstrs(y[i] + (WIDTHS[i]*(1-r[i]) + HEIGHTS[i]*r[i]) <= y[j] + w[i,j] * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)
model.addConstrs(y[j] + (WIDTHS[j]*(1-r[j]) + HEIGHTS[j]*r[j]) <= y[i] + (1-w[i,j]) * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)

model.addConstrs(y[i] + (WIDTHS[i]*(1-r[i]) + HEIGHTS[i]*r[i]) <= Hmax for i in BOXES)
model.addConstrs(x[i] + (WIDTHS[i]*r[i] + HEIGHTS[i]*(1-r[i])) <= Wmax for i in BOXES)

######################        MODEL OBJECTIVE         ##################

model.setObjective(Wmax*Hmax, GRB.MINIMIZE)
model.optimize()


######################       DRAWING UP THE SOLUTION    ##################

# Check if the model has a feasible solution
if model.Status == GRB.OPTIMAL:
    # Extracting the final values of x and y variables
    x_values = model.getAttr('X', x)
    y_values = model.getAttr('X', y)
    H_value = Hmax.X
    W_value = Wmax.X
    r_value = model.getAttr('X', r)
else:
    print("Model did not solve to optimality. The status code is:", model.Status)

"""for i in BOXES:
    rotated_width, rotated_height = (HEIGHTS[i], WIDTHS[i]) if r_value[i] == 1 else (WIDTHS[i], HEIGHTS[i])"""

plt.figure(figsize=(8, 8))  # Adjusted for square proportions to reflect actual packing dimensions accurately
plt.scatter(list(x_values.values()), list(y_values.values()))
for i, (x_val, y_val) in enumerate(zip(x_values.values(), y_values.values())):
    rotated_width, rotated_height = (HEIGHTS[i], WIDTHS[i]) if r_value[i] == 0 else (WIDTHS[i], HEIGHTS[i])
    plt.gca().add_patch(plt.Rectangle((x_val, y_val), rotated_width, rotated_height, edgecolor='blue', facecolor='none', linewidth=2))   
    plt.text(x_val + rotated_width/2, y_val + rotated_height/2, f"{i}, {int(r_value[i])}, {rotated_width}x{rotated_height}", ha='center', va='center', color='red')
    if i == 0:
        plt.plot([], [], color='blue', label='Packed Items', linewidth=2)  # Add a custom legend entry

plt.title('Final Positions of Packed Items')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

plt.xlim(-2, W_value + 2)  # Adjust x-axis limits to reflect the actual width boundary
plt.ylim(-2, H_value + 2)  # Adjust y-axis limits to reflect the actual height boundary
plt.savefig("images/bp_v2.3.png")