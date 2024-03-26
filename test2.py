import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum

HEIGHTS = [1,2,3,4,5,1,2,3,4]
WIDTHS = [2,3,1,4,5,3,4,5,1]

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

Hmax = model.addVar()
Wmax = model.addVar()

#####################     CONSTRAINTS       #######################

model.addConstrs(x[i] + WIDTHS[i] <= x[j] + z[i,j] * BigM + BigM * v[i,j] for (i,j) in BxB)
model.addConstrs(x[j] + WIDTHS[j] <= x[i] + (1-z[i,j]) * BigM + BigM * v[i,j] for (i,j) in BxB)

model.addConstrs(y[i] + HEIGHTS[i] <= y[j] + w[i,j] * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)
model.addConstrs(y[j] + HEIGHTS[j] <= y[i] + (1-w[i,j]) * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)

model.addConstrs(y[i] + HEIGHTS[i] <= Hmax for i in BOXES)
model.addConstrs(x[i] + WIDTHS[i] <= Wmax for i in BOXES)

######################        MODEL OBJECTIVE         ##################

model.setObjective(Wmax+Hmax, GRB.MINIMIZE)
model.optimize()


######################       DRAWING UP THE SOLUTION    ##################


# Check if the model has a feasible solution
if model.Status == GRB.OPTIMAL:
    # Extracting the final values of x and y variables
    x_values = model.getAttr('X', x)
    y_values = model.getAttr('X', y)
else:
    print("Model did not solve to optimality. The status code is:", model.Status)


plt.figure(figsize=(8, 8))  # Adjusted for square proportions to reflect actual packing dimensions accurately
plt.scatter(list(x_values.values()), list(y_values.values()))
for i, (x_val, y_val) in enumerate(zip(x_values.values(), y_values.values())):
    plt.gca().add_patch(plt.Rectangle((x_val, y_val), WIDTHS[i], HEIGHTS[i], edgecolor='blue', facecolor='none', linewidth=2))
    plt.text(x_val + WIDTHS[i]/2, y_val + HEIGHTS[i]/2, str(i), ha='center', va='center', color='red')
    if i == 0:
        plt.plot([], [], color='blue', label='Packed Items', linewidth=2)  # Add a custom legend entry

plt.title('Final Positions of Packed Items')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()
plt.grid(True)
plt.axhline(MINHEIGHT, color='black', linewidth=1)
plt.axhline(MAXHEIGHT, color='black', linewidth=1)
plt.axvline(MINWIDTH, color='black', linewidth=1)
plt.axvline(MAXWIDTH, color='black', linewidth=1)
plt.xlim(0, MAXWIDTH)  # Adjust x-axis limits to reflect the actual width boundary
plt.ylim(0, MAXHEIGHT)  # Adjust y-axis limits to reflect the actual height boundary
plt.savefig("images/bp_v0.png")
