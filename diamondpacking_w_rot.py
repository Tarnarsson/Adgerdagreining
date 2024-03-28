import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum

HEIGHTS = [1,2,3,4,5,1,2,3,4,5]
WIDTHS = [2,3,1,4,5,3,4,5,5,1]

#HEIGHTS = [1,2,3,1,2,3]
#WIDTHS = [1,1,1,2,2,2]

BOXES = range(len(HEIGHTS))

model = gp.Model("BoxPacking")

BigM = 10000

BxB = [(i,j) for i in BOXES for j in BOXES if i < j]

x = model.addVars(BOXES, name="x")
y = model.addVars(BOXES, name="y")

z = model.addVars(BxB, vtype = GRB.BINARY)
w = model.addVars(BxB, vtype = GRB.BINARY)
v = model.addVars(BxB, vtype = GRB.BINARY)

r = model.addVars(BOXES, vtype = GRB.BINARY)

H = model.addVar(name="H", lb = 15)
W = model.addVar(name="W", lb = 15)

#making sure each corner is within the boundry of the canvas
model.addConstrs((y[b] >= -x[b] + W/(2**.5) for b in BOXES), "B0") #BottomLeftAboveLine   #TODO breyta x til þess að representa rétt horn mv línuna sem er takmarkandi.

model.addConstrs((y[b] >= x[b] + (WIDTHS[b]*r[b] + HEIGHTS[b]*(1-r[b])) - W/(2**.5)  for b in BOXES), "B1") #BottomRightAboveLine     #+ WIDTHS[b]

model.addConstrs((y[b] + (WIDTHS[b]*(1-r[b]) + HEIGHTS[b]*r[b]) <= -x[b] - (WIDTHS[b]*r[b] + HEIGHTS[b]*(1-r[b])) + W/(2**.5) + 2*H/(2**.5) for b in BOXES), "B2") #TopRightBelowLine  #+ WIDTHS[b] + HEIGHTS[b]

model.addConstrs((y[b] + (WIDTHS[b]*(1-r[b]) + HEIGHTS[b]*r[b]) <= x[b] +  W/(2**.5) for b in BOXES), "B3")  #TopLeftBelowLine   #+ HEIGHTS[b]

#####################     CONSTRAINTS       ####################### Þarf að breyta til þess að nota rotation #TODO

model.addConstrs(x[i] + (WIDTHS[i]*r[i] + HEIGHTS[i]*(1-r[i])) <= x[j] + z[i,j] * BigM + BigM * v[i,j] for (i,j) in BxB)
model.addConstrs(x[j] + (WIDTHS[j]*r[j] + HEIGHTS[j]*(1-r[j])) <= x[i] + (1-z[i,j]) * BigM + BigM * v[i,j] for (i,j) in BxB)

model.addConstrs(y[i] + (WIDTHS[i]*(1-r[i]) + HEIGHTS[i]*r[i]) <= y[j] + w[i,j] * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)
model.addConstrs(y[j] + (WIDTHS[j]*(1-r[j]) + HEIGHTS[j]*r[j]) <= y[i] + (1-w[i,j]) * BigM + (1-v[i,j]) * BigM for (i,j) in BxB)

######################        MODEL OBJECTIVE         ##################
model.setObjective(quicksum(x[i]+y[i] for i in BOXES), GRB.MINIMIZE) 
model.optimize()

# Check if the model has a feasible solution
if model.Status == GRB.OPTIMAL:
    # Extracting the final values of x and y variables
    x_values = model.getAttr('X', x)
    y_values = model.getAttr('X', y)
    r_value = model.getAttr('X', r)
    H_value = H.X
    W_value = W.X
else:
    print("Model did not solve to optimality. The status code is:", model.Status)

#################    PLOTTING BOXES    ###################

plt.figure(figsize=(10, 10))  # Adjusted for square proportions to reflect actual packing dimensions accurately
plt.scatter(list(x_values.values()), list(y_values.values()))
for i, (x_val, y_val) in enumerate(zip(x_values.values(), y_values.values())):
    rotated_width, rotated_height = (HEIGHTS[i], WIDTHS[i]) if r_value[i] == 0 else (WIDTHS[i], HEIGHTS[i])
    plt.gca().add_patch(plt.Rectangle((x_val, y_val), rotated_width, rotated_height, edgecolor='blue', facecolor='none', linewidth=2))   
    plt.text(x_val + rotated_width/2, y_val + rotated_height/2, f"{i}, {int(r_value[i])}, {rotated_width}x{rotated_height}", ha='center', va='center', color='red')


################   BORDER LINES    ###################

# Drawing the line y = -x + W/sqrt(2)
x0_vals = np.array([0, W_value/(2**.5)])
y0_vals = -x0_vals + W_value/(2**.5)
plt.plot(x0_vals, y0_vals, color='red', linewidth=2)
mid_point_x0 = np.mean(x0_vals)
mid_point_y0 = np.mean(y0_vals)
plt.text(mid_point_x0-1, mid_point_y0-1, 'B0', color='red', horizontalalignment='center', fontsize=20)


# Drawing the line y = x - 6
x1_vals = np.array([W_value/(2**.5) , W_value/(2**.5) + H_value/(2**.5)])
y1_vals = x1_vals - W_value/(2**.5) 
plt.plot(x1_vals, y1_vals, color='red', linewidth=2)
mid_point_x0 = np.mean(x1_vals)
mid_point_y0 = np.mean(y1_vals)
plt.text(mid_point_x0+1, mid_point_y0-1, 'B1', color='red', horizontalalignment='center', fontsize=20)

x2_vals = np.array([W_value/(2**.5), W_value/(2**.5) + H_value/(2**.5)])
y2_vals = -x2_vals + W_value/(2**.5) + 2*H_value/(2**.5)
plt.plot(x2_vals, y2_vals, color='red', linewidth=2)
mid_point_x0 = np.mean(x2_vals)
mid_point_y0 = np.mean(y2_vals)
plt.text(mid_point_x0+1, mid_point_y0+1, 'B2', color='red', horizontalalignment='center', fontsize=20)

# Drawing the line y = x + 6
x3_vals = np.array([0, W_value/(2**.5)])
y3_vals = x3_vals + W_value/(2**.5)
plt.plot(x3_vals, y3_vals, color='red', linewidth=2)
mid_point_x0 = np.mean(x3_vals)
mid_point_y0 = np.mean(y3_vals)
plt.text(mid_point_x0-1, mid_point_y0+1, 'B3', color='red', horizontalalignment='center', fontsize=20)

################   ORIGIN AND AXIS   ###################

# Drawing x and y axis for better visualization
plt.axhline(0, color='black', linewidth=2)  # Y-axis
plt.axvline(0, color='black', linewidth=2)  # X-axis

###################   AESTEHTICS AND SHOW  ###################
plt.xlim(-2, W_value/(2**.5) + H_value/(2**.5) + 2)
plt.ylim(-2, W_value/(2**.5) + H_value/(2**.5) + 2)
plt.title('Pakkaðir kassar undir 45° horni', fontsize=28)
plt.xlabel('X', fontsize=20)
plt.ylabel('Y', fontsize=20)


plt.legend()
plt.grid(True)

plt.savefig("images/dp_v2.jpg")

#plt.show()
