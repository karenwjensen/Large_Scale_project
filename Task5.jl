#### MILP MODEL ### 

##### 
# Parameters 
#####
# Hej karen

# holding cost 
h = [1 1]
# setup cost
c = [1 2]
# Capacity 
C = 10
# demand 
demand = [3 4 5; 0 5 2]
# setup time 
q = [2 1]

# M-parameters 
M1 = M2 = M3 = max(maximum(demand[1,:]),maximum(demand[2,:]))
num_products = 2 
num_periods = 3
########################
########################

# Model - IP 
using JuMP
using Gurobi 

model = Model(Gurobi.Optimizer)

@variable(model, y[1:num_products, 1:num_periods], Bin)         # setup indicator
@variable(model, 0 <= x[1:num_products, 1:num_periods] <= M2, Int)   # production qty
@variable(model, 0 <= s[1:num_products, 1:num_periods] <= M3, Int)   # end-period inventory

# Objective
@objective(model, Min, sum(c[i] * y[i,t] + h[i] * s[i,t] for i in 1:num_products, t in 1:num_periods))

# Inventory‐balance and setup constraints
# first period: x_i1 = d_i1 + s_i1
@constraint(model,[i in 1:num_products], x[i,1] == demand[i,1] + s[i,1])
# second period: t ≥ 2: s_{i,t-1} + x_{i,t} = d_{i,t} + s_{i,t}
@constraint(model,[i in 1:num_products, t in 2:num_periods], s[i,t-1] + x[i,t] == demand[i,t] + s[i,t])
# linking production to setup
@constraint(model,[i in 1:num_products, t in 1:num_periods], x[i,t] <= M1 * y[i,t])
# capacity constraint 
@constraint(model,[t in 1:num_periods], sum(x[i,t] for i in 1:num_products) + sum(q[i] * y[i,t] for i in 1:num_products) <= C)

######################
######################
######################

# Model - LP relaxation 
model_LP = Model(Gurobi.Optimizer)

@variable(model_LP, 0 <= y_LP[1:num_products, 1:num_periods] <= 1)         # setup indicator
@variable(model_LP, 0 <= x_LP[1:num_products, 1:num_periods] <= M2)   # production qty
@variable(model_LP, 0 <= s_LP[1:num_products, 1:num_periods] <= M3)   # end-period inventory

# Objective
@objective(model_LP, Min, 
    sum(c[i] * y_LP[i,t] + h[i] * s_LP[i,t] 
        for i in 1:num_products, t in 1:num_periods)
)

# Inventory‐balance and setup constraints
# first period: x_i1 = d_i1 + s_i1
@constraint(model_LP,[i in 1:num_products], x_LP[i,1] == demand[i,1] + s_LP[i,1])
# t ≥ 2: s_{i,t-1} + x_{i,t} = d_{i,t} + s_{i,t}
@constraint(model_LP,[i in 1:num_products, t in 2:num_periods], s_LP[i,t-1] + x_LP[i,t] == demand[i,t] + s_LP[i,t])
# linking production to setup
@constraint(model_LP,[i in 1:num_products, t in 1:num_periods], x_LP[i,t] <= M1 * y_LP[i,t])

@constraint(model_LP,[t in 1:num_periods], sum(x_LP[i,t] for i in 1:num_products) + sum(q[i] * y_LP[i,t] for i in 1:num_products) <= C)

######################
######################

# Solve
optimize!(model)
optimize!(model_LP)

# Display results
println("Optimal objective IP: ", objective_value(model))
println("\n y (IP) (setup):")
for i in 1:num_products, t in 1:num_periods
    println(" y[$i,$t] = ", value(y[i,t]))
end

println("\n x (IP) (production):")
for i in 1:num_products, t in 1:num_periods
    println(" x[$i,$t] = ", value(x[i,t]))
end

println("\n s (IP) (inventory):")
for i in 1:num_products, t in 1:num_periods
    println(" s[$i,$t] = ", value(s[i,t]))
end
println("-"^50)
println("Optimal objective LP: ", objective_value(model_LP))
println("\n y (LP) (setup):")
for i in 1:num_products, t in 1:num_periods
    println(" y[$i,$t] = ", value(y_LP[i,t]))
end

println("\n x (LP) (production):")
for i in 1:num_products, t in 1:num_periods
    println(" x[$i,$t] = ", value(x_LP[i,t]))
end

println("\n s (LP) (inventory):")
for i in 1:num_products, t in 1:num_periods
    println(" s[$i,$t] = ", value(s_LP[i,t]))
end