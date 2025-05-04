module test

using JuMP, GLPK, LinearAlgebra
include("task6-Data.jl")

# replace the following with your actual initialization:
X = Vector{Matrix{Float64}}(undef, K)
X[1] = [3 4 5 1 1 1 0 0 0]'   # one seed‐column for product 1
X[2] = [3 2 2 1 1 1 3 0 0]'  # one seed‐column for product 2
P = [1,1]                # P[k] number of extreme points for polyhedron k
println("X[1] = ", X[1])
println("X[2] = ", X[2])
println("P = ",P)

# build the master
master = Model(GLPK.Optimizer)

# a separate λ‐array for each product
@variable(master, lambda1[1:P[1]] >= 0)
@variable(master, lambda2[1:P[2]] >= 0)
lambda = [lambda1, lambda2]
# objective: sum_i CV[i]'*X[i]*lambda[i]
@objective(master, Min,
    sum( CV[i]' * X[i] * lambda[i] for i in 1:K )
)

# capacity coupling: A0_V[i]*X[i]*lambda[i] summed over i ≤ b0
cons = Vector{ConstraintRef}(undef, 3)
for t in 1:3
  cons[t] = @constraint(master,
    sum(A0_V[i]*X[i]*lambda[i] for i=1:K)[t] <= b0[t]
  )
end
# convexity: for each i, sum_j λ[i][j] == 1
@constraint(master, conv[i=1:K],
    sum(lambda[i][j] for j in 1:P[i]) == 1
)

optimize!(master)

if termination_status(master) == MOI.OPTIMAL
    println("Master Obj = ", objective_value(master))
    λval = [value.(lambda[i]) for i in 1:K]
    println("λ = ", λval)
    π = [ dual(cons[i]) for i in 1:3 ]
    κ = [ dual(conv[i]) for i in 1:K ]
    println("π = ", π, "  κ = ", κ)
else
    error("Master not optimal: ", termination_status(master))
end

end
