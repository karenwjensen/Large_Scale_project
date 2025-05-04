using LinearAlgebra
using JuMP
using GLPK
include("task4-Data.jl")
include("task4-masterIt4.jl")

X = Vector{Matrix{Float64}}(undef, K)
X[1] = [3 4 5 1 1 1 0 0 0; 7.0 0.0 5.0 1.0 0.0 1.0 4.0 0.0 0.0]'   # one seed‐column for product 1
X[2] = [3 2 2 1 1 1 3 0 0; 0.0 7.0 0.0 0.0 1.0 0.0 0.0 2.0 0.0; 5.0 0.0 2.0 1.0 0.0 1.0 5.0 0.0 0.0; 0.0 5.0 2.0 0.0 1.0 1.0 0.0 0.0 0.0]'  # one seed‐column for product 2println("X[1] = ", X[1])
lambdavals = [[0.6666666666666662, 0.3333333333333338], [0.0, 5.329070518200751e-16, 0.0, 1.0]]
variables_p1 = X[1]*lambdavals[1]
variables_p2 = X[2] * lambdavals[2] 
y1_star = variables_p1[4:6]
y2_star = variables_p2[4:6]


println("y1* = ",y1_star)
println("y2* = ",y2_star)