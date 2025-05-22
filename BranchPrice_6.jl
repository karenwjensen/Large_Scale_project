using LinearAlgebra
using JuMP
using GLPK
include("task6-Data.jl")
#include("task4-masterIt4.jl")


#  y[2] = 0
X = Vector{Matrix{Float64}}(undef, K)
X[1] = [7 0 5 1 0 1 4 0 0]'   # one seed‐column for product 1
X[2] = [0 5 2 0 1 1 0 0 0; 0.0 7.0 0.0 0.0 1.0 0.0 0.0 2.0 0.0]'# one seed‐column for product 2println("X[1] = ", X[1])
lambdavals = [[1.0], [1.0, 0.0]] # lambda values for each product
variables_p1 = X[1]*lambdavals[1]
variables_p2 = X[2] * lambdavals[2] 
y1_star = variables_p1[4:6]
y2_star = variables_p2[4:6]


println("y1* = ",y1_star)
println("y2* = ",y2_star)

#  y[2] = 1
X = Vector{Matrix{Float64}}(undef, K)
X[1] = [7 0 5 1 0 1 4 0 0; 3.0 4.0 5.0 1.0 1.0 1.0 0.0 0.0 0.0]'   # one seed‐column for product 1
X[2] = [0 5 2 0 1 1 0 0 0]'  # one seed‐column for product 2println("X[1] = ", X[1])
lambdavals = [[0.33, 0.66], [1.0]] # lambda values for each product
variables_p1 = X[1]*lambdavals[1]
variables_p2 = X[2] * lambdavals[2] 
y1_star = variables_p1[4:6]
y2_star = variables_p2[4:6]


println("y1* = ",y1_star)
println("y2* = ",y2_star)
println("Variables p1 = ",variables_p1)
println("Variables p2 = ",variables_p2)


