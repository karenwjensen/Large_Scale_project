using LinearAlgebra
using JuMP
using GLPK
include("task4-Data.jl")
#include("task4-masterIt4.jl")

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

# dual variables from master problem:
# it 1: 
# piVal = [0, 0, 0]
# kappa = [3.0, 9.0]
# it 2: 
# piVal = [0.0, -1.0, 0.0]
# kappa = [9.0, 12.0]
# it 3: 
# piVal = [0.0, -0.5, 0.0]
# kappa = [6.0, 8.0]
# it 4: 
piVal = [0.0, -0.5, -0.3333333333333333333]
kappa = [8.3333333333333, 8.0]

for k in 1:K
    sub = Model(GLPK.Optimizer)

    # 1) Variables for 3 periods each
    @variable(sub, x[1:3] >= 0)
    @variable(sub, y[1:3], Bin)
    @variable(sub, s[1:3] >= 0)

    branch_k = 1 
    branch_t = 2
    branch_val = 1
    # 2) Build a single 9-vector [ x; y; s ]
    vec = vcat(x,y,s)

    # 3) Reduced-cost objective
    @objective(sub, Min,
        dot(CV[k],            vec)  # production+setup+hold
      - dot(piVal, A0_V[k] * vec)  # capacity duals
      - kappa[k]                   # convexity dual
    )

    if k == branch_k
        @constraint(sub, y[branch_t] == branch_val)
    end 
    # 4) Flow‐balance equalities (rows 1:3 of A_V, b_sub)
    @constraint(sub,
       A_V[k][1:3, :] * vec .== b_sub[k][1:3]
    )

    # 5) big-M ≤ constraints (rows 4:6 of A_V, b_sub)
    @constraint(sub,
       A_V[k][4:6, :] * vec .<= b_sub[k][4:6]
    )

    optimize!(sub)
    if termination_status(sub) == MOI.OPTIMAL
        rc, patt = objective_value(sub), value.(vec)
        println("--- Subproblem $k optimal:  reduced cost = $rc")
        println("   pattern = ", patt)
    else
        error("Subproblem $k failed with status ", termination_status(sub))
    end
end
# build full A (15×18) and b (15) exactly as in your framework
A = [
    # flow eq & inventory linking for both products
    1 0 0   0 0 0  -1 0 0   0 0 0   0 0 0  0 0 0;
    0 0 0   0 0 0   0 0 0   1 0 0   0 0 0  -1 0 0;
    0 1 0   0 0 0   1 -1 0   0 0 0   0 0 0   0 0 0;
    0 0 1   0 0 0   0 1 -1   0 0 0   0 0 0   0 0 0;
    0 0 0   0 0 0   0 0 0   0 1 0   0 0 0   1 -1 0;
    0 0 0   0 0 0   0 0 0   0 0 1   0 0 0   0 1 -1;
    # big-M
    1 0 0  -12 0 0   0 0 0   0 0 0   0 0 0   0 0 0;
    0 1 0   0 -12 0  0 0 0   0 0 0   0 0 0   0 0 0;
    0 0 1   0 0 -12  0 0 0   0 0 0   0 0 0   0 0 0;
    0 0 0   0 0 0   0 0 0   1 0 0  -12 0 0   0 0 0;
    0 0 0   0 0 0   0 0 0   0 1 0   0 -12 0   0 0 0;
    0 0 0   0 0 0   0 0 0   0 0 1   0 0 -12  0 0 0;
    # Branch 
    0 0 0   0 1 0   0 0 0   0 0 0   0 0 0  0 0 0;
    0 0 0   0 1 0   0 0 0   0 0 0   0 0 0  0 0 0;
    # capacity ≤ 10 in each period
    1 0 0   2 0 0   0 0 0   1 0 0   1 0 0   0 0 0;
    0 1 0   0 2 0   0 0 0   0 1 0   0 1 0   0 0 0;
    0 0 1   0 0 2   0 0 0   0 0 1   0 0 1   0 0 0
]
b = [3, 0, 4, 5, 5, 2, 0,0,0,0,0,0,0,1, 10,10,10]
# master slices
A0 = A[15:17, :]
b0 = b[15:17]
# number of subproblems