using JuMP, GLPK, LinearAlgebra

include("task4-Data.jl")

# dual variables from master problem:
# it 1: 
piVal = [0, 0, 0]
kappa = [6.0, 4.0]
# it 2: 
# piVal = [0.0, -0.5, 0.0]
# kappa = [6.0, 7.0]


for k in 1:K
    sub = Model(GLPK.Optimizer)

    # 1) Variables for 3 periods each
    @variable(sub, x[1:3] >= 0)
    @variable(sub, y[1:3], Bin)
    @variable(sub, s[1:3] >= 0)

    # 2) Build a single 9-vector [ x; y; s ]
    vec = vcat(x,y,s)

    # 3) Reduced-cost objective
    @objective(sub, Min,
        dot(CV[k],vec)  # production+setup+hold
      - dot(piVal, A0_V[k] * vec)  # capacity duals
      - kappa[k]                   # convexity dual
    )

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
