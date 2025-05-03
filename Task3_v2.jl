module DW_ColGen

using JuMP, GLPK, LinearAlgebra

const EPS = 1e-6

"""
    setupSubModel(A1, b1)

Builds a subproblem with variables xₜ ≥ 0, yₜ ∈ {0,1}, sₜ ≥ 0 for t=1:3,
flow‐balance (equality) in rows 1:6 of A1/b1 and big-M ≤ in rows 7:12.
Returns (model, x, y, s).
"""
function setupSubModel(A1, b1)
    sub = Model(GLPK.Optimizer)
    # variables: 3 periods each
    @variable(sub, x[1:3] >= 0)
    @variable(sub, y[1:3], Bin)
    @variable(sub, s[1:3] >= 0)
    # dummy objective
    @objective(sub, Min, 0)
    # equality constraints (first 6 rows)
    #@constraint(sub, A1[1:6,1:3]*x .+ A1[1:6,4:6]*y .+ A1[1:6,7:9]*s .== b1[1:6])
    # ≤ constraints (rows 7–12)
    #@constraint(sub, A1[7:12,1:3]*x .+ A1[7:12,4:6]*y .+ A1[7:12,7:9]*s .<= b1[7:12])
    @constraint(sub,
       A1[1:3,   1:3]*x .+
       A1[1:3,   4:6]*y .+
       A1[1:3,   7:9]*s .== b1[1:3])
    # ≤ constraints (rows 4–6)
    @constraint(sub,
       A1[4:6,   1:3]*x .+
       A1[4:6,   4:6]*y .+
       A1[4:6,   7:9]*s .<= b1[4:6])
    return sub, x, y, s
end

"""
    setupMaster(A0, b0, c_full, X1_1, X1_2)

Constructs the restricted master with two initial columns (one per product).
Returns (master, consCap, (conv1,conv2), Xfull).
"""
function setupMaster(A0, b0, c_full, X1_1, X1_2)
    master = Model(GLPK.Optimizer)

    # Start with “do nothing” for each product:
    #X[1] = [3 4 5 1 1 1 0 0 0]'   # one seed‐column for product 1
    #X[2] = [3 2 2 1 1 1 0 0 0]'  # one seed‐column for product 2
    Z1 = [3 4 5 1 1 1 0 0 0]'
    Z2 = [3 2 2 1 1 1 0 0 0]'
    Xfull = hcat(
        vcat(Z1, zeros(length(Z2))),   # product 1 = all zero
        vcat(zeros(length(Z1)), Z2)    # product 2 = all zero
    )
    p1Count = 1   # one dummy column for product 1
    p2Count = 1   # one dummy column for product 2
    #BIGP = 100000 
    #@variable(master, s[1:size(A0,1)] >= 0)
    @variable(master, λ[1:p1Count+p2Count] >= 0)
    # objective: minimize c_full' * (Xfull * λ)
    @objective(master, Min, sum(c_full' * Xfull[:,j] * λ[j] for j=1:length(λ)))
    # capacity constraints A0 * (Xfull * λ) ≤ b0
    consCap = @constraint(master, A0 * (Xfull * λ) .<= b0)
    # convexity per product
    conv1 = @constraint(master, sum(λ[j] for j=1:p1Count) == 1)
    conv2 = @constraint(master, sum(λ[j] for j=p1Count+1:p1Count+p2Count) == 1)
    return master, consCap, (conv1,conv2), Xfull
end

"""
    addColumn!(master, c_full, A0, patt_full, consCap, convs, which)

Adds a new lambda‐variable for pattern given 
the capacity constraints and the correct convexity constraint `convs[which]`.
"""
function addColumn!(master, c_full, A0, patt_full, consCap, (conv1,conv2), which)
    v = @variable(master, lower_bound=0)
    set_objective_coefficient(master, v, c_full' * patt_full)
    # capacity
    a = A0 * patt_full
    for i in eachindex(a)
        if abs(a[i]) > 1e-9
            set_normalized_coefficient(consCap[i], v, a[i])
        end
    end
    # convexity
    conv = which == 1 ? conv1 : conv2
    set_normalized_coefficient(conv, v, 1.0)
    return v
end

"""
    solveSub(sub, x, y, s, π, κ, c_sub, A0_sub)

Sets the subproblem objective `max c_sub'xys - π'A0_sub*xys - κ`, solves it
"""
function solveSub(sub, x, y, s, π, κ, c_sub, A0_sub)
    vec = vcat(x, y, s)
    @objective(sub, Min, dot(c_sub, vec) - dot(π, A0_sub * vec) - κ)
    optimize!(sub)
    if termination_status(sub) != MOI.OPTIMAL
        error("Subproblem not optimal")
    end
    return objective_value(sub), value.(vec)
end

"""
    DWColGen(A0, A1_1, b1_1, c_sub1, A1_2, b1_2, c_sub2, b0, c_full, X1_1, X1_2)

Runs the full column‐generation loop with two independent subproblems.
"""
function DWColGen(A0, A1_1, b1_1, c_sub1, A1_2, b1_2, c_sub2, b0, c_full, X1_1, X1_2)
    master, consCap, convs, Xfull = setupMaster(A0, b0, c_full, X1_1, X1_2)
    sub1, x1, y1, s1 = setupSubModel(A1_1, b1_1)
    sub2, x2, y2, s2 = setupSubModel(A1_2, b1_2)
    iter = 0
    while true
        optimize!(master)
        @assert termination_status(master) == MOI.OPTIMAL "Master failed"
        π  = dual.(consCap)
        κ1 =  dual(convs[1])
        κ2 =  dual(convs[2])

        r1, p1 = solveSub(sub1, x1,y1,s1, π, κ1, c_sub1, A0[:,1:9])
        r2, p2 = solveSub(sub2, x2,y2,s2, π, κ2, c_sub2, A0[:,10:18])

        # Extract and print duals
        pi = dual.(consCap)
        kappa1, kappa2 = dual(convs[1]), dual(convs[2])
        
        iter += 1
        println("Iteration $iter  |  master obj = $(objective_value(master))  |  red₁ = $r1  |  red₂ = $r2")
        println("Iteration $iter  |  pi = $pi   |  kappa1 = $kappa1  |  kappa2 = $kappa2  |")
        println("Iteration $iter  |  Adding pattern to P1: $p1  |")
        println("Iteration $iter  |  Adding pattern to P2: $p2  |")
        println("-"^100)

        added = false
        if r1 < -EPS
            #println("  adding pattern to product 1: ", p1)  
            patt1 = vcat(p1, zeros(9))
            addColumn!(master, c_full, A0, patt1, consCap, convs, 1)
            added = true
        end
        if r2 < -EPS
            #println("  adding pattern to product 2: ", p2)  
            patt2 = vcat(zeros(9), p2)
            addColumn!(master, c_full, A0, patt2, consCap, convs, 2)
            added = true
        end
        if !added
            break
        end
    end

    println("Done after $iter iterations.  Final master obj = $(objective_value(master))")
end

"""
    test()

Defines the data for your 2-product, 3-period instance,
builds A0, b0, A1_1, b1_1, A1_2, b1_2, c_full, etc., and calls DWColGen.
"""
function test()
    # costs and times
    c1, c2 = 1.0, 2.0
    h1, h2 = 1.0, 1.0
    # subproblem cost vectors (length 9: [x]-coeffs zero; then y; then s)
    c_sub1 = [0,0,0, c1,c1,c1, h1,h1,h1]
    c_sub2 = [0,0,0, c2,c2,c2, h2,h2,h2]
    c_full  = vcat(c_sub1, c_sub2)

    # demands
    d = [3,4,5,  0,5,2]  # first 3 for product1, next 3 for product2
    # big-M used in your A matrix
    M = 12.0

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
        # capacity ≤ 10 in each period
        1 0 0   2 0 0   0 0 0   1 0 0   1 0 0   0 0 0;
        0 1 0   0 2 0   0 0 0   0 1 0   0 1 0   0 0 0;
        0 0 1   0 0 2   0 0 0   0 0 1   0 0 1   0 0 0
    ]
    b = [3, 0, 4, 5, 5, 2, 0,0,0,0,0,0, 10,10,10]

    # master slices
    A0 = A[13:15, :]
    b0 = b[13:15]

    # two subproblem slices (columns 1:9 and 10:18)
    # demand equations for product 1 live in rows 1, 3, 4 of A
    # big-M constraints for product 1 are rows 7, 8, 9
    eq1   = [1, 3, 4]
    ineq1 = [7, 8, 9]
    A1_1 = vcat( A[eq1,    1:9],
                A[ineq1,  1:9] )
    b1_1 = vcat( b[eq1], zeros(length(ineq1)) )

    # demand equations for product 2 live in rows 2, 5, 6 of A
    # big-M constraints for product 2 are rows 10, 11, 12
    eq2   = [2, 5, 6]
    ineq2 = [10, 11, 12]
    A1_2 = vcat( A[eq2,    10:18],
                A[ineq2,  10:18] )
    b1_2 = vcat( b[eq2], zeros(length(ineq2)) )



    # initial extreme patterns for each product (length 9)
    X1_1 = [3,4,5, 1,1,1, 0,0,0]
    X1_2 = [0,5,2, 0,1,1, 0,0,0]

    DWColGen(A0, A1_1, b1_1, c_sub1, A1_2, b1_2, c_sub2, b0, c_full, X1_1, X1_2)
end

end
DW_ColGen.test()