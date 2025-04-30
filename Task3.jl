module DW_ColGen

using JuMP, GLPK, LinearAlgebra

EPSVAL = 0.00001

function setupSub(sub::JuMP.Model, A1, b1)
    (mA1, n) = size(A1)

    # NOTE: Remember to change here if the variables in the sub-problem is not binary.
    @variable(sub, xVars[1:n] >= 0, Int )

    # Dummy objective for now. We define once dual variables become known
    @objective(sub, Min, 0 )

    # NOTE: remember to change "<=" if your sub-problem uses a different type of constraints!
    @constraint(sub, A1*xVars .<= b1 )
    return xVars
end

# X1: initial matrix of extreme points
function setupMaster(master::JuMP.Model, A0, b0)
    #(n,p) = size(X1)

    @variable(master, lambda[1:0] >= 0 ) # No columns initially 
    # NOTE: Remember to consider if we need to maximize or minimize
    # NOTE: remember to change "==" if your master-problem uses a different type of constraints!
    @constraint(master, consRef[i=length(b0)], sum(A0[i,j]*lambda[j] for j in 1:length(lambda)) .<= b0[i])
    #@constraint(master, convCons1, sum(lambda[j] for j in 1:1) == 1)
    #@constraint(master, convCons2, sum(lambda[j] for j in 2:2) == 1)

    @objective(master, Min, 0)

    return lambda, consRef
end

# x is the new extreme point we wish to add to the master problem
function addColumnToMaster(master::JuMP.Model, c, A0, x, consRef, conv)
    j = length(all_variables(master)) + 1
    #(mA0, n) = size(A0)
    #A0x = A0*x
    @variable(master, lambda[j] >= 0)
    # oldvars = JuMP.all_variables(master)
    # new_var = @variable(master, base_name="lambda_$(length(oldvars))", lower_bound=0)
    set_objective_coefficient(master, lambda[j], dot(cost_x,x))

    for i=1:length(cap)
        # only insert non-zero elements (this saves memory)
        coeff = dot(A0[i,:],x)
        if abs(coeff) > EPSVAL
            set_normalized_coefficient(consRef[i], lambda[j], coeff)
        end
    end
    # add to convexity constraint
    set_normalized_coefficient(conv, lambda[j], 1.0)
end

function solveSub(sub, myPi, myKappa, c_sub, A0, xVars)
    # set objective. Remember to consider if maximization or minimization is needed
    @objective(sub, Min, dot(c_sub,xVars)- sum(pi[t] * dot(A0[t,:],xVars) for t in 1:length(pi)) - myKappa )

    optimize!(sub)
    if termination_status(sub) != MOI.OPTIMAL
        throw("Error: Non-optimal sub-problem status")
    end

    return JuMP.objective_value(sub), JuMP.value.(xVars)
end

function DWColGen(A0,A0_1,A0_2,A1_1,A1_2,b0,b1_1,b1_2,c_sub1,c_sub2, cost_x, X1)
    sub1 = Model(GLPK.Optimizer)
    sub2 = Model(GLPK.Optimizer)
    master = Model(GLPK.Optimizer)
    x1 = setupSub(sub1, A1_1, b1_1)
    x2 = setupSub(sub2, A1_2, b1_2)
    (lambda, consRef) = setupMaster(master, A0, b0)
    
    pi0 = zeros(length(b0))
    kappa0 = 0.0
    ### Bootstrap ### 
    # product 1
    _, xpat1 = solveSub(sub1, pi0, kappa0, c_sub1, A0_1, x1)
    full1 = vcat(xpat1, zeros(9))
    addColumnToMaster(master, cost_x, A0, full1, lambda, consRef, conv=missing)

    # product 2
    _, xpat2 = solveSub(sub2, π0, κ0, c_sub2, A0_2, x2)
    full2 = vcat(zeros(9), xpat2)
    addColumnToMaster(master, cost_x, A0, full2, lambda, consRef, conv=missing)
    convCons1 = @constraint(master, lambda[1] == 1)
    convCons2 = @constraint(master, lambda[2] == 1)

    # reset master objective now that we have columns
    @objective(master, Min, sum(objective_coefficient(master, v) * v for v in all_variables(master)))


    done = false
    iter = 1
    while !done
        optimize!(master)
        if termination_status(master) != MOI.OPTIMAL
            throw("Error: Non-optimal master-problem status")
        end
        # negative of dual values because Julia has a different
        # convention regarding duals of maximization problems:
        # We take the transpose ...'... to get a row vector
        myPi = [dual(consRef[i]) for i in 1:length(b0)]
        myKappa1 = dual(convCons1)
        myKappa2 = dual(convCons2)
        redCost1, xVal1 = solveSub(sub1, myPi, myKappa1, c_sub1, A0_1, x1)
        redCost2, xVal2 = solveSub(sub2, myPi, myKappa2, c_sub2, A0_2, x2)
        println("iteration: $iter, objective value = $(JuMP.objective_value(master)), reduced cost 1 = $redCost1, reduced cost 2 = $redCost2")
        # remember to look for negative reduced cost if we are minimizing.
        any_new = false 

        if redCost1 > EPSVAL
            full1 = vcat(xVal1, zeros(9))
            addColumnToMaster(master, cost_x, A0, full1, lambda, consRef, convCons1)
            any_new = true 
        end 

        if redCost2 > EPSVAL
            full2 = vcat(zeros(9), xVal2)
            addColumnToMaster(master, cost_x, A0, full2, lambda, consRef, convCons2)
            any_new = true 
        end
        # No more columns with non-negative cost. We are done.
        done = !any_new
        iter += 1
    end
    println("Done after $iter iterations. Objective value = $(JuMP.objective_value(master))")
end

function test()
    c1 = 1
    c2 = 2
    h1 = 1
    h2 = 1
    cost_x = [0 0 0 1 1 1 1 1 1 0 0 0 2 2 2 1 1 1]
    c_sub1 = [0 0 0 1 1 1 1 1 1] # product 1
    c_sub2 = [0 0 0 2 2 2 1 1 1] # product 2

    γ1 = sum(c1 .* [1,1,1]) + sum(h1 .* [0,0,0])  # = 1*3 + 1*0 = 3
    γ2 = sum(c2 .* [0,1,1]) + sum(h2 .* [0,0,0])  # = 2*2 + 1*0 = 4
    c  = [γ1, γ2]  # = [3, 4]


    # number of constraints and variables
    m = 15
    nvars = 18
    
    A = [
        # 1
        1  0  0   0   0   0  -1   0   0   0   0   0   0   0   0   0   0   0;
        # 2
        0  0  0   0   0   0   0   0   0   1   0   0   0   0   0  -1   0   0;
        # 3
        0  1  0   0   0   0   1  -1   0   0   1   0   0   0   0   0   0   0;
        # 4
        0  0  1   0   0   0   0   1  -1   0   0   1   0   0   0   0   0   0;
        # 5
        0  0  0   0   0   0   0   0   0   0   1   0   1  -1   0   1  -1   0;
        # 6
        0  0  0   0   0   0   0   0   0   0   0   1   0   0  -1   0   1  -1;
        # 7
        1  0  0 -12   0   0   0   0   0   0   0   0   0   0   0   0   0   0;
        # 8
        0  1  0   0 -12   0   0   0   0   0   0   0   0   0   0   0   0   0;
        # 9
        0  0  1   0   0 -12   0   0   0   0   0   0   0   0   0   0   0   0;
        #10
        0  0  0   0   0   0   0   0   0   1   0   0 -12   0   0   0   0   0;
        #11
        0  0  0   0   0   0   0   0   0   0   1   0   0 -12   0   0   0   0;
        #12
        0  0  0   0   0   0   0   0   0   0   0   1   0   0 -12   0   0   0;
        #13
        1  0  0   2   0   0   0   0   0   1   0   0   1   0   0   0   0   0;
        #14
        0  1  0   0   2   0   0   0   0   0   1   0   0   1   0   0   0   0;
        #15
        0  0  1   0   0   2   0   0   0   0   0   1   0   0   1   0   0   0
    ]
    
    
    # build RHS vector
    b = [
    3;    # Eqn 2, i=1
    0;    # Eqn 2, i=2
    4;    # Eqn 3, i=1,t=2
    5;    # Eqn 3, i=1,t=3
    5;    # Eqn 3, i=2,t=2
    2;    # Eqn 3, i=2,t=3
    0;0;0;0;0;0;  # Eqn 4 rows
    10;   # Eqn 5, t=1
    10;   # Eqn 5, t=2
    10;   # Eqn 5, t=3
    ]
    
    # split off the 3 capacity rows for the master
    A0 = A[13:15, :]
    A0_1 = A[13:15, 1:9]      # 3×18
    A0_2 = A[13:15, 10:18]
    b0 = b[13:15]         # length-3
    
    # now define each product’s A1 and b1 (the “convexified” constraints)
    # product 1 uses rows 1–6 & 7–9 of A and only cols 1:9
    A1_1 = A[1:12,1:9]   # big-M for product 1
    b1_1 = vcat( b[1:6], zeros(6) )  # first 6 RHS then three zeros
    
    # product 2 uses rows 1:6 & 7–9 of A but in cols 10:18
    A1_2 = A[1:12, 10:18]
    b1_2 = vcat( b[1:6], zeros(6) )
    
    A0 = A[13:15,:]
    b0 = b[13:15]
    A1 = A[1:12,:]
    b1 = b[1:12]

    # Product 1 demands = [3, 4, 5]
    # so pattern = (x1,x2,x3,y1,y2,y3,s1,s2,s3)
    X1_1 = [  3,  4,  5,   # x_{1t}
    1,  1,  1,   # y_{1t} (setup whenever x>0)
    0,  0,  0 ]  # s_{1t} (no carry)

    # Product 2 demands = [0, 5, 2]
    # note: no setup in period 1
    X1_2 = [  0,  5,  2,   # x_{2t}
    0,  1,  1,   # y_{2t}
    0,  0,  0 ]  # s_{2t}

    # Now build the initial X1 matrix by padding to length 18:
    # column 1: product 1 pattern in rows 1–9, zeros in 10–18
    # column 2: zeros in 1–9, product 2 pattern in 10–18
    X1 = nothing
    # The following is necessary if we only have one extreme point
    #X1 = reshape(X1, length(X1), 1)
    timeStart = time()
    DWColGen(A0,A0_1,A0_2,A1_1,A1_2,b0,b1_1,b1_2,c_sub1, c_sub2, cost_x, X1)
    println("Elapsed time: $(time()-timeStart) seconds")
end

test()

end