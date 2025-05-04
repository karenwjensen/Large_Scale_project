# IP-data

# costs and times
c1, c2 = 1.0, 2.0
h1, h2 = 1.0, 1.0
# subproblem cost vectors (length 9: [x]-coeffs zero; then y; then s)
c_sub1 = [0,0,0, c1,c1,c1, h1,h1,h1]
c_sub2 = [0,0,0, c2,c2,c2, h2,h2,h2]
c_full  = vcat(c_sub1, c_sub2)

# demands
d = [3,4,5,  0,5,2]  # first 3 for product1, next 3 for product2
#display(d)

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
    # branch constraint
    0 0 0   0 1 0   0 0 0   0 0 0   0 0 0   0 0 0;
    0 0 0   0 0 0   0 0 0   0 0 0   0 0 0   0 0 0;
    # capacity ≤ 10 in each period
    1 0 0   2 0 0   0 0 0   1 0 0   1 0 0   0 0 0;
    0 1 0   0 2 0   0 0 0   0 1 0   0 1 0   0 0 0;
    0 0 1   0 0 2   0 0 0   0 0 1   0 0 1   0 0 0
]
b = [3, 0, 4, 5, 5, 2, 0,0,0,0,0,0, 1,0, 10,10,10] # change fixed branch value to 0 or 1
# master slices
A0 = A[15:17, :]
b0 = b[15:17]
# number of subproblems
K = 2

# columns in each subproblem
V = [
    1:9,      # product 1 uses vars 1–9
    10:18     # product 2 uses vars 10–18
]

# row‐blocks for each subproblem: first the 3 demand eqns, then the 3 big–M ineqns
eq   = [[1,3,4],   [2,5,6]]
ineq = [[7,8,9], [10,11,12]]
branch = [[13], [14]]
subBlocks = [
  vcat(eq[1],   ineq[1], branch[1]),
  vcat(eq[2],   ineq[2], branch[2])
]

# pre‐allocate container types
CV    = Vector{Vector{Float64}}(undef, K)
A_V   = Vector{Matrix{Float64}}(undef, K)
A0_V  = Vector{Matrix{Float64}}(undef, K)
b_sub = Vector{Vector{Float64}}(undef, K)

for k in 1:K
    # 1) the cost‐vector for sub‐problem k
    CV[k]   = c_full[V[k]]
    # 2) its full A1 (6×9) and b1 (length 6)
    A_V[k]  = vcat(
                  A[ eq[k],    V[k] ],
                  A[ ineq[k], V[k] ],
                  A[ branch[k], V[k] ]
               )
    b_sub[k] = vcat(
                  b[ eq[k] ],      # the three demand RHS
                  zeros(length(ineq[k])),  # the three big‐M zeros
                  b[ branch[k] ]   # the two branch RHS
               )
    # 3) the capacity slice (3×9)
    A0_V[k] = A0[:, V[k]]
end

display(CV)
display(A_V)
display(b_sub)

