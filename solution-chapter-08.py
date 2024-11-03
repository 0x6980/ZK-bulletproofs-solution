from py_ecc.bn128 import is_on_curve, G1, multiply, add, FQ, eq, Z1
from py_ecc.fields import field_properties
field_mod = field_properties["bn128"]["field_modulus"]
from py_ecc.bn128 import curve_order as p
import numpy as np
from functools import reduce
import random
from hashlib import sha256
from libnum import has_sqrtmod_prime_power, sqrtmod_prime_power

def random_basis(n,  seed):
    b = 3 # for bn128, y^2 = x^3 + 3
    # seed = "RareSkills"

    x = int(sha256(seed.encode('ascii')).hexdigest(), 16) % field_mod 

    entropy = 0

    vector_basis = []

    # modify the code below to generate n points

    for i in range(n):
        while not has_sqrtmod_prime_power((x**3 + b) % field_mod, field_mod, 1):
            # increment x, so hopefully we are on the curve
            x = (x + 1) % field_mod
            entropy = entropy + 1

        # pick the upper or lower point depending on if entropy is even or odd
        y = list(sqrtmod_prime_power((x**3 + b) % field_mod, field_mod, 1))[entropy & 1 == 0]
        point = (FQ(x), FQ(y))
        assert is_on_curve(point, b), "sanity check"
        vector_basis.append(point)

        # new x value
        x = int(sha256(str(x).encode('ascii')).hexdigest(), 16) % field_mod 
    return vector_basis

def random_element():
    return random.randint(0, p)

def add_points(*points):
    return reduce(add, points, Z1)

# if points = G1, G2, G3, G4 and scalars = a,b,c,d vector_commit returns
# aG1 + bG2 + cG3 + dG4
def vector_commit(points, scalars):
    return reduce(add, [multiply(P, i) for P, i in zip(points, scalars)], Z1)

# scalar multiplication example: multiply(G, 42)
# EC addition example: add(multiply(G, 42), multiply(G, 100))

# remember to do all arithmetic modulo p
def commit(a, sL, b, sR, alpha, beta, gamma, tau_1, tau_2, t1, t2):
    A = add_points(vector_commit(G, a), vector_commit(H, b), multiply(B, alpha))
    S = add_points(vector_commit(G, sL), vector_commit(H, sR), multiply(B, beta))
    V = add(multiply(Q, np.inner(a, b)), multiply(B, gamma))
    T1 = add(multiply(Q, t1), multiply(B, tau_1))
    T2 = add(multiply(Q, t2), multiply(B, tau_2))
    return (A, S, V, T1, T2)

def evaluate(f_0, f_1, f_2, u):
    return (f_0 + f_1 * u + f_2 * u**2) % p

def prove(blinding_0, blinding_1, blinding_2, u):
    pi = (blinding_0 + blinding_1 * u + blinding_2 * u**2) % p
    return pi

# return a folded vector of length n/2 for scalars
def fold(scalar_vec, u):
    fold_list = []
    for i in range(len(scalar_vec) - 1):
        if(i % 2 == 0):
            fold_list.append(scalar_vec[i] * u + scalar_vec[i+1] * pow(u, -1, p))
    return fold_list

# return a folded vector of length n/2 for points
def fold_points(point_vec, u):
    fold_points_List = []
    for i in range(len(point_vec) - 1):
        if(i % 2 == 0):
            fold_points_List.append(add(multiply(point_vec[i], u), multiply(point_vec[i+1], pow(u, -1, p))))
    return fold_points_List

# return (L, R)
def compute_secondary_diagonal(G_vec, H_vec, a_input, b_input):
    G_L = Z1
    G_R = Z1
    H_L = Z1
    H_R = Z1
    sum3L = 0
    sum3R = 0
    for i in range(len(G_vec) - 1):
        if(i % 2 == 0):
            G_L = add(G_L, multiply(G_vec[i+1], a_input[i]))
            G_R = add(G_R, multiply(G_vec[i], a_input[i+1]))

            H_L = add(H_L, multiply(H_vec[i+1], b_input[i]))
            H_R = add(H_R, multiply(H_vec[i], b_input[i+1]))

            sum3L += a_input[i]*b_input[i+1]
            sum3R += a_input[i+1]*b_input[i]

    Q_L = multiply(Q, sum3L % p)
    Q_R = multiply(Q, sum3R % p)

    return (add_points(G_L, H_R, Q_L), add_points(G_R, H_L, Q_R))

def verify(Pprime, Gprime, Hprime, aprime, bprime):
    len_check = len(Gprime) == 1 and len(aprime) == 1 and len(Hprime) == 1 and len(bprime) == 1

    G_commit1 = vector_commit(Gprime, aprime)
    H_commit1 = vector_commit(Hprime, bprime)
    Q_commit1 = multiply(Q, np.inner(aprime, bprime))

    if(len_check and add_points(G_commit1, H_commit1, Q_commit1) == Pprime):
        print("accept")
    else:
        print("reject")

## step 0: Prover and verifier agree on G and H, Q, B
# these EC points have unknown discrete logs:


## step 1: Prover creates the commitments

# a1 = np.array([808, 140, 166, 209])
# b1 = np.array([88, 242, 404, 602])
a = np.array([random_element(),random_element(),random_element(),random_element()])
b = np.array([random_element(),random_element(),random_element(),random_element()])
sL = np.array([random_element(),random_element(),random_element(),random_element()])
sR = np.array([random_element(),random_element(),random_element(),random_element()])

# a2 = [433, 651]
# b2 = [282, 521]
# a = np.array([random_element(), random_element()])
# b = np.array([random_element(), random_element()])
# sL = np.array([random_element(), random_element()])
# sR = np.array([random_element(), random_element()])

# a3 = [222]
# a4 = [313]
# a = np.array([random_element()])
# b = np.array([random_element()])
# sL = np.array([random_element()])
# sR = np.array([random_element()])


n = len(a)
G = random_basis(n, "GRareSkills") 
H = random_basis(n, "HRareSkills0x6980")
Q = random_basis(1, "Q0x6980")[0]
B = random_basis(1, "B0x6980")[0]

t1 = np.inner(a, sR) + np.inner(b, sL)
t2 = np.inner(sR, sL)

### blinding terms
alpha = random_element()
beta = random_element()
gamma = random_element()
tau_1 = random_element()
tau_2 = random_element()

A, S, V, T1, T2 = commit(a, sL, b, sR, alpha, beta, gamma, tau_1, tau_2, t1, t2)

## step 2: Verifier picks u
u = random_element()

## step 3: Prover evaluates l(u), r(u), t(u) and creates evaluation proofs
l_u = evaluate(a, sL, 0, u)
r_u = evaluate(b, sR, 0, u)
t_u = evaluate(np.inner(a,b), t1, t2, u)

pi_lr = prove(alpha, beta, 0, u)
pi_t = prove(gamma, tau_1, tau_2, u)

## step 4: Verifier accepts or rejects
# assert t_u == np.inner(np.array(l_u), np.array(r_u)) % p, "tu !=〈lu, ru〉"

C = add(vector_commit(G, l_u), vector_commit(H, r_u))

P = add(C, multiply(Q, t_u))


if(len(a) == 1):
    if(P == add_points(vector_commit(G, l_u), vector_commit(H, r_u), multiply(Q, t_u))):
        print("accept")
    else:
        print("reject")
else:
    L1, R1 = compute_secondary_diagonal(G, H, l_u, r_u)
    u1 = random_element()
    Pprime1 = add_points(multiply(L1, pow(u1, 2, p)), P, multiply(R1, pow(u1, -2, p)))

    aprime = fold(l_u, u1)
    Gprime = fold_points(G, pow(u1, -1, p))

    bprime = fold(r_u, pow(u1, -1, p))
    Hprime = fold_points(H, u1)


    if(len(a) == 2):
        verify(Pprime1, Gprime, Hprime, aprime, bprime)
    else:
        L2, R2 = compute_secondary_diagonal(Gprime, Hprime, aprime, bprime)
        u2 = random_element()
        Pprime = add_points(multiply(L2, pow(u2, 2, p)), Pprime1, multiply(R2, pow(u2, -2, p)))

        aprimeprime = fold(aprime, u2)
        Gprimeprime = fold_points(Gprime, pow(u2, -1, p))

        bprimeprime = fold(bprime, pow(u2, -1, p))
        Hprimeprime = fold_points(Hprime, u2)

        verify(Pprime, Gprimeprime, Hprimeprime, aprimeprime, bprimeprime)
