from py_ecc.bn128 import G1, multiply, add, FQ, eq, Z1
from py_ecc.bn128 import curve_order as p
import numpy as np
from functools import reduce
import random

def random_element():
    return random.randint(0, p)

def add_points(*points):
    return reduce(add, points, Z1)

# if points = G1, G2, G3, G4 and scalars = a,b,c,d vector_commit returns
# aG1 + bG2 + cG3 + dG4
def vector_commit(points, scalars):
    return reduce(add, [multiply(P, i) for P, i in zip(points, scalars)], Z1)

# these EC points have unknown discrete logs:
G = [(FQ(6286155310766333871795042970372566906087502116590250812133967451320632869759), FQ(2167390362195738854837661032213065766665495464946848931705307210578191331138)),
     (FQ(6981010364086016896956769942642952706715308592529989685498391604818592148727), FQ(8391728260743032188974275148610213338920590040698592463908691408719331517047)),
     (FQ(15884001095869889564203381122824453959747209506336645297496580404216889561240), FQ(14397810633193722880623034635043699457129665948506123809325193598213289127838)),
     (FQ(6756792584920245352684519836070422133746350830019496743562729072905353421352), FQ(3439606165356845334365677247963536173939840949797525638557303009070611741415))]

H = [(FQ(13728162449721098615672844430261112538072166300311022796820929618959450231493), FQ(12153831869428634344429877091952509453770659237731690203490954547715195222919)),
    (FQ(17471368056527239558513938898018115153923978020864896155502359766132274520000), FQ(4119036649831316606545646423655922855925839689145200049841234351186746829602)),
    (FQ(8730867317615040501447514540731627986093652356953339319572790273814347116534), FQ(14893717982647482203420298569283769907955720318948910457352917488298566832491)),
    (FQ(419294495583131907906527833396935901898733653748716080944177732964425683442), FQ(14467906227467164575975695599962977164932514254303603096093942297417329342836))]

Q = (FQ(11573005146564785208103371178835230411907837176583832948426162169859927052980), FQ(895714868375763218941449355207566659176623507506487912740163487331762446439))

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

a = [4,2,42,420]
b = [2,3,5,8]

P = add_points(vector_commit(G, a), vector_commit(H, b), multiply(Q, np.inner(a, b)))

L1, R1 = compute_secondary_diagonal(G, H, a, b)
u1 = random_element()

aprime = fold(a, u1)
Gprime = fold_points(G, pow(u1, -1, p))

bprime = fold(b, pow(u1, -1, p))
Hprime = fold_points(H, u1)

L2, R2 = compute_secondary_diagonal(Gprime, Hprime, aprime, bprime)
u2 = random_element()

aprimeprime = fold(aprime, u2)
Gprimeprime = fold_points(Gprime, pow(u2, -1, p))

bprimeprime = fold(bprime, pow(u2, -1, p))
Hprimeprime = fold_points(Hprime, u2)

assert len(Gprimeprime) == 1 and len(aprimeprime) == 1 and len(Hprimeprime) == 1 and len(bprimeprime) == 1, "final vector must be len 1"

Pprime = add_points(multiply(L2, pow(u2, 2, p)), multiply(L1, pow(u1, 2, p)), P, multiply(R1, pow(u1, -2, p)), multiply(R2, pow(u2, -2, p)))
G_commit = vector_commit(Gprimeprime, aprimeprime)
H_commit = vector_commit(Hprimeprime, bprimeprime)
Q_commit = multiply(Q, np.inner(aprimeprime, bprimeprime))

assert eq(add_points(G_commit, H_commit, Q_commit), Pprime), "invalid proof"
print("accept")
