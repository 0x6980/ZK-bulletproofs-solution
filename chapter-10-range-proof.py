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

B = (FQ(12848606535045587128788889317230751518392478691112375569775390095112330602489), FQ(18818936887558347291494629972517132071247847502517774285883500818572856935411))

Q = (FQ(11573005146564785208103371178835230411907837176583832948426162169859927052980), FQ(895714868375763218941449355207566659176623507506487912740163487331762446439))

# scalar multiplication example: multiply(G, 42)
# EC addition example: add(multiply(G, 42), multiply(G, 100))

# remember to do all arithmetic modulo p
def commit(a, sL, b, sR, v, alpha, beta, gamma):
    A = add_points(vector_commit(G, a), vector_commit(H, b), multiply(B, alpha))
    S = add_points(vector_commit(G, sL), vector_commit(H, sR), multiply(B, beta))
    V = add(multiply(Q, v), multiply(B, gamma))
    return (A, S, V)

def commitT1_and_T2(t1, t2, tau_1, tau_2):
    T1 = add(multiply(Q, t1), multiply(B, tau_1))
    T2 = add(multiply(Q, t2), multiply(B, tau_2))
    return (T1, T2)


def evaluate(f_0, f_1, f_2, u):
    return (f_0 + f_1 * u + f_2 * u**2) % p

## step 0: Prover and verifier agree on G and H and B and Q

## step 1: Prover creates the commitments
## The prover chooses v and it’s binary representation aL and computes aR = aL -1.
one = np.array([1, 1, 1, 1]) # 1^{n}
two = np.array([1, 2, 4, 8]) # 2^{n}
# v = 1 * 1 + 0 * 2 + 1 * 4 + 1 * 8. we want to prove v is less than 2^4 (13 < 16)
v = 13
aL = np.array([1, 0, 1, 1]) # binary representation of v.
# aR = aL - one
aR = np.array([0, p-1, 0, 0]) # -1 = p-1 in the field

sL = np.array([random_element(),random_element(),random_element(),random_element()])
sR = np.array([random_element(),random_element(),random_element(),random_element()])

### blinding terms
alpha = random_element()
beta = random_element()
gamma = random_element()
tau_1 = random_element()
tau_2 = random_element()

A, S, V = commit(aL, sL, aR, sR, v, alpha, beta, gamma)

## step 2: Verifier picks y, z which the prover will use to combine the three inner products into a single one.
y = random_element()
z = random_element()

## step 3: Prover combine the three inner products into a single one.
yn = np.array([1, y, pow(y, 2, p), pow(y, 3, p)]) # y^{n}
y_mines_n = np.array([1, pow(y, -1, p), pow(y, -2, p), pow(y, -3, p)]) # y^{-n}

# delta = (z - pow(z, 2, p)) * np.inner(one, yn) - np.inner(one, two) * pow(z, 3, p)
# np.inner(one, two) = 15
delta = (z - pow(z, 2, p)) * np.inner(one, yn) - 15 * pow(z, 3, p)

# z * one
z_one = np.array([z, z, z, z]) # z.1^{n}
mines_z_one = np.array([p-z, p-z, p-z, p-z]) # -z.1^{n}
lhs_of_inner_product = aL + mines_z_one # aL - z.1^{n}

# pow(z, 2, p) * two
z2_two = np.array([pow(z, 2, p), (2 * pow(z, 2, p)) , (4 * pow(z, 2, p)) , (8 * pow(z, 2, p)) ]) # z^2 * 2^{n}
rhs_of_inner_product = (np.multiply(yn, aR) + (yn * z) + z2_two) % p # y^{n} o aR + y^{n} * z + z^2 * 2^{n}

t0 = np.inner(lhs_of_inner_product, rhs_of_inner_product) % p
t1 = (np.inner(lhs_of_inner_product, np.multiply(yn, sR)) + np.inner(rhs_of_inner_product, sL)) % p
t2 = np.inner(np.multiply(yn, sR) , sL) % p

T1, T2 = commitT1_and_T2(t1, t2, tau_1, tau_2)

## step 4: Verifier picks u
u = random_element()

## step 5: Prover evaluates l(u), r(u), t(u) and creates evaluation proofs
l_u = evaluate(lhs_of_inner_product, sL, 0, u)
r_u = evaluate(rhs_of_inner_product, np.multiply(yn, sR), 0, u)
t_u = evaluate(t0, t1, t2, u)

pi_lr = (alpha + beta * u) % p
pi_t = (pow(z, 2, p) * gamma + tau_1 * u + tau_2 * u**2) % p

## step 6: Verifier accepts or rejects
assert t_u == np.inner(np.array(l_u), np.array(r_u)) % p, "tu !=〈lu, ru〉"
# The verifier computes a new basis vector : y^{-n} o H
H_y_mines_one = np.array([multiply(H[0], y_mines_n[0]), multiply(H[1], y_mines_n[1]), multiply(H[2], y_mines_n[2]), multiply(H[3], y_mines_n[3])])
assert eq(add_points(A, multiply(S, u), vector_commit(G, mines_z_one), vector_commit(H_y_mines_one, (yn * z) + z2_two)), add_points(vector_commit(G, l_u), vector_commit(H_y_mines_one, r_u), multiply(B, pi_lr))), "l_u or r_u not evaluated correctly"
assert eq(add(multiply(Q, t_u), multiply(B, pi_t)), add_points(multiply(V, pow(z ,2, p)), multiply(Q, delta), multiply(T1, u), multiply(T2, u**2 % p))), "t_u not evaluated correctly"
print("accept")
