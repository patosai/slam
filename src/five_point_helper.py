import sympy as sp

"""E = xE1 + yE2 + zE3 + wE4, assume w = 1 since E is up to scale."""

# Define symbolic vars
var_str = "x y z"
for i in range(4):
    for j in range(9):
        var_str += " e%d%d" % (i+1, j+1)
sp.var(var_str)

e1 = sp.Matrix([[e11, e12, e13], [e14, e15, e16], [e17, e18, e19]])
e2 = sp.Matrix([[e21, e22, e23], [e24, e25, e26], [e27, e28, e29]])
e3 = sp.Matrix([[e31, e32, e33], [e34, e35, e36], [e37, e38, e39]])
e4 = sp.Matrix([[e41, e42, e43], [e44, e45, e46], [e47, e48, e49]])

e = x*e1 + y*e2 + z*e3 + e4

# determinant equation
det_eq = e.det()
# this is pretty big
# print(det_eq)

e_times_e_transpose = e * e.T
trace_eq = 2 * e_times_e_transpose * e - (e_times_e_transpose.trace() * e)
# trace_eq is 3x3 which should equal 0. this provides 9 equations
print(trace_eq)
print(trace_eq.shape)

# turn these 10 equations (9 from trace, 1 from determinant eq) into a form C(z)X(x, y)
# where X spans [x^3, y^3, x^2*y, x*y^2, x^2, y^2, xy, x, y, 1]
C = TODO
