#!/usr/bin/env python3

import datetime
import sympy as sp

"""E = xE1 + yE2 + zE3 + wE4, assume w = 1 since E is up to scale."""

# Define symbolic vars
var_str = "x y z"
for i in range(4):
    for j in range(9):
        var_str += " e%d%d" % (i, j)
sp.var(var_str)

e0 = sp.Matrix([[e00, e01, e02], [e03, e04, e05], [e06, e07, e08]])
e1 = sp.Matrix([[e10, e11, e12], [e13, e14, e15], [e16, e17, e18]])
e2 = sp.Matrix([[e20, e21, e22], [e23, e24, e25], [e26, e27, e28]])
e3 = sp.Matrix([[e30, e31, e32], [e33, e34, e35], [e36, e37, e38]])

e = x*e0 + y*e1 + z*e2 + e3

# singularity condition
singularity_eq = e.det()

e_times_e_transpose = e * e.T
trace_eq = 2 * e_times_e_transpose * e - (e_times_e_transpose.trace() * e)
# trace_eq is 3x3 which should equal 0. this provides 9 equations
# print(trace_eq)
# print(trace_eq.shape)
print(trace_eq[2])

def collect_with_respect_to_vars(eq, vars):
    assert isinstance(vars, list)
    eq = eq.expand()
    if len(vars) == 0:
        return {1: eq}
    var_map = eq.collect(vars[0], evaluate=False)
    final_var_map = {}
    for var_power in var_map:
        sub_expression = var_map[var_power]
        sub_var_map = collect_with_respect_to_vars(sub_expression, vars[1:])
        for sub_var_power in sub_var_map:
            final_var_map[var_power*sub_var_power] = sub_var_map[sub_var_power]
    return final_var_map

# test_eq = x * y + (2*x*y + 3*x + 4*z*x + 5*z)**2
# print(collect_with_respect_to_vars(test_eq, [x, y])

# turn these 10 equations (9 from trace, 1 from determinant eq) into a form C(z)X
# where X spans [x^3, y^3, x^2*y, x*y^2, x^2, y^2, xy, x, y, 1]
equation_matrix = sp.Matrix([singularity_eq, trace_eq[0], trace_eq[1], trace_eq[2], trace_eq[3], trace_eq[4], trace_eq[5], trace_eq[6], trace_eq[7], trace_eq[8]])
print(datetime.datetime.now(), 'equation matrix set up')
coefficients = []
for equation in [singularity_eq, trace_eq[0], trace_eq[1], trace_eq[2], trace_eq[3], trace_eq[4], trace_eq[5], trace_eq[6], trace_eq[7], trace_eq[8]]:
    var_map = collect_with_respect_to_vars(equation, [x, y])
    coefficients.append([var_map.get(x**3, 0), var_map.get(y**3, 0), var_map.get(x**2*y, 0), var_map.get(x*y**2, 0),
                         var_map.get(x**2, 0), var_map.get(y**2, 0), var_map.get(x*y, 0), var_map.get(x, 0),
                         var_map.get(y, 0), var_map.get(1, 0)])
coefficients = sp.Matrix(coefficients)
print(datetime.datetime.now(), 'coefficients calculated')
print(coefficients)
# print(coefficients.det())
# print(datetime.datetime.now(), 'determinant calculated')
