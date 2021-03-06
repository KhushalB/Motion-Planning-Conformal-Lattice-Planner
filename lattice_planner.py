from math import sin, cos
from scipy.integrate import quad
from scipy.optimize import minimize

def lattice_planner(x0, y0, heading_i, curvature_i, xf, yf, heading_f, curvature_f):
    t0, k0 = heading_i, curvature_i
    tf, kf = heading_f, curvature_f

    path = "Path that satisfies kinematic constraints"

    # NOTE: at this point we don't know sf
    sf = 3
    p1 = 1
    p2 = 1

    p0 = k0
    #p1 = k_s(sf/3, t_params)
    #p2 = k_s(2*sf/3, t_params)
    p3 = kf
    p4 = sf

    # NOTE: the parameters we are solving are p1, p2, and p4

    a0_p = remap_a0(p0)
    a1_p = remap_a1(p0,p1,p2,p3,p4)
    a2_p = remap_a2(p0,p1,p2,p3,p4)
    a3_p = remap_a3(p0,p1,p2,p3,p4)

    t_params = [a0_p, a1_p, a2_p, a3_p, t0]
    
    # Tunable variables
    k_max = 0.5
    alpha = 10
    beta = 10
    gamma = 10

    # probably need to pass p1 and p2 as well
    # Confused about the parameter-spiral mapping (p0 - p4, a0_p - a3_p)
    obj = objective(a0_p, a1_p, a2_p, a3_p, p4, alpha, beta, gamma, x0, y0, xf, yf, tf, t_params)
    constraint_1 = abs(p1) < k_max
    constraint_2 = abs(p2) < k_max

    return path

def objective(a0_p,a1_p,a2_p,a3_p,p4,alpha,beta,gamma,x0,y0,xf,yf,tf,t_params):
    """ the parameters to optimize are p1, p2, and p4 """
    return (f_be(a0_p, a1_p, a2_p, a3_p, p4) + 
            x_soft(alpha, p4, xf, x0, t_params) + 
            y_soft(beta, p4, yf, y0, t_params) + 
            theta_soft(gamma, p4, tf, t_params))

def k_s(s, t_params):
    """ Our cubic spiral equation """
    a0,a1,a2,a3 = tp[1],tp[2],tp[3],tp[4]
    return a3*s**3 + a2*s**2 + a1*s + a0

def fbe_integrand(s, a0, a1, a2, a3):
    """ Integrand to use with objective_function() """
    return (a3*s**3 + a2*s**2 + a1*s + a0)**2

def f_be(a0, a1, a2, a3, sf):
    """ Unconstrained objective function, using the quad integral solver
    from SciPy on our objective_integrand (variable 's') 
    from 0 to sf, using coefficients a0, a1, a2, and a3 """
    return quad(fbe_integrand, 0, sf, args=(a0,a1,a2,a3))

def x_soft(alpha, p4, xf, x0, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by alpha. """
    return alpha*(x_s(p4, x0, theta_params)-xf)

def y_soft(beta, p4, yf, y0, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by beta. """
    return beta*(y_s(p4, y0, theta_params)-yf)

def theta_soft(gamma, p4, tf, theta_params):
    """ Soft inequality constraints, allows a small
    margin of error between goal point and final point
    in the curve. Scaled by gamma. """
    return gamma*(theta_s(p4, theta_params)-tf)

def x_s(s, x0, theta_params):
    """ Estimates x value at location 's' along curve. Requires
    starting x value, as well as args to find theta(s). Uses
    Simpson's rule to divide domain into n=8 sections. """
    n0 = cos(theta_s(0, theta_params))
    n1 = 4*cos(theta_s(1*s/8, theta_params))
    n2 = 2*cos(theta_s(2*s/8, theta_params))
    n3 = 4*cos(theta_s(3*s/8, theta_params))
    n4 = 2*cos(theta_s(4*s/8, theta_params))
    n5 = 4*cos(theta_s(5*s/8, theta_params))
    n6 = 2*cos(theta_s(6*s/8, theta_params))
    n7 = 4*cos(theta_s(7*s/8, theta_params))
    n8 = cos(theta_s(s, theta_params))
    n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
    return x0 + (s/24)*(n_sum)

def y_s(s, y0, theta_params):
    """ Estimates y value at location 's' along curve. Requires
    starting y value, as well as args to find theta(s). Uses
    Simpson's rule to divide domain into n=8 sections. """
    n0 = sin(theta_s(0, theta_params))
    n1 = 4*sin(theta_s(1*s/8, theta_params))
    n2 = 2*sin(theta_s(2*s/8, theta_params))
    n3 = 4*sin(theta_s(3*s/8, theta_params))
    n4 = 2*sin(theta_s(4*s/8, theta_params))
    n5 = 4*sin(theta_s(5*s/8, theta_params))
    n6 = 2*sin(theta_s(6*s/8, theta_params))
    n7 = 4*sin(theta_s(7*s/8, theta_params))
    n8 = sin(theta_s(s, theta_params))
    n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
    return x0 + (s/24)*(n_sum)

def theta_s(s, tp):
    """ Finds theta value at location 's' along curve.
    Takes in theta parameters 'tp', which are the initial theta
    value and the curve's polynomial coefficients a0-a3 """
    t0,a0,a1,a2,a3 = tp[0],tp[1],tp[2],tp[3],tp[4]
    s4 = a3 * s**4 / 4
    s3 = a2 * s**3 / 3
    s2 = a1 * s**2 / 2
    s1 = a0 * s
    return t0+s4+s3+s2+s1

def remap_a0(p0):
    """ Map optimization params back to
    spiral coefficients. """
    return p0

def remap_a1(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = -1*(11*p0/2 - 9*p1 + 9*p2/2 - p3)
    denom = p4
    return num/denom

def remap_a2(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = 9*p0 - 45*p1/2 + 18*p2 - 9*p3/2
    denom = p4**2
    return num/denom

def remap_a3(p0, p1, p2, p3, p4):
    """ Map optimization params back to
    spiral coefficients. """
    num = -1*(9*p0/2 - 27*p1/2 + 27*p2/2 - 9*p3/2)
    denom = p4**3
    return num/denom