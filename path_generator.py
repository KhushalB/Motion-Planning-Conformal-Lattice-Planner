from math import sin, cos
from scipy.integrate import quad
from scipy.optimize import minimize

class PathGenerator:
    def __init__(self, start_x, start_y, start_theta, start_curvature, 
                        goal_x, goal_y, goal_theta, goal_curvature,
                        alpha=10, beta=10, gamma=10, kmax=0.5):
        """ Takes start and end coordinates, heading, and curvature, 
        returns a curve that connects them. Alpha, beta, and gamma
        scale our soft constraints for x, y, and theta. For example,
        if we want our curve endpoint to be constrained very tightly 
        to our goal conditions, we make these large. If we are okay with
        a bit of error, we make these small. Kmax determinines the
        max sharpness of a turn. If we want a very smooth ride, we make this
        small, if we are okay with sharp turns, we make this large."""
        # Start conditions
        self.x0, self.y0 = start_x, start_y
        self.t0, self.k0 = start_theta, start_curvature
        # End conditions
        self.xf, self.yf = goal_x, goal_y
        self.tf, self.kf = goal_theta, goal_curvature
        # Constraints
        self.alpha, self.beta, self.gamma, self.kmax = alpha, beta, gamma, kmax

        # We break the curve into 3 equal sections
        # We are solving for p1, p2, p4 (p0 & p3 are constant)
        self.p0 = self.k0           # Starting point of curve
        self.p1 = "1/3 down curve"  # 1/3 down curve
        self.p2 = "2/3 down curve"  # 2/3 down curve
        self.p3 = self.kf           # Ending point of curve
        self.p4 = "Length of curve" # Length of curve

        # Constraints for the optimization problem
        self.bound_p1 = [-1*self.kmax, self.kmax]    # p1's constraint
        self.bound_p2 = [-1*self.kmax, self.kmax]    # p2's constraint
        self.bounds = [self.bound_p1, self.bound_p2]

        # TODO: optimize self.objective_function
        path_raw = "Optimization of self.objective_function() for params p1, p2, p4"
        # Pseudocode layout?: optimize(self.objective_function, params=[self.p1, self.p2, self.p4], bounds=self.bounds)
        path = "The path above, but mapping the p values back to spiral parameters"
        
        # NOTE: I still don't quite understand the mapping from the p's to the a's and back
        # I get that it makes the optimization problem easier, but I don't get the logistics
        # of how and when we are supposed to do it

        # returns a kinematically feasible cubic spiral from start point to end point
        return path

    def objective_function(self):
        """ the parameters to optimize are p1, p2, and p4 """
        return self.f_be() + self.x_soft() + self.y_soft() + self.theta_soft()

    def k_s(self, s):
        """ Our cubic spiral equation. Not sure if we need this """
        return self.a3_map()*s**3 + self.a2_map()*s**2 + self.a1_map()*s + self.a0_map()

    def f_be_integrand(self, s, a0, a1, a2, a3):
        """ Integrand to use with f_be() 
        This is our cubic spiral equation squared 
        integrated for variable 's' from 0 to p4"""
        return (a3*s**3 + a2*s**2 + a1*s + a0)**2
    def f_be(self):
        """ Unconstrained objective function, using the quad integral solver
        from SciPy on our objective_integrand (variable 's') 
        from 0 to curve length p4, using coefficients a0, a1, a2, and a3 """
        spiral_vals = (self.a0_map(), self.a1_map(), self.a2_map(), self.a3_map())
        return quad(self.f_be_integrand, 0, self.p4, args=spiral_vals)

    def x_soft(self):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by alpha. """
        return self.alpha*(self.x_s(self.p4) - self.xf)

    def y_soft(self):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by beta. """
        return self.beta*(self.y_s(self.p4) - self.yf)

    def theta_soft(self):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by gamma. """
        return self.gamma*(self.theta_s(self.p4) - self.tf)

    def x_s(self, s):
        """ Estimates x value at location 's' along curve
        using Simpson's rule (divide domain into n=8 sections) """
        n0 = cos(self.theta_s(0))
        n1 = 4*cos(self.theta_s(1*s/8))
        n2 = 2*cos(self.theta_s(2*s/8))
        n3 = 4*cos(self.theta_s(3*s/8))
        n4 = 2*cos(self.theta_s(4*s/8))
        n5 = 4*cos(self.theta_s(5*s/8))
        n6 = 2*cos(self.theta_s(6*s/8))
        n7 = 4*cos(self.theta_s(7*s/8))
        n8 = cos(self.theta_s(s))
        n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
        return self.x0 + (s/24)*(n_sum)

    def y_s(self, s):
        """ Estimates y value at location 's' along curve
        using Simpson's rule (divide domain into n=8 sections) """
        n0 = sin(self.theta_s(0))
        n1 = 4*sin(self.theta_s(1*s/8))
        n2 = 2*sin(self.theta_s(2*s/8))
        n3 = 4*sin(self.theta_s(3*s/8))
        n4 = 2*sin(self.theta_s(4*s/8))
        n5 = 4*sin(self.theta_s(5*s/8))
        n6 = 2*sin(self.theta_s(6*s/8))
        n7 = 4*sin(self.theta_s(7*s/8))
        n8 = sin(self.theta_s(s))
        n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
        return self.y0 + (s/24)*(n_sum)

    def theta_s(self, s):
        """ Finds theta value at location 's' along curve """
        s4 = self.a3_map() * s**4 / 4
        s3 = self.a2_map() * s**3 / 3
        s2 = self.a1_map() * s**2 / 2
        s1 = self.a0_map() * s
        return self.t0+s4+s3+s2+s1

    def a0_map(self):
        """ Map between optimization params and spiral coefficients. """
        return self.p0

    def a1_map(self):
        """ Map between optimization params and spiral coefficients. """
        num = -1*(11*self.p0/2 - 9*self.p1 + 9*self.p2/2 - self.p3)
        denom = self.p4
        return num/denom

    def a2_map(self):
        """ Map between optimization params and spiral coefficients. """
        num = 9*self.p0 - 45*self.p1/2 + 18*self.p2 - 9*self.p3/2
        denom = self.p4**2
        return num/denom

    def a3_map(self):
        """ Map between optimization params and spiral coefficients. """
        num = -1*(9*self.p0/2 - 27*self.p1/2 + 27*self.p2/2 - 9*self.p3/2)
        denom = self.p4**3
        return num/denom