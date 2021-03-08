from math import sin, cos
from scipy.integrate import quad
from scipy.optimize import minimize

# This is based on the tutorial found here: https://tinyurl.com/92dwh52
# This sample code will likely be useful: https://tinyurl.com/4br894dt

class PathGenerator:
    def __init__(self, start_x, start_y, start_theta, start_curvature, 
                        goal_x, goal_y, goal_theta, goal_curvature,
                        alpha=15, beta=25, gamma=35, kmax=0.5):
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
        # self.p0 = self.k0           # Starting point of curve
        # self.p1 = "1/3 down curve"  # 1/3 down curve
        # self.p2 = "2/3 down curve"  # 2/3 down curve
        # self.p3 = self.kf           # Ending point of curve
        # self.p4 = "Length of curve" # Length of curve
        self.p0 = self.k0
        self.p1 = 1./3.
        self.p2 = 2./3.
        self.p3 = self.kf
        self.p4 = 1.

        # Constraints for the optimization problem
        self.bound_p1 = [-1*self.kmax, self.kmax]    # p1's constraint
        self.bound_p2 = [-1*self.kmax, self.kmax]    # p2's constraint
        self.bounds = [self.bound_p1, self.bound_p2]

        # TODO: optimize self.objective_function
        # path_raw = "Optimization of self.objective_function() for params p1, p2, p4"
        # Pseudocode layout?: optimize(self.objective_function, params=[self.p1, self.p2, self.p4], bounds=self.bounds)
        x_0 = [self.p1, self.p2, self.p4]  # initial guess
        print('initial guess')
        print(self.objective_function(x_0))
        output_params = minimize(self.objective_function, x_0)
        print(output_params)
        # path = "The path above, but mapping the p values back to spiral parameters"
        self.p1 = output_params['x'][0]
        self.p2 = output_params['x'][1]
        self.p4 = output_params['x'][2]
        
        self.a_list = [self.a0_map(), self.a1_map(self.p1, self.p2, self.p4), self.a2_map(self.p1, self.p2, self.p4), self.a3_map(self.p1, self.p2, self.p4)]

        # Sample the curve to get our points
        s_i = 0
        self.interval = 0.25
        self.t_list = []
        self.k_list = []
        
        # get theta(s) and k(s) for all 's' with step size 'self.interval'
        while s_i <= self.p4:
            k_list.append(self.final_spiral(s_i))
            t_list.append(self.final_theta(s_i))
            s_i += self.interval

        # use our theta(s) values to find x(s) and y(s) values with trapezoid rule
        # the index 's' values should align with t_list and k_list (might be off by 1)
        x_list = self.x_trapezoidal()
        x_list.insert(0, self.x0)
        x_list.insert(-1, self.xf)
        
        y_list = self.y_trapezoidal()
        y_list.insert(0, self.y0)
        y_list.insert(-1, self.yf)
        
        path = [x_list, y_list, t_list, k_list]

        # information of our s values, if helpful
        s_info = {'min': 0, 'max':s_i, 'interval':self.interval}

        return path, s_info

    def final_spiral(self, s):
        """ Our final k(s) equation """
        return self.a_list[3]*s**3 + self.a_list[2]*s**2 + self.a_list[1]*s + self.a_list[0]

    def final_theta(self, s):
        """ Our final theta(s) equation """
        return self.t0 + self.a_list[3]/4*s**4 + self.a_list[2]/3*s**3 + self.a_list[1]/2*s**2 + a_list[0]*s

    def x_trapezoidal(self):
        """ Uses the trapezoidal rule to estimate x. 
        x = x0 + (cos(s)+cos(s+interval))*interval/2
        We use indeces to access our precomputed cos(s) values
        spaced out by our interval (hence the index approach)"""
        x_list = []
        # Calculates values up to the second to last index
        for s_i in range(len(self.t_list)-2):
            x_s = self.x0 + (1/2)*(cos(self.t_list[s_i])+cos(self.t_list[s_i+1]))*self.interval
            x_list.append(x_s)
        return x_list[]

    def y_trapezoidal(self):
        """ Uses the trapezoidal rule to estimate y. 
        y = y0 + (sin(s)+sin(s+interval))*interval/2
        We use indeces to access our precomputed sin(s) values
        spaced out by our interval (hence the index approach)"""
        y_list = []
        # Calculates values up to the second to last index
        for s_i in range(len(self.t_list)-2):
            y_s = self.y0 + (1/2)*(sin(self.t_list[s_i])+sin(self.t_list[s_i+1]))*self.interval
            y_list.append(y_s)
        return y_list[]

    def objective_function(self, x):
        """ the parameters to optimize are p1, p2, and p4 """
        p1 = x[0]
        p2 = x[1]
        p4 = x[2]

        # objective function needs to return a scalar value for scipy.optimize to work - Nicole
        result = self.f_be(p1, p2, p4) + self.x_soft(p1, p2, p4) + self.y_soft(p1, p2, p4) + self.theta_soft(p1, p2, p4)
        return result[0]

    def k_s(self, s, p1, p2, p4):
        """ Our cubic spiral equation. Not sure if we need this """
        return self.a3_map(p1, p2, p4)*s**3 + self.a2_map(p1, p2, p4)*s**2 + self.a1_map(p1, p2, p4)*s + self.a0_map()

    def f_be_integrand(self, s, a0, a1, a2, a3):
        """ Integrand to use with f_be() 
        This is our cubic spiral equation squared 
        integrated for variable 's' from 0 to p4"""
        return (a3*s**3 + a2*s**2 + a1*s + a0)**2
    
    def f_be(self, p1, p2, p4):
        """ Unconstrained objective function, using the quad integral solver
        from SciPy on our objective_integrand (variable 's') 
        from 0 to curve length p4, using coefficients a0, a1, a2, and a3 """
        spiral_vals = (self.a0_map(), self.a1_map(p1, p2, p4), self.a2_map(p1, p2, p4), self.a3_map(p1, p2, p4))
        return quad(self.f_be_integrand, 0, p4, args=spiral_vals)

    def x_soft(self, p1, p2, p4):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by alpha. """
        s = p4
        return self.alpha*(self.x_s(s, p1, p2, p4) - self.xf)

    def y_soft(self, p1, p2, p4):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by beta. """
        s = p4
        return self.beta*(self.y_s(s, p1, p2, p4) - self.yf)

    def theta_soft(self, p1, p2, p4):
        """ Soft inequality constraints, allows a small
        margin of error between goal point and final point
        in the curve. Scaled by gamma. """
        s = p4
        return self.gamma*(self.theta_s(s, p1, p2, p4) - self.tf)

    def x_s(self, s, p1, p2, p4):
        """ Estimates x value at location 's' along curve
        using Simpson's rule (divide domain into n=8 sections) """
        n0 = cos(self.theta_s(0, p1, p2, p4))
        n1 = 4*cos(self.theta_s(1*s/8, p1, p2, p4))
        n2 = 2*cos(self.theta_s(2*s/8, p1, p2, p4))
        n3 = 4*cos(self.theta_s(3*s/8, p1, p2, p4))
        n4 = 2*cos(self.theta_s(4*s/8, p1, p2, p4))
        n5 = 4*cos(self.theta_s(5*s/8, p1, p2, p4))
        n6 = 2*cos(self.theta_s(6*s/8, p1, p2, p4))
        n7 = 4*cos(self.theta_s(7*s/8, p1, p2, p4))
        n8 = cos(self.theta_s(s, p1, p2, p4))
        n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
        return self.x0 + (s/24)*(n_sum)

    def y_s(self, s, p1, p2, p4):
        """ Estimates y value at location 's' along curve
        using Simpson's rule (divide domain into n=8 sections) """
        n0 = sin(self.theta_s(0, p1, p2, p4))
        n1 = 4*sin(self.theta_s(1*s/8, p1, p2, p4))
        n2 = 2*sin(self.theta_s(2*s/8, p1, p2, p4))
        n3 = 4*sin(self.theta_s(3*s/8, p1, p2, p4))
        n4 = 2*sin(self.theta_s(4*s/8, p1, p2, p4))
        n5 = 4*sin(self.theta_s(5*s/8, p1, p2, p4))
        n6 = 2*sin(self.theta_s(6*s/8, p1, p2, p4))
        n7 = 4*sin(self.theta_s(7*s/8, p1, p2, p4))
        n8 = sin(self.theta_s(s, p1, p2, p4))
        n_sum = n0+n1+n2+n3+n4+n5+n6+n7+n8
        return self.y0 + (s/24)*(n_sum)

    def theta_s(self, s, p1, p2, p4):
        """ Finds theta value at location 's' along curve """
        s4 = self.a3_map(p1, p2, p4) * s**4 / 4
        s3 = self.a2_map(p1, p2, p4) * s**3 / 3
        s2 = self.a1_map(p1, p2, p4) * s**2 / 2
        s1 = self.a0_map() * s
        return self.t0+s4+s3+s2+s1

    def a0_map(self):
        """ Map between optimization params and spiral coefficients. """
        return self.p0

    def a1_map(self, p1, p2, p4):
        """ Map between optimization params and spiral coefficients. """
        num = -1*(11*self.p0/2 - 9*p1 + 9*p2/2 - self.p3)
        denom = p4
        return num/denom

    def a2_map(self, p1, p2, p4):
        """ Map between optimization params and spiral coefficients. """
        num = 9*self.p0 - 45*p1/2 + 18*p2 - 9*self.p3/2
        denom = p4**2
        return num/denom

    def a3_map(self, p1, p2, p4):
        """ Map between optimization params and spiral coefficients. """
        num = -1*(9*self.p0/2 - 27*p1/2 + 27*p2/2 - 9*self.p3/2)
        denom = p4**3
        return num/denom
