"""Our CR3BP equations, analytic Jacobian and adaptive RK4 integration."""
from math import isfinite
MU = 0.01215058560962404

def field_and_jacobian(s, mu=MU):
    x,y,z,vx,vy,vz = s
    acc = [x+2*vy, y-2*vx, 0.]
    hess = [[float(i==j and i<2) for j in range(3)] for i in range(3)]
    for mass, center in [(1-mu, -mu), (mu, 1-mu)]:
        d = [x-center,y,z]
        r2 = sum(v*v for v in d)
        if r2 < 1e-24:
            raise ValueError("Collision with primary")
        inv = r2**-1.5
        for i in range(3):
            acc[i] -= mass*d[i]*inv
            for j in range(3):
                hess[i][j] += mass*inv*(3*d[i]*d[j]/r2-(i==j))
    jac = [[0.]*6 for _ in range(6)]
    for i in range(3):
        jac[i][i+3] = 1.
        jac[i+3][:3] = hess[i]
    jac[3][4],jac[4][3] = 2.,-2.
    return [vx,vy,vz,*acc],jac

def augmented_rhs(s):
    f,jac = field_and_jacobian(s[:6])
    return f+[sum(jac[i][k]*s[6+6*k+j] for k in range(6))
              for i in range(6) for j in range(6)]

def rk4(f,s,h):
    a = f(s)
    b = f([v+h*k/2 for v,k in zip(s,a)])
    c = f([v+h*k/2 for v,k in zip(s,b)])
    d = f([v+h*k for v,k in zip(s,c)])
    return [v+h*(a+2*b+2*c+d)/6 for v,a,b,c,d in zip(s,a,b,c,d)]

def integrate(f, initial, times, tolerance=1e-10):
    if not times or times[0] != 0 or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("Times must start at zero and increase")
    state,t,step,attempts = list(initial),0.,0.01,0
    output = [state[:]]
    for target in times[1:]:
        while t < target:
            attempts += 1
            if attempts > 200000:
                raise RuntimeError("Integration step limit exceeded")
            h = min(step,target-t)
            if t+h == t:
                raise RuntimeError("Integration step underflow")
            coarse = rk4(f,state,h)
            fine = rk4(f,rk4(f,state,h/2),h/2)
            err = max(abs(a-b)/(15*tolerance*(1+max(abs(v),abs(a))))
                      for a,b,v in zip(fine,coarse,state))
            if not isfinite(err):
                raise RuntimeError("Nonfinite integration")
            if err <= 1:
                state = [a+(a-b)/15 for a,b in zip(fine,coarse)]
                t = target if h == target-t else t+h
            step = h*min(2.,max(0.2,0.9*max(err,1e-16)**-0.2))
        output.append(state[:])
    return output

def transition_matrices(initial,times,tolerance=1e-10):
    identity = [float(i==j) for i in range(6) for j in range(6)]
    solutions = integrate(augmented_rhs,[*initial,*identity],times,tolerance)
    return [[s[6+6*i:12+6*i] for i in range(6)] for s in solutions]
