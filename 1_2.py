import numpy as np
import random
from scipy.optimize import minimize

def distance_point_to_line_3d(p1, p2, p3):

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    vec_p1p2 = p2 - p1  
    vec_p1p3 = p3 - p1   
    
    cross_product = np.cross(vec_p1p2, vec_p1p3)
    distance = np.linalg.norm(cross_product) / np.linalg.norm(vec_p1p2)
    
    return distance

def find_true_interval(f, t1, t2, tol=1e-6):

    f_t1 = f(t1)
    f_t2 = f(t2)
    if f_t1 and f_t2:
        return t2 - t1

    elif f_t1:
        low, high = t1, t2
        while high - low > tol:
            mid = (low + high) / 2
            if f(mid):
                low = mid
            else:
                high = mid
        return high-t1

    elif f_t2:
        low, high = t1, t2
        while high - low > tol:
            mid = (low + high) / 2
            if f(mid):
                high = mid
            else:
                low = mid
        return t2-low
    
    else:
        max_tries=10000
        t_mid = None
        for _ in range(max_tries):
            t = random.uniform(t1, t2)
            if f(t) > 0:
                t_mid = t
                break
    
        if t_mid is None:
            return None  

        left, right = t1, t_mid
        while right - left > tol:
            mid = (left + right) / 2 
            if f(mid) > 0:
                right = mid
            else:
                left = mid
        t_left = (left + right) / 2

        left, right = t_mid, t2
        while right - left > tol:
            mid = (left + right) / 2
            if f(mid) > 0:
                left = mid
            else:
                right = mid
        t_right = (left + right) / 2

        return t_right-t_left

def test_t_time(t):
    cos1=10/101**0.5
    sin1=1/101**0.5
    yanwu=np.array([17188,0,1751.796-3*t])
    m1=np.array([20000-300*cos1*t,0,2000-300*sin1*t])
    def distance1(x):
        return distance_point_to_line_3d(np.array([7*np.sin(x),207-7*np.cos(x),0]),m1,yanwu)
    def distance2(x):
        return distance_point_to_line_3d(np.array([7*np.sin(x),207-7*np.cos(x),10]),m1,yanwu)
    result1 = minimize(lambda x: -distance1(x[0]),  
                  x0=np.array([0.0]),           
                  bounds=[(0, 2*np.pi)])
    d1 = -result1.fun
    x1=  result1.x[0]
    result2 = minimize(lambda x: -distance2(x[0]),  
                  x0=np.array([0.0]),          
                  bounds=[(0, 2*np.pi)])
    d2 = -result2.fun
    x2= result2.x[0]
    if np.linalg.norm(yanwu - m1)>10 and np.dot(m1-yanwu,m1-np.array([7*np.sin(x1),207-7*np.cos(x1),0]))<0 and np.dot(m1-yanwu,m1-np.array([7*np.sin(x2),207-7*np.cos(x2),10]))<0:
        return False
    elif d1<=10 and d2<=10:
        return True
    else:
        return False
    
def main():
    print (find_true_interval(test_t_time,5.1,25.1))

if __name__ == "__main__":

    main()
