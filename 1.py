import numpy as np

def distance_point_to_line_3d(p1, p2, p3):

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    vec_p1p2 = p2 - p1  
    vec_p1p3 = p3 - p1   
    
    cross_product = np.cross(vec_p1p2, vec_p1p3)
    distance = np.linalg.norm(cross_product) / np.linalg.norm(vec_p1p2)
    
    return distance

def find_intersections(p1, p2):

    x1, y1 = p1
    x2, y2 = p2

    if y1 == y2:
        return (None,None)

    if x1 == x2:
        return (x1, x1)  

    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1

    x_at_y10 = (10 - b) / k
    x_at_y0 = (0 - b) / k

    return (x_at_y10 , x_at_y0)

def find_farthest_point_on_circle(center, radius, point):

    xc, yc = center
    xp, yp = point
    r = radius

    dx = xc - xp
    dy = yc - yp

    distance_pc = np.sqrt(dx**2 + dy**2)

    if distance_pc == 0:
        return (xc + radius, yc)  

    ux = dx / distance_pc
    uy = dy / distance_pc

    t_far = distance_pc + r
    x_far = xp + t_far * ux
    y_far = yp + t_far * uy
    
    return (x_far, y_far)

def find_closest_point_on_circle(center, radius, point):

    xc, yc = center
    xp, yp = point
    r = radius

    dx = xc - xp
    dy = yc - yp

    distance_pc = np.sqrt(dx**2 + dy**2)

    if distance_pc == 0:
        return (xc + radius, yc)  

    ux = dx / distance_pc
    uy = dy / distance_pc

    t_close = distance_pc - r
    x_close = xp + t_close * ux
    y_close = yp + t_close * uy
    
    return (x_close, y_close)

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
        1



def test_t_time(t):
    cos1=100/10001**0.5
    sin1=1/10001**0.5
    yanwu=np.array([17188,0,1750.5-3*t])
    m1=np.array([20000-300*cos1*t,0,2000-300*sin1*t])
    x_at_y10,x_at_y0=find_intersections((yanwu[0],yanwu[2]),(m1[0],m1[2]))
    if m1[2]>yanwu[2]:
        x_circle1,y_circle1=find_farthest_point_on_circle((0,200),7,(x_at_y10,0))
        x_circle2,y_circle2=find_farthest_point_on_circle((0,200),7,(x_at_y0,0))
    if m1[2]<yanwu[2]:
        x_circle1,y_circle1=find_closest_point_on_circle((0,200),7,(x_at_y10,0))
        x_circle2,y_circle2=find_closest_point_on_circle((0,200),7,(x_at_y0,0))
    d1=distance_point_to_line_3d(m1,(x_circle1,y_circle1,10),yanwu)
    d2=distance_point_to_line_3d(m1,(x_circle2,y_circle2,0),yanwu)
    if d1<=10 and d2<=10:
        return True
    else:
        return False
    
def main():
    print (find_true_interval(test_t_time,5.1,25.1))

if __name__ == "__main__":
    main()