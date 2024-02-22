import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7)
plt.style.use('dark_background')
plt.rcParams["font.size"] = 15
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import cv2 as cv
import os
import math
import yaml

# import scipy.stats as stats
# import scipy.optimize as optimize
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

modes = ['inactive', 'refraction', 'reflection', 'partial', 'absorption',"diffuse","detector"]

lens_display_theta=10    #theta= longitudinal resolution
lens_display_phi=20      #phi= circular resolution
ray_display_density=0.5
obj_display_density=0.5
show_grid = True
fill_with_grey = False
density_for_intensity=True
show_plot = True
coexist = True
#    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

angle_to_normal = 'element' #choose between element and hit
hit_coordinates = 'absolute' #choose between absolute and relative

def normalize(vec):
    scale = 1 / np.linalg.norm(vec[0:3])
    return scale * vec

def xdot(mat, vm):  # if vm is matrix, its shape must be n by 3 or 4
    if len(vm.shape) == 1:
        if len(mat) == 4 and len(vm) == 3:
            v = np.append(vm, 1)
            return np.dot(mat, v)[0:3]
        if len(mat) == 3 and len(vm) == 4:
            return np.dot(mat, vm[0:3])
        else:
            return np.dot(mat, vm)[0:3]
    else:
        if np.min(mat.shape) == 4 and vm.shape[1] == 3:
            vm = np.concatenate([vm, np.ones((vm.shape[0], 1))], axis=1)
        out = np.dot(mat, vm.transpose())
        return out.transpose()[:, 0:3]

def wavelength2rgb(wavelength, gamma=0.8):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return R, G, B

def upright_rot_transform(a, b):
    if np.dot(a, b) == 1:
        return np.eye(3)
    elif np.dot(a, b) == -1:
        return -1 * np.eye(3)
    elif np.dot(a,b)==0:
        return rot_transform(a,b)
    fv1 = normalize(np.array([a[0], a[1], 0]))
    fv2 = normalize(np.array([b[0], b[1], 0]))
    theta = -np.arccos(np.dot(fv1, fv2))
    if b[1] < 0:  # Not elegant.
        theta = 2 * np.pi - theta
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])

    a_rotated = np.dot(Rz, a)
    effect = np.dot(a_rotated,b)
    if effect>1:
        effect=1
    elif effect<-1:
        effect=-1
    phi = np.arccos(effect)
    if phi == 0:
        return Rz
    axis = np.cross(a_rotated, b)
    axis /= np.linalg.norm(axis)

    R_phi = np.array([[np.cos(phi) + axis[0] ** 2 * (1 - np.cos(phi)),
                       axis[0] * axis[1] * (1 - np.cos(phi)) - axis[2] * np.sin(phi),
                       axis[0] * axis[2] * (1 - np.cos(phi)) + axis[1] * np.sin(phi)],
                      [axis[1] * axis[0] * (1 - np.cos(phi)) + axis[2] * np.sin(phi),
                       np.cos(phi) + axis[1] ** 2 * (1 - np.cos(phi)),
                       axis[1] * axis[2] * (1 - np.cos(phi)) - axis[0] * np.sin(phi)],
                      [axis[2] * axis[0] * (1 - np.cos(phi)) - axis[1] * np.sin(phi),
                       axis[2] * axis[1] * (1 - np.cos(phi)) + axis[0] * np.sin(phi),
                       np.cos(phi) + axis[2] ** 2 * (1 - np.cos(phi))]])

    rotation_matrix = np.dot(R_phi, Rz)
    return rotation_matrix


def rot_transform(a, b):
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # assuming b is the normal vector to align surface to
    if np.dot(a, b) == 1:
        return np.eye(3)
    elif np.dot(a, b) == -1:
        return -1 * np.eye(3)
    v = np.cross(a, b[0:3])
    s = np.linalg.norm(v)  # sine
    c = np.dot(a, b)  # cosine
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / s / s
    return r

def axial_rotation(axis, rad):
    theta = rad
    u_x = axis[0]
    u_y = axis[1]
    u_z = axis[2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rhr = np.array([
        [cos_theta + (1 - cos_theta) * u_x ** 2, (1 - cos_theta) * u_x * u_y - sin_theta * u_z,
         (1 - cos_theta) * u_x * u_z + sin_theta * u_y],
        [(1 - cos_theta) * u_x * u_y + sin_theta * u_z, cos_theta + (1 - cos_theta) * u_y ** 2,
         (1 - cos_theta) * u_y * u_z - sin_theta * u_x],
        [(1 - cos_theta) * u_x * u_z - sin_theta * u_y, (1 - cos_theta) * u_y * u_z + sin_theta * u_x,
         cos_theta + (1 - cos_theta) * u_z ** 2]
    ])
    return rhr

def ang2vec(angles):
    theta = np.radians(angles[0] + 180)
    phi = np.radians(angles[1])
    nor = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
    return nor

def simple_convex_hull(fourpoints, cand):
    fp = fourpoints.reshape((4, 2))
    cand = cand.reshape(2)
    vectors = fp - cand
    cross = np.cross(vectors, np.roll(vectors, -1, axis=0))
    if all(cp >= 0 for cp in cross) or all(cp <= 0 for cp in cross):
        return 1
    else:
        return 0

def point_light(samples,divergence = np.pi):
    #Utilizes fibonacci sphere and surface area of spherical segment
    #divergence is the half angle of beam
    h = 1 - np.cos(divergence)
    valid_area = 2*np.pi*h
    total_area = 4*np.pi
    oversampling = int(total_area/valid_area * samples)

    phi = np.pi * (3. - np.sqrt(5.))

    axial = np.linspace(1,-1,oversampling)
    r_cross = np.sqrt(1-np.square(axial))
    theta = np.arange(oversampling)*phi
    xs = axial
    ys = r_cross * np.cos(theta)
    zs = r_cross * np.sin(theta)
    points = np.stack([xs,ys,zs],axis=1)
    return points[0:samples,:]

class ray:
    def __init__(self, container, position=None, vector=None, wavelength=0):
        self.coord = np.zeros(3, np.float64)
        self.container = container
        self.vec = np.array([1, 0, 0], np.float64)
        self.trace = []
        self.active = True
        self.active_since = 0
        self.active_until = -1
        self.intensity = 1
        self.intensities=[[0,1]]
        if not position is None and not vector is None:
            self.coord = position
            self.vec[0:3] = vector[0:3]
            self.trace.append(position)
        self.wv = wavelength
        if self.wv ==0:
            self.color = (1,1,1)
        else:
            self.color = wavelength2rgb(self.wv)

        self.peer_dist=0
    def travel(self):
        points = np.asarray(self.trace)
        steps = np.diff(points, axis=0)
        steps = np.linalg.norm(steps, axis=1)
        dist = np.sum(steps)
        return dist

    def change_wl(self, wl):
        self.wv = wl
        self.color = wavelength2rgb(self.wv)

    def add_siblings(self, sib):
        sib.coord = sib.trace[-1]
        sib.trace = sib.trace[1:]
        sib.container = self.container
        self.container.append(sib)

class surface:
    def __init__(self,
                 coord, normal, shape,
                 angles=None, radius=1, semidia=1.0,
                 dial=0, height=1, width=1,
                 mode="inactive", n1=1, n2=1, transmission=1,
                 color=None, alpha=obj_display_density):
        if coord is None:
            coord = np.array([0,0,0])
        self.vertex = coord
        self.normal = np.array([-1, 0, 0], np.float64)
        self.dial = dial
        self.intersects=[]
        if not normal is None:
            self.normal = normalize(normal)
        elif normal is None and not angles is None:
            nor = ang2vec(angles)
            self.normal = normalize(nor)

        self.shape = shape
        self.cylindrical = False
        self.disk = False

        self.radius = radius  # +( -)
        self.semidia = semidia
        self.height=height
        self.width=width
        self.base = np.array([[0, -semidia, -semidia],
                              [0, -semidia, semidia],
                              [0, semidia, semidia],
                              [0, semidia, -semidia]])

        if shape == "spherical":
            self.dial = 0
        elif shape == "cylindrical":
            self.height = height
            self.width = width
            self.base = np.array([[0, -width / 2, -height / 2],
                                  [0, -width / 2, height / 2],
                                  [0, width / 2, height / 2],
                                  [0, width / 2, -height / 2]])
            self.cylindrical = True
            if abs(self.radius) < abs(0.5*self.width):
                self.radius = 1
                self.width = 1
        elif shape == "plano":
            self.radius = 0
            if isinstance(semidia,float) and semidia==1: #which means the default param is not used, its a round mirror
                self.disk = False
                self.height = height
                self.width = width
                self.base = np.array([[0, -width / 2, -height / 2],
                                      [0, -width / 2, height / 2],
                                      [0, width / 2, height / 2],
                                      [0, width / 2, -height / 2]])
            else:
                self.disk = True
                self.semidia = semidia
                self.base = np.array([[0, -semidia, -semidia],
                                      [0, -semidia, semidia],
                                      [0, semidia, semidia],
                                      [0, semidia, -semidia]])

        self.n2 = n2
        self.n1 = n1
        self.mode = mode
        self.transmission = transmission

        self.move = np.eye(4)  # transform incoming ray into surface definition coords
        self.inverse = np.eye(4)  # transform coords from surface definition coords to real world for viz, hence inverse

        self.calc_mat()

        self.color = (1,1,1)
        if isinstance(color,int):
            self.color=wavelength2rgb(color)
        self.alpha = alpha

        if not color is None:
            self.color = color
            self.alpha = alpha

        self.sid = 0
        self.rendered=False

        self.going=None
        self.fan=0
    def copy(self,afresh = False):
        neu = surface(self.vertex.copy(),self.normal.copy(),self.shape)
        neu.dial = self.dial
        neu.mode = self.mode
        neu.n1 = self.n1
        neu.n2 = self.n2
        neu.cylindrical = self.cylindrical
        neu.disk = self.disk
        neu.radius = self.radius
        neu.semidia = self.semidia
        neu.base = self.base.copy()
        neu.height = self.height
        neu.width = self.width
        neu.mode = self.mode
        neu.transmission = self.transmission
        neu.move = self.move.copy()
        neu.inverse = self.inverse.copy()
        neu.color = self.color
        neu.alpha = self.alpha
        neu.rendered = False
        if afresh is False:
            neu.rendered = True
        return neu
    def relocate(self,coord,normal=None,angles=None):
        self.vertex = coord
        if not normal is None:
            self.normal = normal
        elif not angles is None:
            self.normal = ang2vec(angles)
        self.calc_mat()

    def calc_mat(self):
        #r=rot_transform(np.array([-1,0,0]),self.normal)
        r = upright_rot_transform(np.array([-1, 0, 0]), self.normal)
        t = self.vertex - self.normal * self.radius
        ang = self.dial
        if ang != 0:
            theta = np.radians(ang)
            rhr = axial_rotation(self.normal, theta)
            r = np.dot(rhr, r)
        self.inverse *= 0
        self.inverse[0:3, 0:3] = r
        self.inverse[0:3, 3] = t
        self.inverse[3, 3] = 1

        self.move = np.linalg.inv(self.inverse)

    def translate(self, shift):
        self.vertex = self.vertex + shift
        self.move = np.eye(4)  # transform incoming ray into surface definition coords
        self.inverse = np.eye(4)  # transform coords from surface definition coords to real world for viz, hence inverse
        self.calc_mat()
class assembly:
    def __init__(self):
        self.surfaces=[]
        self.position=np.zeros(3)
        self.normal=np.array([-1,0,0])
        self.normals=[]
        self.positions=[]

        self.last_surface= np.zeros(3)

    def add(self, s:surface, relative, normal=None, angles=None, dial=0):
        if not normal is None:
            nor = normal
        elif normal is None and not angles is None:
            nor = ang2vec(angles)
        else:
            nor = np.array([-1,0,0])
        if isinstance(relative,float) or isinstance(relative,int):
            relative = np.array([relative,0,0])
        vertex = self.last_surface + relative
        s.vertex=vertex
        s.normal=nor
        self.normals.append(nor)
        self.positions.append(vertex)
        self.last_surface=vertex

        self.surfaces.append(s)

    def place(self,position,vector=None,angles=None):
        if not angles is None:
            vector = ang2vec(angles)
        vector=normalize(vector)
        self.position=position
        mat = upright_rot_transform(np.array([-1,0,0]),vector)
        for i,s in enumerate(self.surfaces):
            aligned = np.dot(mat, s.normal)
            s.normal = aligned
            relative = self.positions[i]
            if np.linalg.norm(relative)==0:
                s.vertex=self.position
                s.calc_mat()
                continue
            relative = np.dot(mat,relative)
            s.vertex = self.position+relative
            s.calc_mat()
        self.last_surface=self.last_surface+position
        self.positions.append(self.last_surface)

    def extend(self,additional:'assembly',relative, where="tail", normal=None):
        if np.dot(self.normal,additional.normal)<1:
            return
        if isinstance(relative, float) or isinstance(relative, int):
            relative = np.array([relative, 0, 0])
        for i,s in enumerate(additional.surfaces):
            if where=="tail" or i>0:
                s.vertex = s.vertex + self.last_surface
            elif where=="head":
                s.vertex = s.vertex + self.position
            if i == 0:
                s.vertex+=relative
            self.surfaces.append(s)
            self.normals.append(s.normal)
            self.positions.append(s.vertex)
            self.last_surface = s.vertex

    def invert(self):
        bef = np.stack(self.positions)
        diff = bef[1:]-bef[0:-1]
        diff = diff[::-1,:]
        self.surfaces.reverse()
        for i in range(1,len(self.positions)):
            self.positions[i]=diff[i-1,:]+self.positions[i-1]
        for i,s in enumerate(self.surfaces):
            swa=s.n1
            s.n1=s.n2
            s.n2=swa
            if s.shape in ["spherical","cylindrical"]:
                s.radius = -s.radius
        self.last_surface = self.positions[-1]
    def copy(self):
        neu = assembly()
        neu.surfaces=[]
        for i,s in enumerate(self.surfaces):
            ns = s.copy(True)
            neu.surfaces.append(ns)
            neu.normals.append(self.normals[i].copy())
            neu.positions.append(self.positions[i].copy())
        neu.normal=np.array([-1,0,0])
        neu.position=np.zeros(3)
        neu.last_surface=self.last_surface.copy()
        return neu
def interact_vhnrs(v, h, n, r: ray, s: surface, forced=None):  # vector, hit, normal, ray2bupdated, surface
    v_rel_from = v.copy()
    v_abs_from = v.copy()
    if s.mode == "refraction" and forced is None or forced == "refraction":
        # vector representation of snell's law https://en.wikipedia.org/wiki/Snell%27s_law
        snell = s.n1 / s.n2
        if snell > 1:
            critical = np.arcsin(s.n2 / s.n1)
            theta = np.arccos(max(np.dot(n, v), np.dot(-n, v)))
            if theta > critical:
                interact_vhnrs(v, h, n, r, s, forced="reflection")
                return
        c1 = -1 * np.dot(n, v)
        if c1 < 0:
            n = -1 * n
            c1 = -1 * np.dot(n, v)
        c2 = np.sqrt(1 - snell * snell * (1 - c1 * c1))
        refracted = snell * v + (snell * c1 - c2) * n
        towards = h + refracted
        r_hit = xdot(s.inverse, h)
        r_towards = xdot(s.inverse, towards)
        v = normalize(r_towards - r_hit)
        r.vec = v
        r.trace.append(r_hit)
        #store ray stats
        stat = [r_hit]
        ray_sta = extract_ray_info(r)
        ray_sta.append(v_rel_from)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        return
    if s.mode == "reflection" and forced is None or forced == "reflection":
        reflected = v - 2 * np.dot(v, n) * n
        towards = h + reflected
        r_hit = xdot(s.inverse, h)
        r_towards = xdot(s.inverse, towards)
        v = normalize(r_towards - r_hit)
        r.vec = v
        r.trace.append(r_hit)
        #store ray stats
        stat = [r_hit]
        ray_sta = extract_ray_info(r)
        ray_sta.append(v_rel_from)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        return
    if s.mode == "partial" and forced is None:
        r_hit = xdot(s.inverse, h)
        shadow = ray(None, r_hit - 1e-3 * r.vec, r.vec.copy(), r.wv)
        aa=s.transmission
        bb=1-s.transmission
        if s.transmission > 0.5:
            interact_vhnrs(v, h, n, r, s, forced="refraction")
            interact_vhnrs(v, h, n, shadow, s, forced="reflection")
        else:
            interact_vhnrs(v, h, n, r, s, forced="reflection")
            interact_vhnrs(v, h, n, shadow, s, forced="refraction")
            aa=1-s.transmission
            bb=s.transmission
        shadow.active_since = s.sid + 1
        pri=r.intensity*aa
        r.intensities.append([s.sid - r.active_since + 1, pri])
        sec=r.intensity * bb
        shadow.intensities = [[0,sec]]
        r.intensity=pri
        shadow.intensity=sec
        shadow.intensities=[[0,sec]]
        r.add_siblings(shadow)
        return
    if s.mode == "absorption":
        r_hit = xdot(s.inverse, h)
        r.trace.append(r_hit)
        #store ray stats
        stat = [r_hit]
        ray_sta = extract_ray_info(r)
        ray_sta.append(v_rel_from)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        r.active = False
        r.active_until=s.sid
        return
    if s.mode =="diffuse":
        ctr = s.going
        r_hit = xdot(s.inverse, h)
        r_towards = xdot(s.inverse, ctr)
        v=normalize(r_towards-r_hit)
        v=not_yet_lambertian_reflection(v,s.fan)
        r.vec=v
        r.trace.append(r_hit)
        #store ray stats
        stat = [r_hit]
        ray_sta = extract_ray_info(r)
        ray_sta.append(v_rel_from)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        return
def extract_ray_info(incident:ray):
    return [incident.wv,incident.intensity]
def not_yet_lambertian_reflection(vec,rad):
    theta = np.random.rand()*2*np.pi
    phi=(np.random.rand()-0.5)*2*rad
    r = np.sin(phi)
    x = np.cos(theta)*r
    y = np.sin(theta)*r
    z = np.cos(phi)
    ran = np.array([x,y,z])
    move = rot_transform(np.array([0,0,1]),vec)
    return xdot(move,ran)

def interact_sphere(incident: ray, sur: surface):
    # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    r = abs(sur.radius)
    last = incident.trace[-1]
    towards = last + incident.vec
    t_last = xdot(sur.move, last)
    t_towards = xdot(sur.move, towards)
    v = normalize(t_towards - t_last)

    if np.dot(-v, np.array([-1, 0, 0])) < 0:
        # backstabbing
        msg=",".join(["backstabbing","sid:"+str(sur.sid),"shape:"+sur.shape])
        #print(msg)
        return

    c = np.linalg.norm(t_last)
    c = c * c - r * r
    delta = (np.dot(v, t_last)) ** 2 - c

    if delta > 0:
        d = -np.dot(v, t_last)
        d1 = d - np.sqrt(delta)
        d2 = d + np.sqrt(delta)
        d1 = t_last + d1 * v
        d2 = t_last + d2 * v
        if sur.radius > 0:
            d = d1
        else:
            d = d2
        # d is the coordinates for interaction
        if np.linalg.norm(d[1:3]) - 1e-6 > sur.semidia:
            return
        n = normalize(-1 * d)
        interact_vhnrs(v, d, n, incident, sur)

def interact_cylinder(incident: ray, sur: surface):
    r = abs(sur.radius)
    last = incident.trace[-1]
    towards = last + incident.vec
    t_last = xdot(sur.move, last)
    t_towards = xdot(sur.move, towards)
    v = normalize(t_towards - t_last)
    if np.dot(-v, np.array([-1, 0, 0])) < 0:
        msg=",".join(["backstabbing","sid:"+str(sur.sid),"shape:"+sur.shape])
        #print(msg)
        return
    v_flat = normalize(np.array([v[0], v[1], 0]))
    t_flat = np.array([t_last[0], t_last[1], 0])
    c = np.linalg.norm(t_flat)
    c = c * c - r * r
    delta = (np.dot(v_flat, t_flat)) ** 2 - c
    if delta > 0:
        d = -np.dot(v_flat, t_flat)
        d1 = d - np.sqrt(delta)
        d2 = d + np.sqrt(delta)
        d1 = t_flat + d1 * v_flat
        d2 = t_flat + d2 * v_flat
        if sur.radius > 0:
            d = d1
        else:
            d = d2
        if abs(d[1] / r) - 1e-6 > sur.width / r / 2:
            return
        if v[0] ==0:
            return
        dz = t_last + abs(d[0] - t_last[0]) / abs(v[0]) * v
        if abs(dz[2]) - 1e-6 > sur.height / 2:
            return
        n = normalize(-1 * d)
        interact_vhnrs(v, dz, n, incident, sur)

def interact_plane(incident: ray, sur: surface):
    if np.dot(incident.vec,sur.normal)>=0:
        msg=",".join(["backstabbing","sid:"+str(sur.sid),"shape:"+sur.shape])
        #print(msg)
        return
    n = np.array([-1, 0, 0])
    last = incident.trace[-1]
    towards = last + incident.vec
    t_last = xdot(sur.move, last)
    t_towards = xdot(sur.move, towards)
    v = normalize(t_towards - t_last)
    if v[0] == 0:
        return
    hit = t_last - t_last[0] / v[0] * v
    f = hit[1:3]
    if sur.disk is False:
        validate = simple_convex_hull(sur.base[:, 1:3], f)
        if validate == 0:
            return
    else:
        if np.linalg.norm(f)>sur.semidia:
            return
    interact_vhnrs(v, hit, n, incident, sur)

def lens_vertices(radius, semidia):
    r = abs(radius)
    if semidia >= r:
        semidia = r
    rad = np.arcsin(semidia / r)
    theta = np.linspace(0, rad, lens_display_theta)

    phi = np.linspace(0, 2 * np.pi, lens_display_phi)
    theta, phi = np.meshgrid(theta, phi)

    z = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    x = -radius * np.cos(theta)

    x = x.transpose().flatten()
    y = y.transpose().flatten()
    z = z.transpose().flatten()

    coordinates = np.stack([x, y, z], axis=1)

    return coordinates

def cylinder_vertices(radius, height, width):
    r = abs(radius)
    rad = np.arcsin(0.5*width / r)
    theta = np.linspace(-rad, rad, lens_display_phi)

    x = -1 * radius * np.cos(theta)
    y = r * np.sin(theta)
    z = 0.5 * height * np.ones_like(x)

    upper = np.stack([x, y, z], axis=1)
    lower = np.stack([x, y, -z], axis=1)
    coordinates = np.zeros((2, len(upper), 3))
    coordinates[0, :, :] = upper
    coordinates[1, :, :] = lower
    return coordinates

def plane_coords(sur: surface):
    coord = xdot(sur.inverse, sur.base)
    out = [[]]
    for i in range(len(coord)):
        out[0].append(tuple(coord[i, 0:3]))
    return out

def cylindrical_coords(sur: surface):
    base = cylinder_vertices(sur.radius, sur.height, sur.width)
    upper = xdot(sur.inverse, base[0, :, :])
    lower = xdot(sur.inverse, base[1, :, :])
    base[0, :, :] = upper
    base[1, :, :] = lower

    poly3d = []
    for i in range(base.shape[1] - 1):
        a = tuple(base[0, i, :])
        b = tuple(base[0, i + 1, :])
        c = tuple(base[1, i + 1, :])
        d = tuple(base[1, i, :])
        poly3d.append([a, b, c, d])

    return poly3d

def spherical_coords(sur: surface):
    if sur.disk is False:
        base = lens_vertices(sur.radius, sur.semidia)
        base[:, 0] += sur.radius
        coord = xdot(sur.inverse, base)
        coord = coord + sur.radius * sur.normal
        x = coord[:, 0].reshape((lens_display_theta, lens_display_phi))
        y = coord[:, 1].reshape((lens_display_theta, lens_display_phi))
        z = coord[:, 2].reshape((lens_display_theta, lens_display_phi))

        verts = [list(zip(x[i], y[i], z[i])) for i in range(lens_display_theta)]

        poly3d = [[verts[i][j], verts[i][j + 1], verts[i + 1][j + 1], verts[i + 1][j]]
                  for i in range(lens_display_theta - 1) for j in range(lens_display_phi - 1)]
        out = poly3d
        return out
    else:
        rad = np.linspace(-np.pi,np.pi,lens_display_phi)
        y = sur.semidia * np.cos(rad)
        z = sur.semidia * np.sin(rad)
        x = np.zeros_like(z)
        coords = np.stack([x,y,z],axis=1)
        coords = xdot(sur.inverse,coords)
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]

        verts = [list(zip(x, y, z))]
        return verts

def plot_surface(ax: Axes3D, sur: surface, normal=False):
    if sur.rendered is True:
        return
    if sur.radius == 0:
        if sur.disk is True:
            coords = spherical_coords(sur)
        else:
            coords = plane_coords(sur)
    else:
        if sur.cylindrical is True:
            coords = cylindrical_coords(sur)
        else:
            coords = spherical_coords(sur)
    ax.add_collection3d(Poly3DCollection(coords, facecolor=sur.color, alpha=sur.alpha))
    if normal is True:
        ver = sur.vertex
        nor = sur.normal * 0.1 * max(sur.semidia,(max(sur.height,sur.width)))
        x = [ver[0], ver[0] + nor[0]]
        y = [ver[1], ver[1] + nor[1]]
        z = [ver[2], ver[2] + nor[2]]
        ax.plot(x, y, z, color=(1,1,1), alpha=obj_display_density)
    sur.rendered = True

def plot_ray(ax: Axes3D, incident: ray):
    if len(incident.trace) >= 1:
        xs = []
        ys = []
        zs = []
        tail=np.array([len(incident.trace),incident.intensity])
        intensities = np.concatenate([incident.intensities,tail.reshape((1,2))])
        density = intensities
        if density_for_intensity is False:
            density[:,1] = 1
        for i in range(len(incident.trace)):
            pt = incident.trace[i]
            xs.append(pt[0])
            ys.append(pt[1])
            zs.append(pt[2])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        for i in range(len(density)-1):
            sta = int(density[i][0])
            end = int(density[i+1][0])+1
            alp = density[i][1]
            ax.plot(xs[sta:end], ys[sta:end], zs[sta:end], color=incident.color, alpha=ray_display_density * alp)
        overshoot = incident.active
        if overshoot:
            travelled= incident.peer_dist
            xo = [xs[-1],xs[-1]+incident.vec[0] * travelled * 0.1]
            yo = [ys[-1],ys[-1]+incident.vec[1] * travelled * 0.1]
            zo = [zs[-1],zs[-1]+incident.vec[2] * travelled * 0.1]
            ax.plot(xo, yo, zo, linestyle="dotted", color=incident.color, alpha=ray_display_density * density[-1][1])

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if matplotlib.__version__ > "3.3.0":
        ax.set_box_aspect((np.ptp(np.linspace(x_middle - plot_radius, x_middle + plot_radius, 3)),
                           np.ptp(np.linspace(y_middle - plot_radius, y_middle + plot_radius, 3)),
                           np.ptp(np.linspace(z_middle - plot_radius, z_middle + plot_radius, 3))))
    if show_grid is False:
        ax.set_axis_off()
    elif fill_with_grey is False:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

def save_figure(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #cv.imwrite("rendering.jpg", cv.cvtColor(data, cv.COLOR_BGR2RGB))

class light:
    def __init__(self, position, vector, number=6,wavelength=0):
        self.lid=0
        self.rays = []
        self.position=position
        self.vector=normalize(vector)
        self.number=number
        self.divergence=0
        self.wavelength=wavelength
        self.birth = "linear"
    def reassign_container(self,container):
        for r in self.rays:
            r.container = container
    def linear(self,width,dial=0):
        self.birth = "linear"
        rad=np.radians(dial)
        core = np.zeros((self.number, 3))
        core[:, 0] = np.linspace(-width/2,width/2,self.number)
        core[:, 1] = core[:,0]*np.cos(rad+np.pi/2)
        core[:, 2] = core[:,0]*np.sin(rad+np.pi/2)
        core[:, 0] = 0
        transformed = self.rotate(self.vector,core) + self.position
        for i in range(len(transformed)):
            self.rays.append(ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength))
    def ring(self,r, whr=1, dial=0):
        self.birth = "ring"
        w=1
        h=1
        if whr>1:
            h=h/whr
        elif whr<1:
            w*=whr
        rad=np.radians(dial)
        rhr=axial_rotation(np.array([-1,0,0]),rad)
        core = np.zeros((self.number, 3))
        core[:, 0] = np.linspace(0, 2 * np.pi, self.number+1)[0:-1]
        core[:, 1] = np.cos(core[:, 0]) * r * w
        core[:, 2] = np.sin(core[:, 0]) * r * h
        core[:, 0] = 0
        core=np.dot(core,rhr)
        transformed = self.rotate(self.vector,core) + self.position
        for i in range(len(transformed)):
            self.rays.append(ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength))
    def uniform(self,r):
        self.birth="uniform"
        # distribute n points into concentric rings
        n = self.number
        nol = int(np.power(2 * (n - 1), 0.3))
        if nol == 1:
            self.ring(r)
            return
        incre = int(2 * (n - 1) / (nol * nol - nol))
        T = np.arange(0, nol * incre, incre)
        R = np.linspace(0,r,nol)
        def rtpairs(r, n):
            for i in range(len(r)):
                for j in range(n[i]):
                    yield r[i], j * (2 * np.pi / n[i])
        ys=[0]
        zs=[0]
        for r,t in rtpairs(R,T):
            ys.append(r*np.cos(t))
            zs.append(r*np.sin(t))
        x = np.zeros(len(ys))
        y = np.asarray(ys)
        z = np.asarray(zs)
        core = np.zeros((len(x), 3))
        core[:, 0] = 0
        core[:, 1] = y
        core[:, 2] = z
        transformed = self.rotate(self.vector, core) + self.position
        for i in range(len(transformed)):
            self.rays.append(
                ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength))

    def point(self,divergence=0):
        self.birth = "point"
        self.divergence = divergence
        if divergence>0:
            core = point_light(self.number,divergence)
            r = rot_transform(np.array([1,0,0]),self.vector)
            transformed = np.dot(r,core.transpose()).transpose()
            for i in range(len(transformed)):
                self.rays.append(ray(container=self.rays, position=self.position, vector=transformed[i,:], wavelength=self.wavelength))

    def rotate(self, vector, plotted):
        rot = rot_transform(np.array([1, 0, 0]), vector)
        #rot = upright_rot_transform(np.array([1, 0, 0]), vector)
        transformed = np.dot(rot, plotted.transpose())
        return transformed.transpose()

class train:
    def __init__(self, show_normal=True, show_surfaces=True):
        self.surfaces = []
        self.light = None
        self.rays = []

        self.show_nor = show_normal
        self.show_surfaces = show_surfaces
        self.extremes = np.zeros((3, 2))
        self.longest = 0

    def set_light(self, source: light):
        self.light = source
        self.rays = source.rays
    def extend_light(self,additional: light):
        additional.reassign_container(self.rays)
        self.rays.extend(additional.rays)
    def add(self, assem: assembly):
        for s in assem.surfaces:
            self.append(s)

    def append(self, sur: surface):
        self.surfaces.append(sur)
        for i in range(3):
            v = sur.vertex[i]
            if v < self.extremes[i, 0]:
                self.extremes[i][0] = v
            if v > self.extremes[i, 1]:
                self.extremes[0][1] = v

    def plot_surfaces(self, ax: Axes3D):
        for s in self.surfaces:
            plot_surface(ax, s, self.show_nor)

    def plot_rays(self, ax: Axes3D):
        for r in self.rays:
            plot_ray(ax, r)

    def organize(self):
        if self.light.birth=="point" and self.light.divergence==0:
            first = self.surfaces[0]
            relative = first.vertex - self.light.position
            vec = normalize(relative)
            self.light.vector=vec
            distance = np.linalg.norm(relative)
            ratio = -np.dot(vec,first.normal)

            area = first.height*first.width+first.semidia**2*np.pi-1
            radi = np.sqrt(area/np.pi)
            divergence = abs(np.arctan(radi*ratio/distance))
            self.light.point(divergence)

        for i, s in enumerate(self.surfaces):
            s.sid = i
            if s.mode == "diffuse":
                if i==len(self.surfaces)-1:
                    raise Exception("The last surface of an optical train cannot be a diffuse surface")
                first = self.surfaces[i + 1]
                relative = xdot(s.move,first.vertex)
                vec = normalize(relative)
                distance = np.linalg.norm(relative)
                ratio = -np.dot(vec,first.normal)
                area = first.height * first.width + first.semidia ** 2 * np.pi - 1
                radi = np.sqrt(area / np.pi)
                divergence = abs(np.arctan(radi * ratio / distance))
                s.fan = divergence
                s.going = relative
    def propagate(self):
        self.organize()
        for r in self.rays:
            for i in range(r.active_since, len(self.surfaces)):
                s = self.surfaces[i]
                if s.mode == "inactive":
                    continue
                if s.shape == "plano":
                    interact_plane(r, s)
                elif s.shape == "spherical":
                    interact_sphere(r, s)
                elif s.shape == "cylindrical":
                    interact_cylinder(r, s)
                if r.active_until >= i and r.active_until != -1:
                    break
            r.intensities = np.asarray(r.intensities)
            self.longest = max(self.longest, r.travel())
        for r in self.rays:
            r.peer_dist = self.longest
    def render(self, ax):
        self.plot_rays(ax)
        if self.show_surfaces is True:
            self.plot_surfaces(ax)
        set_axes_equal(ax)

    def translate(self, shift):
        for s in self.surfaces:
            s.translate(shift)

def parse_yaml_file(file_path):
    with open(file_path, 'r') as yaml_file:
        parsed_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return parsed_data

def load_result_settings(param):
    glo = param['result_settings']
    global angle_to_normal, hit_coordinates
    if 'angle_to_normal' in glo.keys():
        angle_to_normal = glo['angle_to_normal']
    if 'hit_coordinates' in glo.keys():
        hit_coordinates = glo['hit_coordinates']

def load_display(param):
    glo = param['display_settings']
    global lens_display_theta, lens_display_phi, \
        ray_display_density, obj_display_density, \
        fill_with_grey, show_grid, density_for_intensity, \
        show_plot, coexist
    if 'show_plot' in glo.keys():
        show_plot = glo['show_plot']
    if 'coexist' in glo.keys():
        coexist = glo['coexist']
        if coexist is False:
            plt.close('all')
    if 'figsize' in glo.keys():
        plt.rcParams['figure.figsize'] = tuple(glo['figsize'])
    if 'theme' in glo.keys():
        plt.style.use(glo['theme'])
    if 'lens_display_phi' in glo.keys():
        lens_display_phi = int(glo['lens_display_phi'])
    if 'lens_display_theta' in glo.keys():
        lens_display_theta = int(glo['lens_display_theta'])
    if 'ray_display_density' in glo.keys():
        ray_display_density = float(glo['ray_display_density'])
    if 'obj_display_density' in glo.keys():
        obj_display_density = float(glo['obj_display_density'])
    if 'fill_with_grey' in glo.keys():
        fill_with_grey = glo['fill_with_grey']
    if 'show_grid' in glo.keys():
        show_grid = glo['show_grid']
    if 'density_for_intensity' in glo.keys():
        density_for_intensity = glo['density_for_intensity']
        ray_display_density= 1

def list2vec(unknown):
    out = np.zeros(len(unknown))
    for i in range(len(unknown)):
        out[i]=float(unknown[i])
    return out

def load_lights(param):
    raw_lights = param
    lights = []
    for i in range(len(raw_lights)):
        raw_light = raw_lights[i]
        raw_light = list(raw_light.values())[0]
        lid = raw_light['lid']
        posi = list2vec(raw_light['position'])
        vec = normalize(list2vec(raw_light['vector']))
        lights.append(light(posi, normalize(vec),
                            number=int(raw_light['number']),
                            wavelength=int(raw_light['wavelength'])))
        if raw_light['type'] == "ring":
            para = raw_light['param']
            wid = 0
            wh = 1
            dia = 0
            if not isinstance(para,list):
                para = [para]
            wid = float(para[0])
            if len(para) > 1:
                wh = float(para[1])
            if len(para) > 2:
                dia = int(para[2])
            lights[-1].lid = lid
            lights[-1].ring(wid, wh, dia)
        elif raw_light['type'] == "linear":
            para = raw_light['param']
            wid =0
            if not isinstance(para, list):
                para = [para]
            wid = int(para[0])
            dia = 0
            if len(para) == 2:
                dia = int(para[1])
            lights[-1].lid = lid
            lights[-1].linear(wid, dia)
        elif raw_light['type'] == "point":
            para = raw_light['param']
            if para is None:
                lights[-1].lid = lid
                lights[-1].point()
            else:
                di = float(raw_light['param'])
                lights[-1].lid = lid
                lights[-1].point(di)
        elif raw_light['type'] == "uniform":
            ra = float(raw_light['param'])
            lights[-1].lid = lid
            lights[-1].uniform(ra)
    return lights

def check_registered(newly, collec):
    if isinstance(newly, surface):
        for c in collec:
            if newly.normal == c.normal and newly.vertex == c.vertex and newly.radius == c.radius:
                return True
        return False

def build_surface(raw):
    coord = None
    if "coord" in raw.keys():
        coord = np.array(raw['coord'])
    normal = None
    if "normal" in raw.keys():
        if isinstance(raw['normal'],list):
            normal = normalize(list2vec(raw['normal']))
    angles = None
    if "angles" in raw.keys():
        if isinstance(raw['angles'],list):
            angles = raw['angles']
    shape = raw['shape']
    radius = 1
    if "radius" in raw.keys():
        radius = float(raw['radius'])
    semidia = 1.0
    if "semidia" in raw.keys():
        semidia = raw['semidia']
    dial = 0
    if "dial" in raw.keys():
        dial = float(raw['dial'])
    height = 1
    if "height" in raw.keys():
        height = float(raw['height'])
    width = 1
    if "width" in raw.keys():
        width = float(raw['width'])
    mode = "inactive"
    if "mode" in raw.keys():
        mode =raw['mode']
    n1=1
    n2=1
    if "n1" in raw.keys():
        n1 = raw['n1']
    if "n2" in raw.keys():
        n2 = raw['n2']
    transmission = 1.0
    if "transmission" in raw.keys():
        transmission = raw['transmission']
    color=None
    if "color" in raw.keys():
        color = raw['color']
    alpha = obj_display_density
    if "alpha" in raw.keys():
        alpha = float(raw['alpha'])
    return surface(coord,normal,shape,angles,radius,semidia,dial,height,width,mode,n1,n2,transmission,color,alpha)
def load_surfaces(surfaces):
    surs={}
    for i in range(len(surfaces)):
        para = surfaces[i]
        para = list(para.values())[0]
        sur = build_surface(para)
        sid = para['sid']
        surs[sid] = sur
    return surs
def load_assemblies(assemblies):
    asms = {}
    for i in range(len(assemblies)):
        para = assemblies[i]
        aid = para['aid']
        faces = list(para.values())[1:]
        building = assembly()
        for j in range(len(faces)):
            face = faces[j]
            rela = face['relative']
            s = build_surface(face)
            building.add(s, rela,s.normal)
        asms[aid]=building
    return asms
def load_paths(param):
    surfaces = load_surfaces(param['surfaces'])
    assemblies = load_assemblies(param['assemblies'])
    lights = load_lights(param['light_sources'])
    used_sid=[]
    existing_surfaces=[]
    used_aid=[]

    masterplan = param['optical_trains']
    paths=[]
    for i in range(len(masterplan)):
        a = train()
        path = list(masterplan[i].values())
        for element in path:
            if "lid" in element.keys():
                li = element['lid']
                li = lights[li]
                if a.light is None:
                    a.set_light(li)
                else:
                    a.extend_light(li)
            elif "aid" in element.keys() or "sid" in element.keys():
                posi = list2vec(element['position'])
                nor = None
                angles = None
                if "normal" in element.keys():
                    nor = normalize(list2vec(element['normal']))
                if "angles" in element.keys():
                    angles = list2vec(element['angles'])
                if "aid" in element.keys():
                    ind = element['aid']
                    asem = assemblies[ind].copy()
                    if "flip" in element.keys():
                        if element['flip'] is True:
                            asem.invert()
                    asem.place(posi,vector=nor, angles=angles)
                    a.add(asem)
                elif "sid" in element.keys():
                    ind = element['sid']
                    if ind in used_sid:
                        sur = surfaces[ind].copy(afresh=True)
                    else:
                        sur = surfaces[ind]
                        used_sid.append(ind)
                    sur.relocate(posi,normal=nor, angles=angles)
                    a.append(sur)
        paths.append(a)
    return paths

def simple_ray_tracer_main(parameters):
    if 'display_settings' in parameters.keys():
        load_display(parameters)
    trains = load_paths(parameters)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for t in trains:
        t.propagate()
        t.render(ax)
    set_axes_equal(ax)
    plt.tight_layout()
    global via_gui
    if via_gui is True:
        fig.canvas.manager.window.wm_geometry("+%d+%d"%(10,10))
    plt.show()

def smooth_line(raw,noe):
    if isinstance(raw,str):
        for i in range(noe-1):
            raw=raw+","
        return raw
    elif isinstance(raw,list):
        line =""
        for i in range(noe):
            if i<len(raw):
                line=line+raw[i]
            line=line+","
        return line
def simple_ray_tracer_main_w_analysis(parameters):
    folder="SRT_result"
    if folder in os.listdir():
        for item in os.listdir(folder):
            os.remove(folder+"/"+item)
    else:
        os.mkdir(folder)
    if 'result_settings' in parameters.keys():
        load_result_settings(parameters)
    if 'display_settings' in parameters.keys():
        load_display(parameters)
    plt.style.use('default')
    trains = load_paths(parameters)
    plotted_surfaces=[]
    for i,t in enumerate(trains):
        t.propagate()
        for j,s in enumerate(t.surfaces):
            if not s in plotted_surfaces:
                plotted_surfaces.append(s)
    columns=['id','v1','v2','v3','c1','c2','c3','ang2normal','wv','intensity']
    lines=[]
    lines.append(smooth_line(columns,len(columns))+"\n")
    for s in plotted_surfaces:
        section = s.shape+"  "+s.mode+"  "+str(s.sid)
        lines.append(smooth_line(section,len(columns))+"\n")
        base = s.base[:,1:]
        disk = False
        if s.shape == "spherical":
            disk = True
        if s.shape == "plano":
            if isinstance(s.semidia, float) and s.semidia == 1:
                disk = False
            else:
                disk = True
        if disk is True:
            xs=[]
            ys=[]
            for theta in range(0,360,10):
                xs.append(np.cos(np.radians(theta)) * s.semidia)
                ys.append(np.sin(np.radians(theta)) * s.semidia)
            base = np.stack([xs,ys],axis=1)
        rhr = np.radians(s.dial)
        rtm = np.array([[np.cos(rhr),-np.sin(rhr)],[np.sin(rhr),np.cos(rhr)]])
        base = np.dot(base, rtm)
        if show_plot is True:
            fig,ax=plt.subplots()
            ax.fill(base[:, 0], base[:, 1], color='k', alpha=0.1)

        for k,stat in enumerate(s.intersects):
            abs_coord = stat[0]
            rel_coord = xdot(s.move,abs_coord)
            rel_coord[0] +=s.radius
            reduc = rel_coord[1:3]
            reduc = np.dot(reduc,rtm)
            wv = stat[1]
            inten = stat[2]
            rel_vec = normalize(stat[3])
            abs_vec = normalize(xdot(s.inverse,rel_vec)-xdot(s.inverse,np.zeros(3)))
            ang = np.degrees(np.arccos(np.dot(-1*rel_vec,np.array([-1,0,0]))))#,np.dot(rel_vec,np.array([-1,0,0])))))
            c=(1,1,1)
            info = str(k)+","

            abs_vec = abs_vec.astype('str').tolist()
            abs_vec = ','.join(abs_vec)+","
            abs_coor = abs_coord.astype('str').tolist()
            abs_coor = ','.join(abs_coor)+","

            rel_vec = rel_vec.astype('str').tolist()
            rel_vec = ','.join(rel_vec) + ","
            rel_coor = rel_coord.astype('str').tolist()
            rel_coor = ','.join(rel_coor) + ","

            if hit_coordinates =='absolute':
                info = info + abs_vec + abs_coor + str(ang) + "," + str(wv) + "," + str(inten)
            else:
                info = info + rel_vec + rel_coor + str(ang) + "," + str(wv) + "," + str(inten)

            lines.append(info+"\n")
            if isinstance(wv,int):
                if wv!=0:
                    c=wavelength2rgb(wv)
            if show_plot is True:
                ax.scatter(reduc[0],reduc[1],color=c,marker='o',alpha=inten)

        if show_plot is True:
            ax.set_title("surface"+str(s.sid)+","+s.shape)
            ylocs, lbls = plt.yticks()
            interval = ylocs[1] - ylocs[0]
            ax.set_yticks(ylocs)
            lims = ax.get_xlim()
            lefty = -np.arange(0,-lims[0],interval)
            righty = np.arange(0,lims[1],interval)
            xtic = np.concatenate([lefty,righty])
            ax.set_xticks(xtic)
            ax.xaxis.set_ticklabels([])
            ax.grid(True)
            plt.savefig(folder+"/s"+str(s.sid)+".png")
            plt.close(fig)
        with open(folder+"/"+"ray_sur_interactions.csv",'w') as writer:
            writer.writelines(lines)
    messagebox.showinfo("Info","Result is stored in SRT_result folder")
    plt.style.use('dark_background')

    if show_plot is False:
        return
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for t in trains:
        t.render(ax)
    set_axes_equal(ax)
    plt.tight_layout()
    global via_gui
    if via_gui is True:
        fig.canvas.manager.window.wm_geometry("+%d+%d"%(10,10))
    plt.show()

def ota():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    red = light(np.array([0, 0, -3]), normalize(np.array([1, 0, 0.1])), number=5,wavelength = 633)
    red.linear(10)
    green = light(np.array([0, 0, -1.5]), normalize(np.array([1, 0, 0.05])), number=5, wavelength=532)
    green.linear(10)
    blue = light(np.array([0, 0, 0]), normalize(np.array([1, 0, 0])), number=5, wavelength=488)
    blue.linear(10)

    f1 = surface(coord=None, normal=None, shape="spherical",
                 radius=23.71, semidia=10, mode="refraction",n2=1.691)
    f2 = surface(coord=None, normal=None, shape="spherical",
                 radius=7331, semidia=10, mode="refraction", n1=1.691)
    f3 = surface(coord=None, normal=None, shape="spherical",
                 radius=-24.45, semidia=6, mode="refraction", n2=1.673)
    f4 = surface(coord=None, normal=None, shape="spherical",
                 radius=21.896, semidia=6, mode="refraction", n1=1.673)
    f5 = surface(coord=None, normal=None, shape="spherical",
                 radius=86.759, semidia=8, mode="refraction", n2=1.691)
    f6 = surface(coord=None, normal=None, shape="spherical",
                 radius=-20.494, semidia=8, mode="refraction", n1=1.691)
    e1=assembly()
    e1.add(f1,0)
    e1.add(f2,4.831)

    e2 = assembly()
    e2.add(f3, 0)
    e2.add(f4, 0.975)

    e3 = assembly()
    e3.add(f5, 0)
    e3.add(f6, 3.127)

    e1.extend(e2,5.86)
    e1.extend(e3,4.822)
    e1.place(np.array([5,0,0]),np.array([-1,0,0]))

    focus=surface(coord=None, normal=None,shape="plano",height=20,width=20, mode="refraction")
    termino = surface(coord=e1.last_surface + np.array([61.24, 0, 0]), normal=None, angles=[0, 0],
                      shape="plano", height=20, width=20, mode="refraction")
    focus.relocate(e1.last_surface+np.array([41.24,0,0]))

    sb = train()
    sb.add(e1)
    sb.append(focus)
    sb.append(termino)

    sb.set_light(red)
    sb.propagate()
    sb.render(ax)

    sb.set_light(green)
    sb.propagate()
    sb.render(ax)

    sb.set_light(blue)
    sb.propagate()
    sb.render(ax)

    set_axes_equal(ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    global via_gui
    if via_gui is True:
      #cv.imwrite("rendering.jpg",cv.cvtColor(data,cv.COLOR_BGR2RGB))
      fig.canvas.manager.window.wm_geometry("+%d+%d"%(10,10))
    plt.show()

#GUI
import tkinter as tk
from tkinter import filedialog,messagebox

def run_gui():
    root = tk.Tk()
    root.title("SRT")

    greeting_label = tk.Label(root, text="Open a ray tracing script")
    greeting_label.pack()

    file_path_frame = tk.Frame(root)
    file_path_entry = tk.Entry(file_path_frame)
    file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    file_path_entry.insert(0, "default.yml")

    def browse_files():
        file_path = filedialog.askopenfilename(initialdir="")
        if file_path:
            file_path_entry.delete(0, tk.END)
            file_path_entry.insert(0, file_path)

    browse_button = tk.Button(file_path_frame, text="...", command=browse_files)
    browse_button.pack(side=tk.RIGHT)
    file_path_frame.pack(fill=tk.X, padx=10, pady=5)

    parameters = {"path": None, "analysis":False,"quit":False}

    def on_run():
        parameters["path"] = file_path_entry.get()
        parameters["analysis"] = False
        try:
            instruction=parse_yaml_file(parameters['path'])
            simple_ray_tracer_main(instruction)
        except Exception as e:
            messagebox.showerror("error", str(e))
            return None
        root.quit()

    def on_run2():
        parameters["path"] = file_path_entry.get()
        parameters["analysis"] = True
        try:
            instruction = parse_yaml_file(parameters['path'])
            simple_ray_tracer_main_w_analysis(instruction)
        except Exception as e:
            messagebox.showerror("error", str(e))
            return None
        root.quit()

    def on_exit():
        parameters["path"] = ""
        parameters["analysis"] = False
        parameters["quit"] = "exit"
        root.quit()

    run_button = tk.Button(root, text="Run", command=on_run)
    run_button.pack(side=tk.LEFT, padx=10, pady=10)
    run_button2 = tk.Button(root, text="Run with analysis", command=on_run2)
    run_button2.pack(side=tk.LEFT, padx=10, pady=10)
    exit_button = tk.Button(root, text="Exit", command=on_exit)
    exit_button.pack(side=tk.LEFT, padx=10, pady=10)
    root.protocol("WM_DELETE_WINDOW",on_exit)
    w = root.winfo_reqwidth()
    h = root.winfo_reqheight()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    root.geometry('+%d+%d' % (x, y))
    root.mainloop()
    try:
        root.destroy()
    except:
        pass
    return parameters

via_gui = False

try:
    import google.colab.files as files
except:
    via_gui = True

#via_gui = False

if via_gui is True:
    while True:
        response = run_gui()
        if response["quit"]== "exit":
            break
else:
    ota()
