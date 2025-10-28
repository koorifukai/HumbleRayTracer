import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7)
plt.style.use('dark_background')
plt.rcParams["font.size"] = 15
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    matplotlib.rcParams['axes3d.mouserotationstyle'] = 'azel'
except:
    pass
import cv2 as cv
from PIL import Image
import os
import copy
import yaml

import tkinter as tk
from tkinter import filedialog,messagebox
from collections import Counter
import datetime

modes = ['inactive', 'refraction', 'reflection', 'partial', 'absorption',
         'diffuse', 'aperture']

lens_display_theta=10    #theta= longitudinal resolution
lens_display_phi=20      #phi= circular resolution
ray_display_density=0.5
obj_display_density=0.5
overshoot_travelled=0.1
overshoot_solid = True
show_grid = True
fill_with_grey = False
density_for_intensity=True
show_plot = True
plot_last_only = False
coexist = True
top_most = False

azimuth = -60
elevation = 30
x_limits = (0,1)
y_limits = (0,1)
z_limits = (0,1)

#    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

angle_to_normal = 'element' #choose between element and hit
hit_coordinates = 'absolute' #choose between absolute and relative

num_of_iterations = 0
optimization_mode = "aberrations"
optimization_param = None

result_folder="SRT_result"
already_shown=[False,False]
last_path = "default.yml"

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

    # Promote to 3D: [x, y] -> [x, y, 0]
    vectors3D = np.hstack([vectors, np.zeros((4, 1))])
    rolled3D = np.roll(vectors3D, -1, axis=0)
    cross = np.cross(vectors3D, rolled3D)

    # Use only the Z-component of the 3D cross product
    z_cross = cross[:, 2]

    if np.all(z_cross >= 0) or np.all(z_cross <= 0):
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
    def __init__(self, container, position=None, vector=None, wavelength=0, lid=-1):
        self.mutable_params = {
            'position': position.copy(),
            'vector': vector.copy(),
        }
        self.coord = np.zeros(3, np.float64)
        self.container = container
        self.vec = np.array([1, 0, 0], np.float64)
        self.trace = []
        self.active = True
        self.active_since = 0
        self.active_until = -1
        self.intensity = 1
        self.intensities=[[0,1]]
        self.lid = lid
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

    def reset(self):
        orig = self.mutable_params
        self.coord = np.zeros(3, np.float64)
        self.vec = np.array([1, 0, 0], np.float64)
        self.trace = []
        self.active = True
        self.active_since = 0
        self.active_until = -1
        self.intensity = 1
        self.intensities = [[0, 1]]
        if orig['position'] is not None and orig['vector'] is not None:
            self.coord = orig['position']
            self.vec[0:3] = orig['vector'][0:3]
            self.trace.append(orig['position'])
        self.peer_dist = 0
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
                 angles=None, radius=1, semidia=1.0, efl=0,
                 dial=0, height=1, width=1,
                 mode="inactive", n1=1, n2=1, transmission=1,
                 select=None,
                 color=None, alpha=obj_display_density):
        self.mutable_params = {}
        if coord is None:
            coord = np.array([0,0,0],np.float64)
        else:
            self.mutable_params['coord']=coord.copy()
        self.vertex = np.float64(coord)
        self.normal = np.array([-1, 0, 0], np.float64)
        self.dial = float(dial)
        self.intersects=[]
        if not normal is None:
            self.normal = normalize(normal)
            self.mutable_params['normal']=self.normal.copy()
        elif normal is None and not angles is None:
            self.mutable_params['angles']=angles.copy()
            self.mutable_params['normal']=None
            nor = ang2vec(angles)
            self.normal = normalize(nor)

        self.shape = shape
        self.cylindrical = False
        self.disk = False

        self.radius = float(radius)  # +( -)
        self.mutable_params['radius']=radius
        self.semidia = semidia
        self.height=float(height)
        self.width=float(width)
        self.base = np.array([[0, -semidia, -semidia],
                              [0, -semidia, semidia],
                              [0, semidia, semidia],
                              [0, semidia, -semidia]],np.float64)

        self.efl = float(efl)
        self.n2 = float(n2)
        self.n1 = float(n1)
        self.mode = mode
        self.transmission = float(transmission)
        self.select = select
        self.sel_cond = []
        if not select is None:
            if "-" in select:
                sel = select.split("-")
            else:
                sel = [select]
            for s in sel:
                dire = s[0].upper()
                which = int(s[1:])
                self.sel_cond.append([dire,which])
        if shape == "spherical":
            self.dial = 0
        elif shape == "cylindrical":
            self.height = height
            self.width = width
            self.base = np.array([[0, -width / 2, -height / 2],
                                  [0, -width / 2, height / 2],
                                  [0, width / 2, height / 2],
                                  [0, width / 2, -height / 2]],np.float64)
            self.cylindrical = True
            if abs(self.radius) < abs(0.5*self.width):
                self.radius = 1
                self.width = 1
        elif shape == "plano":
            self.radius = 0
            if isinstance(semidia,float) and semidia==1: #which means default param is not used, surface is a disk
                self.disk = False
                self.height = height
                self.width = width
                self.base = np.array([[0, -width / 2, -height / 2],
                                      [0, -width / 2, height / 2],
                                      [0, width / 2, height / 2],
                                      [0, width / 2, -height / 2]],np.float64)
            else:
                self.disk = True
                self.semidia = semidia
                self.base = np.array([[0, -semidia, -semidia],
                                      [0, -semidia, semidia],
                                      [0, semidia, semidia],
                                      [0, semidia, -semidia]],np.float64)
        elif shape == "perfect":
            self.n2 = 1.5 # roughly BK7, used only for visualizing surface
            self.semidia = semidia
            self.base = np.array([[0, -semidia, -semidia],
                                  [0, -semidia, semidia],
                                  [0, semidia, semidia],
                                  [0, semidia, -semidia]],np.float64)
            if mode == "refraction":
                # model with one flat surface
                # viz with a thin lens
                v = self.efl
                vn = 1/v/(self.n2-1)
                self.radius = 2/vn
            elif mode == "reflection":
                self.radius = self.efl * -2

        self.move = np.eye(4)  # transform incoming ray into surface definition coords
        self.inverse = np.eye(4)  # transform coords from surface definition coords to real world for viz, hence inverse

        self.calc_mat()

        self.color = (1,1,1)
        if isinstance(color,int):
            self.color=wavelength2rgb(color)
        self.alpha = float(alpha)

        if not color is None:
            self.color = color
            self.alpha = alpha

        self.sid = 0
        self.rendered=False

        self.going=None
        self.fan=0
        self.light_edge = None#(np.random.rand(),np.random.rand(),np.random.rand())

    def copy(self,afresh = False):
        neu = surface(self.vertex.copy(),self.normal.copy(),self.shape)
        neu.mutable_params = self.mutable_params.copy()
        neu.dial = self.dial
        neu.mode = self.mode
        neu.n1 = self.n1
        neu.n2 = self.n2
        neu.cylindrical = self.cylindrical
        neu.disk = self.disk
        neu.radius = self.radius
        neu.semidia = self.semidia
        neu.efl = self.efl
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
        neu.going = self.going
        neu.fan = self.fan
        neu.select = self.select
        neu.sel_cond = self.sel_cond
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

    def calc_mat(self, upright=True, external_rot=None):
        if upright:
            r = upright_rot_transform(np.array([-1, 0, 0]), self.normal)
        elif external_rot is not None:
            r = external_rot[0:3,0:3]
        else:
            r = rot_transform(np.array([-1, 0, 0]), self.normal)

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
        self.position=np.zeros(3,np.float64)
        self.normal=np.array([-1,0,0],np.float64)
        self.normals=[]
        self.positions=[]

        self.last_surface= np.zeros(3,np.float64)

    def add(self, s:surface, relative, normal=None, angles=None, dial=0):
        if not normal is None:
            nor = normal
        elif normal is None and not angles is None:
            nor = ang2vec(angles)
        else:
            nor = np.array([-1,0,0],np.float64)
        if isinstance(relative,float) or isinstance(relative,int):
            relative = np.array([relative,0,0],np.float64)
        vertex = self.last_surface + relative
        s.vertex=vertex
        s.normal=nor
        self.normals.append(nor)
        self.positions.append(vertex)
        self.last_surface=vertex

        self.surfaces.append(s)

    def place(self, position, normal=np.array([-1,0,0]), angles=None, dial=None):
        if not angles is None:
            normal = ang2vec(angles)
        normal = normalize(normal)
        self.position = position

        R_align = upright_rot_transform(np.array([-1, 0, 0]), normal)
        if dial is not None:
            R_dial = axial_rotation(normal, np.radians(dial))
            R_total = np.dot(R_dial, R_align)
        else:
            R_total = R_align

        # Create 4x4 homogeneous transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R_total
        T[0:3, 3] = position

        for i, s in enumerate(self.surfaces):
            rel_pos = self.positions[i]
            rel_norm = self.normals[i]
            new_pos = xdot(T, rel_pos)
            new_norm = normalize(np.dot(R_total, rel_norm))

            s.vertex = new_pos
            s.normal = new_norm

            local_mat = np.eye(4)
            local_mat[0:3, 0:3] = upright_rot_transform(np.array([-1, 0, 0]), rel_norm)
            local_mat[0:3, 3] = rel_pos

            global_mat = np.dot(T,local_mat)
            #2 4x4 homogeneous matrices in succession
            s.calc_mat(upright=False, external_rot=global_mat)

        self.last_surface = self.positions[-1]

    def extend(self, additional: 'assembly', relative, where="tail"): #only for axial objects
        if np.dot(self.normal, additional.normal) < 1:
            return
        if isinstance(relative, float) or isinstance(relative, int):
            relative = np.array([relative, 0, 0], np.float64)
        for i, s in enumerate(additional.surfaces):
            additional.positions[i]=additional.positions[i]+relative+self.last_surface
            self.surfaces.append(s)
            self.normals.append(s.normal)
            self.positions.append(additional.positions[i])
        if where=="tail":
            self.last_surface =additional.positions[-1]

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
        stat = [r_hit,v_rel_from]
        ray_sta = extract_ray_info(r)
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
        stat = [r_hit,v_rel_from]
        ray_sta = extract_ray_info(r)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        return
    if s.mode == "partial" and forced is None:
        r_hit = xdot(s.inverse, h)
        shadow = ray(None, r_hit - 1e-6 * r.vec, r.vec.copy(), r.wv,lid=r.lid+0.1)
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
        stat = [r_hit,v_rel_from]
        ray_sta = extract_ray_info(r)
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
        stat = [r_hit,v_rel_from]
        ray_sta = extract_ray_info(r)
        stat.extend(ray_sta)
        s.intersects.append(stat)
        return
    if s.mode == "aperture":
        interact_vhnrs(v, h, n, r, s, forced="refraction")
        return
def extract_ray_info(incident:ray):
    return [incident.wv,incident.intensity,incident.lid]
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
        print(msg)
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
        print(msg)
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
        print(msg)
        return
    n = np.array([-1, 0, 0],np.float64)
    last = incident.trace[-1]
    towards = last + incident.vec
    t_last = xdot(sur.move, last)
    t_towards = xdot(sur.move, towards)
    v = normalize(t_towards - t_last)
    if v[0] == 0 or t_last[0]>=0:
        return
    hit = t_last - t_last[0] / v[0] * v
    f = hit[1:3]
    outsider = False
    if sur.disk is False:
        validate = simple_convex_hull(sur.base[:, 1:3], f)
        if validate == 0:
            outsider = True
    else:
        if np.linalg.norm(f)>sur.semidia:
            outsider = True
    if outsider is True:
        if sur.mode == "aperture":
            interact_vhnrs(v, hit, n, incident, sur)
            rec = xdot(sur.inverse,hit)
            incident.trace.append(rec)
            incident.active_until = sur.sid
            incident.active = False
        return
    interact_vhnrs(v, hit, n, incident, sur)

def interact_perfect(incident: ray, sur: surface):
    if np.dot(incident.vec,sur.normal)>=0:
        msg=",".join(["backstabbing","sid:"+str(sur.sid),"shape:"+sur.shape])
        print(msg)
        return
    last = incident.trace[-1]
    towards = last + incident.vec
    t_last = xdot(sur.move, last)
    t_towards = xdot(sur.move, towards)
    v = normalize(t_towards - t_last)
    if v[0] == 0 or t_last[0]>=0:
        return
    hit = t_last - t_last[0] / v[0] * v
    f = hit[1:3]
    if np.linalg.norm(f)>sur.semidia:
        return
    #interact_vhnrs(v, hit, n, incident, sur)

    oh = f
    a1 = np.arccos(np.dot(np.array([0, 0, 1]), np.array([0, oh[0], oh[1]]), np.float64))
    a2 = np.pi/2-a1
    m1 = axial_rotation(np.array([1, 0, 0]), a1)
    m2 = axial_rotation(np.array([1, 0, 0]), a2)


    dy = v[1] / v[0]
    if dy == 0:
        vy = 0
    else:
        oy = hit[1] / dy
        iy = 1 / (1 / sur.efl - 1 / oy)
        vy = hit[1]/iy

    dz = v[2] / v[0]
    if dz ==0:
        vz = 0
    else:
        oz = hit[2] / dz
        iz = 1 / (1 / sur.efl - 1 / oz)
        vz = hit[2]/iz
    v_out = normalize(np.array([1,vy,vz]))

    r_hit = xdot(sur.inverse,hit)
    r_towards = xdot(sur.inverse,v_out)
    incident.vec = r_towards
    incident.trace.append(r_hit)
    stat = [r_hit, v]
    ray_sta = extract_ray_info(incident)
    stat.extend(ray_sta)
    sur.intersects.append(stat)

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
def highlight_surface_edge(coords):
    # Dictionary to store edge counts
    edge_counts = Counter()
    # Process each facet to extract edges
    for facet in coords:
        # Extract all edges from the facet
        edges = [
            (facet[0], facet[1]),
            (facet[1], facet[2]),
            (facet[2], facet[3]),
            (facet[3], facet[0]),
        ]
        # Normalize each edge to avoid duplicates (e.g., (v1, v2) and (v2, v1))
        for edge in edges:
            # Sort vertices in the edge to have a consistent order (v1, v2) where v1 < v2
            normalized_edge = tuple(sorted(edge))
            # Update the counter
            edge_counts[normalized_edge] += 1

    # Find edges that have only been visited once
    single_visit_edges = [edge for edge, count in edge_counts.items() if count == 1]
    return single_visit_edges

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
    if sur.mode == "aperture":
        if sur.radius == 0 and sur.disk is False:
            coords[0].append(coords[0][0])
        coords = np.asarray(coords)[0]
        ax.plot(coords[:,0],coords[:,1],coords[:,2],color='k')
    else:
        ax.add_collection3d(Poly3DCollection(coords, facecolor=sur.color, alpha=sur.alpha))
    if normal is True:
        ver = sur.vertex
        nor = sur.normal * 0.1 * max(sur.semidia,(max(sur.height,sur.width)))
        x = [ver[0], ver[0] + nor[0]]
        y = [ver[1], ver[1] + nor[1]]
        z = [ver[2], ver[2] + nor[2]]
        ax.plot(x, y, z, color=(1,1,1), alpha=obj_display_density)
    if not sur.light_edge is None:
        edges = highlight_surface_edge(coords)
        vertices = [item for sublist in coords for item in sublist]
        for edge in edges:
            sta = edge[0]
            end = edge[1]
            df = np.array(vertices)-sta
            df = np.sum(np.abs(df),axis=1)
            proximi = np.array(np.where(df<1e-4))[0]
            if len(proximi) ==4:
                continue
            df = np.array(vertices)-end
            df = np.sum(np.abs(df),axis=1)
            proximi = np.array(np.where(df<1e-4))[0]
            if len(proximi) ==4:
                continue
            xs = [sta[0], end[0]]
            ys = [sta[1], end[1]]
            zs = [sta[2], end[2]]
            ax.plot(xs, ys, zs, color=sur.light_edge)
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
            global overshoot_travelled,overshoot_solid
            linesty = "solid"
            if overshoot_solid is False:
                linesty = "dotted"
            travelled= incident.peer_dist
            xo = [xs[-1],xs[-1]+incident.vec[0] * travelled * overshoot_travelled]
            yo = [ys[-1],ys[-1]+incident.vec[1] * travelled * overshoot_travelled]
            zo = [zs[-1],zs[-1]+incident.vec[2] * travelled * overshoot_travelled]
            ax.plot(xo, yo, zo, linestyle=linesty, color=incident.color, alpha=ray_display_density * density[-1][1])

def set_axes_equal(ax):
    # Extract 3D limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Compute centers and plot radius
    x_middle = 0.5 * sum(x_limits)
    y_middle = 0.5 * sum(y_limits)
    z_middle = 0.5 * sum(z_limits)
    plot_radius = 0.5 * max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0])
    )

    # Set symmetric limits around center
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Enforce box aspect ratio (requires matplotlib ≥ 3.3)
    if hasattr(ax, 'set_box_aspect'):
        ax.set_box_aspect([1, 1, 1])

    # Optional appearance tweaks
    if not show_grid:
        ax.set_axis_off()
        return
    elif not fill_with_grey:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def save_figure(fig):
    fig.canvas.draw()
    #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = np.array(fig.canvas.buffer_rgba())
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, 0:3]
    #cv.imwrite("rendering.jpg", cv.cvtColor(data, cv.COLOR_BGR2RGB))

def on_key(event):
    if event.key == 'enter':
        plt.close(event.canvas.figure)
    if event.key == 'escape':
        plt.close(event.canvas.figure)
class light:
    def __init__(self, position, vector, angles=None, number=6,wavelength=0):
        self.lid=-1
        self.rays = []
        self.position=position
        if angles is None:
            self.vector=normalize(vector)
        else:
            azi = np.radians(angles[0])
            ele = np.radians(angles[1])
            self.vector = np.array([np.cos(azi)*np.cos(ele),np.sin(azi)*np.cos(ele),np.sin(ele)])
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
        core = np.zeros((self.number, 3),np.float64)
        core[:, 0] = np.linspace(-width/2,width/2,self.number)
        core[:, 1] = core[:,0]*np.cos(rad+np.pi/2)
        core[:, 2] = core[:,0]*np.sin(rad+np.pi/2)
        core[:, 0] = 0
        transformed = self.rotate(self.vector,core) + self.position
        for i in range(len(transformed)):
            self.rays.append(ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength,lid=self.lid))
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
        core = np.zeros((self.number, 3),np.float64)
        core[:, 0] = np.linspace(0, 2 * np.pi, self.number+1)[0:-1]
        core[:, 1] = np.cos(core[:, 0]) * r * w
        core[:, 2] = np.sin(core[:, 0]) * r * h
        core[:, 0] = 0
        core=np.dot(core,rhr)
        transformed = self.rotate(self.vector,core) + self.position
        for i in range(len(transformed)):
            self.rays.append(ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength,lid=self.lid))
    """
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
        core = np.zeros((len(x), 3),np.float64)
        core[:, 0] = 0
        core[:, 1] = y
        core[:, 2] = z
        transformed = self.rotate(self.vector, core) + self.position
        for i in range(len(transformed)):
            self.rays.append(
                ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength,lid=self.lid))
    """
    def uniform(self,r):
        self.birth="uniform"
        # distribute n points into hexagonal grid
        pts = [[0,0]]
        layer = 1
        nol = int(round(np.sqrt(2*self.number/6)))
        for l in range(1,nol+1):
            ra = l/nol*r
            nump = 6*layer
            for i in range(nump):
                ang = (i/nump)*2*np.pi
                x = ra * np.cos(ang)
                y = ra * np.sin(ang)
                pts.append([x,y])
            layer+=1
        plane = np.asarray(pts)
        core = np.zeros((len(pts), 3),np.float64)
        core[:, 0] = 0
        core[:, 1] = np.asarray(plane[:, 0])
        core[:, 2] = np.asarray(plane[:, 1])
        transformed = self.rotate(self.vector, core) + self.position
        for i in range(len(transformed)):
            self.rays.append(
                ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength,lid=self.lid))
    def gaussian(self,half_e_square):
        self.birth="gaussian"
        # randomly distribute N points

        sigma = half_e_square/2/np.sqrt(2)
        xs = np.random.normal(0, sigma, self.number)
        ys = np.random.normal(0, sigma, self.number)

        core = np.zeros((len(xs), 3),np.float64)
        core[:, 0] = 0
        core[:, 1] = np.asarray(xs)
        core[:, 2] = np.asarray(ys)
        transformed = self.rotate(self.vector, core) + self.position
        for i in range(len(transformed)):
            self.rays.append(
                ray(container=self.rays, position=transformed[i, :], vector=self.vector, wavelength=self.wavelength,lid=self.lid))

    def point(self,divergence=0):
        self.birth = "point"
        self.divergence = divergence
        if divergence>0:
            core = point_light(self.number,divergence)
            r = rot_transform(np.array([1,0,0]),self.vector)
            transformed = np.dot(r,core.transpose()).transpose()
            for i in range(len(transformed)):
                self.rays.append(ray(container=self.rays, position=self.position, vector=transformed[i,:], wavelength=self.wavelength,lid=self.lid))

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
        self.extremes = np.zeros((3, 2),np.float64)
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
            ratio = np.dot(vec,-1 * first.normal)
            area = first.height * first.width + first.semidia ** 2 * np.pi - 1
            radi = np.sqrt(area / np.pi)
            divergence = 0.9*abs(np.arctan(radi * ratio / distance))
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
                #shoot towards next surface
                ratio = np.dot(vec,np.array([1,0,0]))
                area = first.height * first.width + first.semidia ** 2 * np.pi - 1
                radi = np.sqrt(area / np.pi)
                divergence = 0.9*abs(np.arctan(radi * ratio / distance))
                #shrink cone angle to make sure all rays intersect
                s.fan = divergence
                s.going = relative
    def propagate(self):
        self.organize()
        for r in self.rays:
            for i in range(r.active_since, len(self.surfaces)):
                s = self.surfaces[i]
                if s.mode == "inactive":
                    continue
                effective = True
                if not s.select is None:
                    for con in s.sel_cond:
                        which_way,id = con
                        if which_way == 'O':
                            if id == r.lid or id == r.wv:
                                effective = True
                                break
                            else:
                                effective = False
                        elif which_way == 'X':
                            if id == r.lid or id == r.wv:
                                effective = False
                                break
                if effective is False:
                    continue
                if s.shape == "plano":
                    interact_plane(r, s)
                elif s.shape == "spherical":
                    interact_sphere(r, s)
                elif s.shape == "cylindrical":
                    interact_cylinder(r, s)
                elif s.shape.startswith("perfect"):
                    interact_perfect(r, s)
                if r.active_until >= i and r.active_until != -1:
                    break
            r.intensities = np.asarray(r.intensities)
            self.longest = max(self.longest, r.travel())
        for r in self.rays:
            r.peer_dist = self.longest
    def clear_results(self):
        for r in self.rays:
            r.reset()
        for s in self.surfaces:
            s.intersects=[]
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
def load_optimization_settings(param):
    glo = param['optimization_settings']
    global optimization_mode,optimization_param,num_of_iterations
    if 'mode' in glo.keys():
        optimization_mode = glo['mode']
    if "param" in glo.keys():
        optimization_param = glo['param']
    if "iterations" in glo.keys():
        num_of_iterations = glo['iterations']
def load_display(param):
    glo = param['display_settings']
    global lens_display_theta, lens_display_phi, \
        ray_display_density, obj_display_density, \
        fill_with_grey, show_grid, density_for_intensity, \
        show_plot, coexist, top_most, overshoot_travelled,\
        overshoot_solid, plot_last_only

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
    if 'overshoot_travelled' in glo.keys():
        overshoot_travelled = float(glo['overshoot_travelled'])
    if 'overshoot_solid' in glo.keys():
        overshoot_solid = bool(glo['overshoot_solid'])
    if 'fill_with_grey' in glo.keys():
        fill_with_grey = bool(glo['fill_with_grey'])
    if 'show_grid' in glo.keys():
        show_grid = bool(glo['show_grid'])
    if 'density_for_intensity' in glo.keys():
        density_for_intensity = bool(glo['density_for_intensity'])
        ray_display_density= 1
    if 'top_most' in glo.keys():
        top_most = bool(glo['top_most'])
    if 'plot_last_only' in glo.keys():
        plot_last_only = glo['plot_last_only']


def list2vec(unknown):
    out = np.zeros(len(unknown))
    for i in range(len(unknown)):
        out[i]=float(unknown[i])
    return out
def load_lights(param):
    raw_lights = param
    lights = []

    if len(param) == 1 and isinstance(param[0], dict):
        if len(param[0].items())>1:
            raw_lights = []
            for key,val in param[0].items():
                raw_lights.append({key:val})
        else:
            raw_lights = param

    for i in range(len(raw_lights)):
        raw_light = raw_lights[i]
        raw_light = list(raw_light.values())[0]
        lid = raw_light['lid']
        posi = list2vec(raw_light['position'])
        if not "angles" in raw_light.keys():
            vec = normalize(list2vec(raw_light['vector']))
            lights.append(light(posi, normalize(vec),
                                number=int(raw_light['number']),
                                wavelength=int(raw_light['wavelength'])))
        else:
            vec = None
            angs = raw_light['angles']
            lights.append(light(posi, None,list2vec(angs),
                                number=int(raw_light['number']),
                                wavelength=int(raw_light['wavelength'])))
        if raw_light['type'] == "ring":
            para = raw_light['param']
            radius = 1
            wh = 1
            deg_cw = 0
            if not isinstance(para,list):
                para = [para]
            radius = float(para[0])
            if len(para) > 1:
                wh = float(para[1])
            if len(para) > 2:
                deg_cw = float(para[2])
            lights[-1].lid = lid
            lights[-1].ring(radius, wh, deg_cw)
        elif raw_light['type'] == "linear":
            para = raw_light['param']
            if not isinstance(para, list):
                para = [para]
            wid = float(para[0])
            deg_cw = 0
            if len(para) == 2:
                deg_cw = float(para[1])
            lights[-1].lid = lid
            lights[-1].linear(wid, deg_cw)
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
        elif raw_light['type'] == "gaussian":
            half_e_square = float(raw_light['param'])
            lights[-1].lid = lid
            lights[-1].gaussian(half_e_square)
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
    radius = 0
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
    efl = 0
    if "efl" in raw.keys():
        efl = float(raw['efl'])
    n1=1
    n2=1
    if "n1" in raw.keys():
        n1 = float(raw['n1'])
    if "n2" in raw.keys():
        n2 = float(raw['n2'])
    transmission = 1.0
    if "transmission" in raw.keys():
        transmission = float(raw['transmission'])
    select=None
    if "select" in raw.keys():
        select = str(raw['select'])
    color=None
    if "color" in raw.keys():
        color = raw['color']
    alpha = obj_display_density
    if "alpha" in raw.keys():
        alpha = float(raw['alpha'])
    return surface(coord,normal,shape,angles,radius,semidia,efl,dial,height,width,mode,n1,n2,transmission,select,color,alpha)
def load_surfaces(surfaces):
    surs={}
    if len(surfaces) == 1 and isinstance(surfaces[0], dict):
        if len(surfaces[0].items())>1:
            new_sur=[]
            for key,val in surfaces[0].items():
                new_sur.append({key:val})
            surfaces = new_sur
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
                    dial = None
                    if "flip" in element.keys():
                        if element['flip'] is True:
                            asem.invert()
                    if "dial" in element.keys():
                        dial = float(element['dial'])
                    asem.place(posi, normal=nor, angles=angles, dial=dial)
                    a.add(asem)
                elif "sid" in element.keys():
                    ind = element['sid']
                    if ind in used_sid:
                        sur = surfaces[ind].copy(afresh=True)
                    else:
                        sur = surfaces[ind]
                        used_sid.append(ind)
                    if "dial" in element.keys():
                        sur.dial = float(element['dial'])
                    sur.relocate(posi,normal=nor, angles=angles)
                    a.append(sur)
        paths.append(a)
    return paths

def img2gif(path):
    files=os.listdir(path)
    files.sort(key= lambda f: os.path.getmtime(os.path.join(path,f)))
    images=[]
    for file in files:
        dia=cv.imread(path+"/"+file)
        dia=cv.cvtColor(dia,cv.COLOR_BGR2RGB)
        images.append(Image.fromarray(dia).convert('RGB'))
    images[0].save(path+"/"+'animation.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=400, loop=0)
def simple_ray_tracer_main_with_seq(parameters):
    if not result_folder in os.listdir():
        os.mkdir(result_folder)
    sub_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    while sub_folder in os.listdir(result_folder):
        sub_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    os.mkdir(result_folder+"/"+sub_folder)
    if 'display_settings' in parameters.keys():
        load_display(parameters)
    if 'optimization_settings' in parameters.keys():
        if replace_tag_in_dict(parameters, "V1", "V1") or \
                replace_tag_in_dict(parameters, "V2", "V2") or \
                replace_tag_in_dict(parameters, "V3", "V3") or \
                replace_tag_in_dict(parameters, "V4", "V4"):
            messagebox.showinfo("Warning", "Optimization variables must not be in the optical train in this mode")

    step = int(parameters['sequence_settings']['step'])
    seqs = {}
    mids = {}
    for k,v in parameters['sequence_settings'].items():
        if isinstance(v,list):
            if len(v) == 2:
                row = np.linspace(v[0],v[1],step)
                mids[k] = np.median(row)
                seqs[k] = row
    param = copy.deepcopy(parameters)
    for k,v in mids.items():
        replace_tag_in_dict(param, k, v)
    trains = load_paths(param)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    def on_click(event):
        azi, ele = ax.azim, ax.elev
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()
        global azimuth, elevation, x_limits, y_limits, z_limits
        azimuth = azi
        elevation = ele
        x_limits = x_lim
        y_limits = y_lim
        z_limits = z_lim

    for t in trains:
        t.propagate()
        t.render(ax)
    set_axes_equal(ax)
    cid = fig.canvas.mpl_connect('button_release_event', on_click)
    plt.tight_layout()
    fig.canvas.manager.window.wm_geometry("+%d+%d" % (10, 10))
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    for frame in range(step):
        param = copy.deepcopy(parameters)
        for k, v in seqs.items():
            replace_tag_in_dict(param, k, v[frame])
        trains = load_paths(param)
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        for t in trains:
            t.propagate()
            t.render(ax)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlim3d = x_limits
        ax.set_ylim3d = y_limits
        ax.set_zlim3d = z_limits
        plt.savefig(result_folder +"/"+ sub_folder + "/frame_var_" + str(frame) + ".png")
        plt.close(fig)
    img2gif(result_folder+"/"+sub_folder)
    messagebox.showinfo("Complete",str(step)+" frames rendered")
def simple_ray_tracer_main(parameters):
    if 'display_settings' in parameters.keys():
        load_display(parameters)
    if 'sequence_settings' in parameters.keys():
        simple_ray_tracer_main_with_seq(parameters)
        return
    if 'optimization_settings' in parameters.keys():
        if replace_tag_in_dict(parameters,"V1","V1") or \
                replace_tag_in_dict(parameters,"V2","V2") or \
                replace_tag_in_dict(parameters,"V3","V3") or \
                replace_tag_in_dict(parameters,"V4","V4"):
            messagebox.showinfo("Warning","Optimization variables must not be in the optical train in this mode")
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
        fig.canvas.mpl_connect('key_press_event', on_key)
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
def replace_tag_in_list(l,tag,val):
    replaced = False
    for i,item in enumerate(l):
        if isinstance(item, dict):
            if replace_tag_in_dict(item, tag, val):
                replaced = True
        elif isinstance(item, list):
            if replace_tag_in_list(item,tag,val):
                replaced = True
        else:
            if item == tag:
                l[i]=val
                replaced = True
    return replaced
def replace_tag_in_dict(d, tag, val):
    replaced = False
    for key, value in d.items():
        if isinstance(value,dict):
            if replace_tag_in_dict(value, tag, val):
                replaced = True
        elif isinstance(value,list):
            if replace_tag_in_list(value,tag,val):
                replaced = True
        else:
            if value==tag:
                d[key]=val
                replaced = True
    return replaced

def simple_ray_tracer_main_w_analysis(parameters):
    if not result_folder in os.listdir():
        os.mkdir(result_folder)
    sub_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    while sub_folder in os.listdir(result_folder):
        sub_folder = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    os.mkdir(result_folder+"/"+sub_folder)
    if 'result_settings' in parameters.keys():
        load_result_settings(parameters)
    if 'display_settings' in parameters.keys():
        load_display(parameters)
    if 'optimization_settings' in parameters.keys():
        optimized_param=run_PSO(parameters)
        parameters = optimized_param
    trains = load_paths(parameters)
    plotted_surfaces=[]
    for i,t in enumerate(trains):
        t.propagate()
        for j,s in enumerate(t.surfaces):
            if not s in plotted_surfaces:
                plotted_surfaces.append(s)

    columns=['id','v1','v2','v3','c1','c2','c3','ang2normal','wv','intensity','lid']
    lines=[]
    lines.append(smooth_line(columns,len(columns))+"\n")
    for sid,s in enumerate(plotted_surfaces):
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
            if plot_last_only is False or sid==(len(plotted_surfaces)-1):
                fig,ax=plt.subplots()
                ax.fill(base[:, 0], base[:, 1], color=(0.5,0.5,0.5))

        for k,stat in enumerate(s.intersects):
            abs_coord = stat[0]
            rel_coord = xdot(s.move,abs_coord)
            rel_coord[0] +=s.radius
            reduc = rel_coord[1:3]
            reduc = np.dot(reduc,rtm)

            rel_vec = normalize(stat[1])
            abs_vec = normalize(xdot(s.inverse, rel_vec) - xdot(s.inverse, np.zeros(3)))
            ang = np.degrees(
                np.arccos(np.dot(-1 * rel_vec, np.array([-1, 0, 0]))))  # ,np.dot(rel_vec,np.array([-1,0,0])))))

            wv = stat[2]
            inten = stat[3]
            lid = stat[4]
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
                info = info + abs_vec + abs_coor + str(ang) + "," + str(wv) + "," + str(inten)+ "," +str(lid)
            else:
                info = info + rel_vec + rel_coor + str(ang) + "," + str(wv) + "," + str(inten)+ "," +str(lid)

            lines.append(info+"\n")
            if isinstance(wv,int):
                if wv!=0:
                    c=wavelength2rgb(wv)
            if show_plot is True:
                if plot_last_only is False or sid == (len(plotted_surfaces) - 1):
                    ax.scatter(reduc[0],reduc[1],color=c,marker='o',alpha=inten)

        if show_plot is True:
            if plot_last_only is False or sid == (len(plotted_surfaces) - 1):
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
                plt.savefig(result_folder+"/"+sub_folder+"/s"+str(s.sid)+".png")
                plt.close(fig)
        with open(result_folder+"/"+sub_folder+"/"+"ray_sur_interactions.csv",'w') as writer:
             writer.writelines(lines)
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
        fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
def _test_current_config(param,candidates:dict,optim_settings:dict):
    if "V1" in candidates.keys():
        replace_tag_in_dict(param, "V1", candidates['V1'])
    if "V2" in candidates.keys():
        replace_tag_in_dict(param, "V2", candidates['V2'])
    if "V3" in candidates.keys():
        replace_tag_in_dict(param, "V3", candidates['V3'])
    if "V4" in candidates.keys():
        replace_tag_in_dict(param, "V4", candidates['V4'])
    t = load_paths(param)[0]
    t.propagate()
    loss= loss_function(t.surfaces[optim_settings['obj']],optim_settings['mode'],optim_settings['param'])
    return loss
def run_current_config(parameters,ks,vs,optim_settings:dict):
    param=copy.deepcopy(parameters)
    for i in range(len(ks)):
        ret = replace_tag_in_dict(param,ks[i],vs[i])
    t = load_paths(param)[0]
    t.propagate()
    loss = loss_function(t.surfaces[optim_settings['obj']], optim_settings['mode'], optim_settings['param'])
    return loss,param

def loss_function(sur: surface, metric="aberrations", params=None,as_it_is = False):
    #metrics:
    #aberrations: how focused are rays
    #angle: applies to collimated light
    #params: list of optional param:
    #        1. the expected angle between rays and the surface normal
    #        2. whether to include distance to center in loss
    sources={}
    for hit in sur.intersects:
        world_coord= hit[0]
        coord = xdot(sur.move,world_coord)
        vec = hit[1]
        lid= hit[4]
        if not lid in sources.keys():
            sources[lid]=[]
        sources[lid].append(np.concatenate([coord,vec]))
    loss = 1000
    denominator = 1
    if metric == "aberrations":
        dist2ctr = False
        if not params is None:
            if params[1] is True:
                dist2ctr = True
        for lid in sources.keys():
            val = np.asarray(sources[lid])
            coords = val[:,0:3]
            vecs = val[:,3:6]
            centroid = np.mean(coords,axis=0)
            disp = coords-centroid.reshape((1,-1)).repeat(len(val),0)
            disp = np.square(disp)
            disp = np.sum(np.sqrt(np.sum(disp,axis=1)))
            loss+=disp
            if dist2ctr is True:
                spreading = xdot(sur.move, coords)
                loss +=np.mean(np.abs(spreading))
            denominator+=len(val)
    elif metric == "angle":
        target_ang = 0
        dist2ctr = False
        if not params is None:
            if isinstance(params,int) or isinstance(params,float):
                target_ang=float(params)
            else:
                if isinstance(params[0],int) or isinstance(params[0],float):
                    target_ang=float(params[0])
                if params[1] is True:
                    dist2ctr = True
        for lid in sources.keys():
            val = np.asarray(sources[lid])
            coords = val[:,0:3]
            vecs = -1*val[:,3:6]
            nor = np.array([-1,0,0])
            dot = np.dot(vecs,nor)
            ang = np.degrees(np.arccos(dot))
            deviation = np.abs(ang-target_ang)
            loss +=np.mean(deviation)
            if dist2ctr is True:
                spreading = xdot(sur.move, coords)
                loss +=np.mean(np.abs(spreading))
            denominator+=len(val)
    if as_it_is is True:
        return loss-1000
    return loss/denominator
class normalizer:
    def __init__(self,policy:dict):
        self.lows=[]
        self.highs=[]
        self.scales=[]
        if "V1" in policy.keys():
            v = policy["V1"][0:2]
            self.lows.append(v[0])
            self.highs.append(v[1])
            self.scales.append(v[1] - v[0])
        if "V2" in policy.keys():
            v = policy["V2"][0:2]
            self.lows.append(v[0])
            self.highs.append(v[1])
            self.scales.append(v[1] - v[0])
        if "V3" in policy.keys():
            v = policy["V3"][0:2]
            self.lows.append(v[0])
            self.highs.append(v[1])
            self.scales.append(v[1] - v[0])
        if "V4" in policy.keys():
            v = policy["V4"][0:2]
            self.lows.append(v[0])
            self.highs.append(v[1])
            self.scales.append(v[1] - v[0])
        self.lows= np.asarray(self.lows,np.float64)
        self.highs=np.asarray(self.highs,np.float64)
        self.scales=np.asarray(self.scales,np.float64)
    def recover(self,vals):
        if isinstance(vals,list):
            out = []
            for i in range(len(vals)):
                out.append(vals[i]*self.scales[i]+self.lows[i])
            return out
        else:
            out = vals*self.scales+self.lows
            return out
    def convert(self,vals):
        if isinstance(vals, list):
            out = []
            for i in range(len(vals)):
                out.append(vals[i] - self.lows[i])
                out[-1]/=self.scales[i]
            return out
        else:
            out = vals-self.lows
            return self.scales*out
def run_PSO(parameters):
    param = copy.deepcopy(parameters)
    load_optimization_settings(param)
    policy = parameters['optimization_settings']
    target = {}
    if "V1" in policy.keys():
        if replace_tag_in_dict(param, "V1", "V1"):
            target['V1'] = policy['V1']
    if "V2" in policy.keys():
        if replace_tag_in_dict(param, "V2", "V2"):
            target['V2'] = policy['V2']
    if "V3" in policy.keys():
        if replace_tag_in_dict(param, "V3", "V3"):
            target['V3'] = policy['V3']
    if "V4" in policy.keys():
        if replace_tag_in_dict(param, "V4", "V4"):
            target['V4'] = policy['V4']
    if len(target)==0:
        return param
    nor = normalizer(policy)
    particles = []
    velocities=[]

    performances=[]
    loci=[]
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    w = 0.5  # Inertia weight
    num_iterations = policy['iterations']
    for d in target.keys():
        arr=np.linspace(0,1,target[d][2])
        target[d]=np.float64(arr)
    if len(target) == 1:
        for v1 in list(target.values())[0]:
            particles.append(np.array([v1]))
            velocities.append(0.5*(np.random.rand()-0.5))
    elif len(target) == 2:
        for v1 in list(target.values())[0]:
            for v2 in list(target.values())[1]:
                particles.append(np.array([v1, v2]))
                velocities.append(0.5*(np.random.rand(2)-0.5))
    elif len(target) == 3:
        for v1 in list(target.values())[0]:
            for v2 in list(target.values())[1]:
                for v3 in list(target.values())[2]:
                    particles.append(np.array([v1, v2, v3]))
                    velocities.append(0.5*(np.random.rand(3)-0.5))
    elif len(target) == 4:
        for v1 in list(target.values())[0]:
            for v2 in list(target.values())[1]:
                for v3 in list(target.values())[2]:
                    for v4 in list(target.values())[3]:
                        particles.append(np.array([v1, v2, v3, v4]))
                        velocities.append(0.5*(np.random.rand(4)-0.5))

    if num_iterations*len(particles)>1000:
        if already_shown[1] is False:
            ret = messagebox.askyesno("Warning","Current setup runs raytracing more than 1000 times\nDo you wish to proceed?")
            if ret:
                already_shown[1]=True
            else:
                messagebox.showinfo("Exit","Optimization cancelled")
                for i, d in enumerate(target.keys()):
                    replace_tag_in_dict(parameters, d, 0)
                return
    keys = list(target.keys())
    num_particles = len(particles)
    best_positions = particles.copy()
    best_fitness = []
    for p in particles:
        rec = nor.recover(p)
        loss,attempt = run_current_config(param,keys,rec,policy)
        best_fitness.append(loss)
        performances.append([])
        loci.append([])
    global_best_index = np.argmin(best_fitness)
    global_best_position = particles[global_best_index].copy()
    history = []
    best_trace_score=[]
    for iteration in range(num_iterations):
        for i in range(num_particles):
            velocity = w * velocities[i] + c1 * np.random.rand() * (
                    best_positions[i] - particles[i]) + c2 * np.random.rand() * (
                               global_best_position - particles[i])
            velocities[i]=velocity
            particles[i] = particles[i] + velocity
            loci[i].append(particles[i][0])
            current_fitness,attempt = run_current_config(param,keys,nor.recover(particles[i]),policy)
            if current_fitness < best_fitness[i]:
                best_fitness[i] = current_fitness
                best_positions[i] = particles[i].copy()
                if current_fitness < best_fitness[global_best_index]:
                    global_best_index = i
                    global_best_position = particles[i].copy()
            performances[i].append(best_fitness[i])
        history.append(best_fitness[global_best_index])
        best_trace_score.append([particles[global_best_index].copy(),best_fitness[global_best_index]])
    recovered_best_params=nor.recover(global_best_position)
    optim_result = {}
    str_res=[]
    info = ""
    for i,d in enumerate(target.keys()):
        replace_tag_in_dict(parameters,d,recovered_best_params[i])
        replace_tag_in_dict(policy,d,recovered_best_params[i])
        optim_result[d]=recovered_best_params[i]
        str_res.append(str(recovered_best_params[i])[0:6])
        info = info+str(d)+"= "+str(recovered_best_params[i])[0:6]+"\n"

    t = load_paths(parameters)[0]
    t.propagate()
    loss = loss_function(t.surfaces[policy['obj']], policy['mode'], policy['param'], as_it_is=True)
    info = info+"loss= "+str(loss)[0:6]

    lines=[]
    lines.append("Optimization results")
    lines.append("Variables: " + "&".join(list(target.keys())))
    lines.append("Best: " + "&".join(str_res))
    lines.append("Loss: " + str(loss))
    lines.append("Iteration,"+",".join(list(target.keys()))+",loss")
    for i in range(num_of_iterations):
        cand = best_trace_score[i]
        loss = cand[1]
        cand = nor.recover(cand[0])
        lines.append(str(i)+","+",".join(list(np.array(cand).astype('str')))+","+str(loss))

    performances = np.asarray(performances)
    best = np.asarray(history).reshape((1,-1))
    sta = np.concatenate([performances,best],axis=0)
    sta = sta.astype('str')
    columns=[]
    for i in range(len(particles)):
        columns.append("P"+str(i))
    columns.append("best")

    with open(result_folder + "/" + "optim_progress.csv", 'w') as writer:
        for l in lines:
            writer.writelines(l+"\n")
        #writer.writelines(",".join(columns)+"\n")
        #for l in range(sta.shape[0]):
        #    writer.writelines(",".join(sta[i].tolist())+"\n")
    messagebox.showinfo("Optimization parameters",info)
    return parameters
def ota():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    red = light(np.array([0, 0, -12]), normalize(np.array([1, 0, 0.249328])), number=5, wavelength=633)
    red.linear(20)
    green = light(np.array([0, 0, -6]), normalize(np.array([1, 0, 0.176327])), number=5, wavelength=532)
    green.linear(20)
    blue = light(np.array([0, 0, 0]), normalize(np.array([1, 0, 0])), number=5, wavelength=488)
    blue.linear(20)

    f1 = surface(coord=None, normal=None, shape="spherical",
                 radius=54.153, semidia=29.213, mode="refraction", n2=1.60738)
    f2 = surface(coord=None, normal=None, shape="spherical",
                 radius=152.522, semidia=28.127, mode="refraction", n1=1.60738)
    f3 = surface(coord=None, normal=None, shape="spherical",
                 radius=36, semidia=24.292, mode="refraction", n2=1.62041)
    f4 = surface(coord=None, normal=None, shape="plano",
                 radius=0, semidia=21.284, mode="refraction", n1=1.62041, n2=1.60342)
    f5 = surface(coord=None, normal=None, shape="spherical",
                 radius=22.27, semidia=14.917, mode="refraction", n1=1.60342)
    f6 = surface(coord=None, normal=None, shape="plano",
                 radius=0, semidia=10.325, mode="aperture")
    f7 = surface(coord=None, normal=None, shape="spherical",
                 radius=-25.68, semidia=13.197, mode="refraction", n2=1.60342)
    f8 = surface(coord=None, normal=None, shape="plano",
                 radius=0, semidia=16.482, mode="refraction", n1=1.60342, n2=1.62041)
    f9 = surface(coord=None, normal=None, shape="spherical",
                 radius=-36.98, semidia=18.942, mode="refraction", n1=1.62041)
    f10 = surface(coord=None, normal=None, shape="spherical",
                  radius=196.41, semidia=21.327, mode="refraction", n2=1.60738)
    f11 = surface(coord=None, normal=None, shape="spherical",
                  radius=-67.14, semidia=21.662, mode="refraction", n1=1.60738)
    f12 = surface(coord=None, normal=None, shape="plano",
                  radius=0, semidia=25, mode="absorption")

    e1 = assembly()
    e1.add(f1, 0)
    e1.add(f2, 8.747)

    e2 = assembly()
    e2.add(f3, 0)
    e2.add(f4, 14)
    e2.add(f5, 3.777)

    e3 = assembly()
    e3.add(f7, 0)
    e3.add(f8, 3.777)
    e3.add(f9, 10.834)

    e4 = assembly()
    e4.add(f10, 0)
    e4.add(f11, 6.858)

    e1.extend(e2, 0.5)
    e1.add(f6, 14.253)
    e1.extend(e3, 12.428)
    e1.extend(e4, 0.5)
    e1.add(f12, 57.315)

    e1.place(np.array([0, 0, 0]), np.array([-1, 0, 0]))

    # focus=surface(coord=None, normal=None,shape="plano",height=20,width=20, mode="refraction")
    # termino = surface(coord=e1.last_surface + np.array([61.24, 0, 0]), normal=None, angles=[0, 0],
    #                  shape="plano", height=20, width=20, mode="refraction")
    # focus.relocate(e1.last_surface+np.array([41.24,0,0]))

    sb = train()
    sb.add(e1)
    # sb.append(focus)
    # sb.append(termino)

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
    ax.view_init(elev=0, azim=-90)
    plt.tight_layout()
    fig.canvas.draw()
    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = np.array(fig.canvas.buffer_rgba())
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data = data[:, :, 0:3]
    plt.show()

def run_gui():
    global last_path
    root = tk.Tk()
    root.title("SRT")

    greeting_label = tk.Label(root, text="Open a ray tracing script")
    greeting_label.pack()

    file_path_frame = tk.Frame(root)
    file_path_entry = tk.Entry(file_path_frame)
    file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    current_path = "default.yml"
    if os.path.exists(last_path):
        current_path = last_path
    file_path_entry.insert(0, str(current_path))

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
            if already_shown[0] is False:
                messagebox.showinfo("Info","Result is stored in SRT_result folder")
                already_shown[0]=True
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
    root.attributes('-topmost',top_most)
    root.bind('<Return>', lambda event: on_run())
    root.mainloop()
    last_path = parameters["path"]
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




