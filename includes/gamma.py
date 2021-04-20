from numba import jit,vectorize,guvectorize,cuda
import cupy as cp
from cupyx import scatter_add
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

def load_inputfile(filename):
    nodes = []
    node_sets = {}
    elements = []
    birth_list_element = []
    birth_list_node = []

    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*NODE':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        node_base = int(text[0])
                        first = False
                    nodes.append([float(text[1]),float(text[2]),float(text[3])])
            if line.split()[0] == '*END':
                break
    birth_list_node = [-1 for _ in range(len(nodes))]
    
    
    with open(filename) as f:
        line = next(f)
        while True:
            if not line.split():
                line = next(f)
                continue
            elif line.split()[0] == '*SET_NODE_LIST':
                line = next(f)
                line = next(f)
                key = int(line.split()[0])
                node_list = []
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    for text in line.split():
                        node_list.append(int(text)-node_base)
                node_sets[key] = node_list
            elif line.split()[0] == '*END':
                break
            else:
                line = next(f)
                
                
    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*ELEMENT_SOLID':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        element_base = int(text[0])
                        first = False
                    elements.append([int(text[2])-node_base, int(text[3])-node_base, int(text[4])-node_base, int(text[5])-node_base,
                                     int(text[6])-node_base, int(text[7])-node_base, int(text[8])-node_base, int(text[9])-node_base])
            if line.split()[0] == '*END':
                break

    birth_list_element = [0.0]*len(elements)
    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*DEFINE_CURVE':
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    birth_list_element[int(float(text[1]))-element_base] = float(text[0])
            if line.split()[0] == '*END':
                break
    for element, birth_element in zip(elements, birth_list_element):
        if birth_element < 0:
            continue
        for node in element:
            if (birth_list_node[node] > birth_element or 
                                        birth_list_node[node] < 0):
                                    birth_list_node[node] = birth_element
        
    return np.array(nodes), np.array(birth_list_node), np.array(elements), np.array(birth_list_element)




@jit(nopython=True)
def creatElElConn(elements,connect_surf):
    for i in range (0,elements.shape[0]):
        element = elements[i]
        node = element[0]
        neighbor = np.where(elements==node) 
        for j in neighbor[0]:
            ind = np.zeros(4)
            num = 0
            s_element = elements[j]
            if (i == j):
                continue
            for k in range(0,8):
                for l in range(0,8):
                    if element[k] == s_element[l]:
                        ind[num] = k
                        num = num + 1
                        break
            if ind[0] == 0 and ind[1] == 1 and ind[2] == 2 and ind[3] == 3:
                connect_surf[i][1] = j
            if ind[0] == 0 and ind[1] == 1 and ind[2] == 4 and ind[3] == 5:
                connect_surf[i][2] = j
            if ind[0] == 0 and ind[1] == 3 and ind[2] == 4 and ind[3] == 7:
                connect_surf[i][4] = j

        node = element[6]
        neighbor = np.where(elements==node) 
        for j in neighbor[0]:
            ind = np.zeros(4)
            num = 0
            s_element = elements[j]
            if (i == j):
                continue
            for k in range(0,8):
                for l in range(0,8):
                    if element[k] == s_element[l]:
                        ind[num] = k
                        num = num + 1
                        break
            if ind[0] == 4 and ind[1] == 5 and ind[2] == 6 and ind[3] == 7:
                connect_surf[i][0] = j
            if ind[0] == 2 and ind[1] == 3 and ind[2] == 6 and ind[3] == 7:
                connect_surf[i][3] = j
            if ind[0] == 1 and ind[1] == 2 and ind[2] == 5 and ind[3] == 6:
                connect_surf[i][5] = j
                
                
                
                
def creat_surfaces(elements,element_birth,nodes):
    connect_surf = -np.ones([elements.shape[0],6],dtype=np.int32)
    creatElElConn(elements,connect_surf)
    surfaces = np.zeros([elements.shape[0]*6,4],dtype=np.int32)
    surface_birth = np.zeros([elements.shape[0]*6,2])
    surface_node_coords = np.zeros([elements.shape[0]*6,4,2])
    surface_xy = np.zeros([elements.shape[0]*6,1],dtype=np.int32)
    
    surface_num = 0
    index = np.array([[4,5,6,7],[0,1,2,3],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]])
    coord_ind = np.array([[0,1],[0,1],[0,2],[0,2],[1,2],[1,2]])
    norm_ind = np.array([1,1,0,0,0,0])
    for i in range (0,elements.shape[0]):
        element = elements[i]
        birth_current = element_birth[i]
        for j in range(0,6):
            if connect_surf[i][j] == -1:
                birth_neighbor = 1e10
            else:
                birth_neighbor = element_birth[connect_surf[i][j]]
            if birth_neighbor > birth_current:
                surfaces[surface_num] = element[index[j]]
                surface_birth[surface_num,0] = birth_current
                surface_birth[surface_num,1] = birth_neighbor
                surface_node_coords[surface_num,:,:] = nodes[element[index[j]]][:,coord_ind[j]]
                surface_xy[surface_num] = norm_ind[j]
                surface_num += 1

    surfaces = surfaces[0:surface_num]
    surface_birth = surface_birth[0:surface_num]
    surface_xy = surface_xy[0:surface_num]
    surface_node_coords = surface_node_coords[0:surface_num]
    
    surface_flux = np.zeros([surface_num,1],dtype=np.int32)
    for i in range(0,surface_num):
        if min(nodes[surfaces[i,:]][:,2])>=0:
            surface_flux[i] = 1
    
    return surfaces, surface_birth, surface_xy, surface_node_coords, surface_flux




def load_toolpath(filename = 'toolpath.crs'):
    toolpath_raw=pd.read_table(filename, delimiter=r"\s+",header=None, names=['time','x','y','z','state'])
    return toolpath_raw.to_numpy()

def get_toolpath(toolpath_raw,dt,endtime):
    time = np.arange(0,endtime,dt)
    x = np.interp(time,toolpath_raw[:,0],toolpath_raw[:,1])
    y = np.interp(time,toolpath_raw[:,0],toolpath_raw[:,2])
    z = np.interp(time,toolpath_raw[:,0],toolpath_raw[:,3])

    laser_state = np.interp(time,toolpath_raw[:,0],toolpath_raw[:,4])
    l = np.zeros_like(laser_state)
    for i in range(1,laser_state.shape[0]):
        if laser_state[i] == 1:
            l[i] = 1
        if laser_state[i]>laser_state[i-1]:
            l[i] = 1
    laser_state = l
    laser_state = laser_state* (time<=toolpath_raw[-1,0]) #if time >= toolpath time, stop laser
    
    return np.array([x,y,z,laser_state]).transpose()

def shape_fnc_element(parCoord):
    chsi = parCoord[0]
    eta = parCoord[1]
    zeta = parCoord[2]
    N =  0.125 * np.stack([(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta),(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta),
                           (1.0 + chsi)*(1.0 + eta)*(1.0 - zeta), (1.0 - chsi)*(1.0 + eta)*(1.0 - zeta),
                           (1.0 - chsi)*(1.0 - eta)*(1.0 + zeta), (1.0 + chsi)*(1.0 - eta)*(1.0 + zeta),
                           (1.0 + chsi)*(1.0 + eta)*(1.0 + zeta), (1.0 - chsi)*(1.0 + eta)*(1.0 + zeta)])
    return N
    
def derivate_shape_fnc_element(parCoord):
    oneMinusChsi = 1.0 - parCoord[0]
    onePlusChsi  = 1.0 + parCoord[0]
    oneMinusEta  = 1.0 - parCoord[1]
    onePlusEta   = 1.0 + parCoord[1]
    oneMinusZeta = 1.0 - parCoord[2]
    onePlusZeta  = 1.0 + parCoord[2]
    B = 0.1250 * np.array([[-oneMinusEta * oneMinusZeta, oneMinusEta * oneMinusZeta, 
                                onePlusEta * oneMinusZeta, -onePlusEta * oneMinusZeta, 
                                -oneMinusEta * onePlusZeta, oneMinusEta * onePlusZeta, 
                                onePlusEta * onePlusZeta, -onePlusEta * onePlusZeta],
                              [-oneMinusChsi * oneMinusZeta, -onePlusChsi * oneMinusZeta, 
                               onePlusChsi * oneMinusZeta, oneMinusChsi * oneMinusZeta, 
                               -oneMinusChsi * onePlusZeta, -onePlusChsi * onePlusZeta, 
                               onePlusChsi * onePlusZeta, oneMinusChsi * onePlusZeta],
                               [-oneMinusChsi * oneMinusEta, -onePlusChsi * oneMinusEta, 
                                -onePlusChsi * onePlusEta, -oneMinusChsi * onePlusEta, 
                                oneMinusChsi * oneMinusEta, onePlusChsi * oneMinusEta, 
                                onePlusChsi * onePlusEta, oneMinusChsi * onePlusEta]])
    return B

def shape_fnc_surface(parCoord):
    N = np.zeros((4))
    chsi = parCoord[0]
    eta  = parCoord[1]
    N = 0.25 * np.array([(1-chsi)*(1-eta), (1+chsi)*(1-eta), (1+chsi)*(1+eta), (1-chsi)*(1+eta)])
    return N


def derivate_shape_fnc_surface(parCoord):
    oneMinusChsi = 1.0 - parCoord[0]
    onePlusChsi  = 1.0 + parCoord[0]
    oneMinusEta  = 1.0 - parCoord[1]
    onePlusEta   = 1.0 + parCoord[1]
    B = 0.25 * np.array([[-oneMinusEta, oneMinusEta, onePlusEta, -onePlusEta], 
                         [-oneMinusChsi, -onePlusChsi, onePlusChsi, oneMinusChsi]])
    return B




class domain_mgr():
    def __init__(self,filename,toolpath_file):
        self.filename = filename
        self.toolpath_file = toolpath_file
        
        
        parCoords_element = np.array([[-1.0,-1.0,-1.0],[1.0,-1.0,-1.0],[1.0, 1.0,-1.0],[-1.0, 1.0,-1.0],
                                      [-1.0,-1.0,1.0],[1.0,-1.0, 1.0], [ 1.0,1.0,1.0],[-1.0, 1.0,1.0]]) * 0.5773502692
        parCoords_surface = np.array([[-1.0,-1.0],[-1.0, 1.0],[1.0,-1.0],[1.0,1.0]])* 0.5773502692
        self.Nip_ele = cp.array([shape_fnc_element(parCoord) for parCoord in parCoords_element])[:,:,np.newaxis]
        self.Nip_ele = cp.array([shape_fnc_element(parCoord) for parCoord in parCoords_element])
        self.Bip_ele = cp.array([derivate_shape_fnc_element(parCoord) for parCoord in parCoords_element])
        self.Nip_sur = cp.array([shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
        self.Bip_sur = cp.array([derivate_shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
        
        self.init_domain()
        self.current_time = 0
        self.get_ele_J()
        self.get_surf_ip_pos_and_J()
        
    def init_domain(self):
        # reading input files
        start = time.time()
        nodes, node_birth, elements, element_birth = load_inputfile(self.filename)
        end = time.time()
        print("Time of reading input files: {}".format(end-start))
        self.nodes = cp.asarray(nodes)
        self.nN = self.nodes.shape[0]
        self.node_birth = cp.asarray(node_birth)
        self.elements = cp.asarray(elements)
        self.nE = self.elements.shape[0]
        self.element_birth = cp.asarray(element_birth)
        
        # assign element materials
        ##### modifications needed, from input file
        self.element_mat = cp.ones(self.nE)
        self.element_mat += self.nodes[self.elements][:,:,2].max(axis=1)<=0 # node maxZ<=0 -> substrate material
        self.mat_num = 2
        self.ele_min_Cp_Rho_overCond = [0.368*0.0081/0.01,0.5*0.008/0.0214]
        self.density = [0.0081,0.008]
        
        # calculating critical timestep
        #### modification needed, from input file
        self.defaultFac = 0.95
        start = time.time()
        self.get_timestep()
        end = time.time()
        print("Time of calculating critical timestep: {}".format(end-start))

        
        # reading and interpolating toolpath
        start = time.time()
        toolpath_raw = load_toolpath(filename = self.toolpath_file)
        endtime = 2000;
        toolpath = get_toolpath(toolpath_raw,self.dt,endtime)
        end = time.time()
        print("Time of reading and interpolating toolpath: {}".format(end-start))
        self.toolpath = cp.asarray(toolpath)

        print("Number of nodes: {}".format(len(nodes)))
        print("Number of elements: {}".format(len(elements)))
        print("Number of time-steps: {}".format(len(toolpath)))
        
                
        # generating surface
        start = time.time()
        surface, surface_birth,surface_xy,surface_node_coords,surface_flux = creat_surfaces(elements,element_birth,nodes)
        end = time.time()
        print("Time of generating surface: {}".format(end-start))
        self.surface_node_coords = cp.asarray(surface_node_coords)
        self.surface = cp.asarray(surface)
        self.surface_birth = cp.asarray(surface_birth)
        self.surface_xy = cp.asarray(surface_xy)
        self.surface_flux = cp.asarray(surface_flux)

    def update_birth(self):
        self.active_elements = self.element_birth<self.current_time
        self.active_nodes = self.node_birth<self.current_time
        self.active_surface = (self.surface_birth[:,0]<self.current_time)*(self.surface_birth[:,1]>self.current_time)
    
    def get_ele_J(self):
        nodes_pos = self.nodes[self.elements]
        Jac = cp.matmul(self.Bip_ele,nodes_pos[:,cp.newaxis,:,:].repeat(8,axis=1)) # J = B*x [B:8(nGP)*3(dim)*8(nN), x:nE*8*8*3]
        self.ele_detJac = cp.linalg.det(Jac)
        
        iJac = cp.linalg.inv(Jac) #inv J (nE*nGp*dim*dim)
        self.ele_gradN = cp.matmul(iJac,self.Bip_ele) # dN/dx = inv(J)*B
    
    def get_surf_ip_pos_and_J(self):
        self.surf_ip_pos = self.Nip_sur@self.nodes[self.surface]
        
        nodes_pos = self.nodes[self.surface]
        mapped_surf_nodes_pos = cp.zeros([nodes_pos.shape[0],4,2])
        u = nodes_pos[:,1,:] - nodes_pos[:,0,:]
        v = nodes_pos[:,2,:] - nodes_pos[:,1,:]
        w = nodes_pos[:,3,:] - nodes_pos[:,0,:]
        l1 = cp.linalg.norm(u,axis=1)
        l2 = cp.linalg.norm(v,axis=1)
        l4 = cp.linalg.norm(w,axis=1)
        cos12 = (u[:,0]*v[:,0] + u[:,1]*v[:,1] + u[:,2]*v[:,2])/(l1*l2)
        cos14 = (u[:,0]*w[:,0] + u[:,1]*w[:,1] + u[:,2]*w[:,2])/(l1*l4)
        sin12 = cp.sqrt(1.0 - cos12*cos12)
        sin14 = cp.sqrt(1.0 - cos14*cos14)
        mapped_surf_nodes_pos[:,1,0] = l1
        mapped_surf_nodes_pos[:,2,0] = l1 + l2*cos12
        mapped_surf_nodes_pos[:,2,1] = l2*sin12
        mapped_surf_nodes_pos[:,3,0] = l4*cos14
        mapped_surf_nodes_pos[:,3,1] = l4*sin14
        Jac = cp.matmul(self.Bip_sur,mapped_surf_nodes_pos[:,cp.newaxis,:,:].repeat(4,axis=1))
        self.surf_detJac = cp.linalg.det(Jac)

    def get_timestep(self):
        #element volume
        nodes_pos = self.nodes[self.elements]
        # J = B*x [B:8(nGP)*3(dim)*8(nN), x:nE*8*8*3]
        Jac = cp.matmul(self.Bip_ele,nodes_pos[:,np.newaxis,:,:].repeat(8,axis=1))
        ele_detJac = cp.linalg.det(Jac)
        ele_vol = ele_detJac.sum(axis=1)

        #surface area
        element_surface = self.elements[:,[[4,5,6,7],[0,1,2,3],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]]]
        surf_ip_pos = self.Nip_sur@self.nodes[element_surface]
        nodes_pos = self.nodes[element_surface]
        mapped_surf_nodes_pos = cp.zeros([nodes_pos.shape[0],6,4,2])
        u = nodes_pos[:,:,1,:] - nodes_pos[:,:,0,:]
        v = nodes_pos[:,:,2,:] - nodes_pos[:,:,1,:]
        w = nodes_pos[:,:,3,:] - nodes_pos[:,:,0,:]
        l1 = cp.linalg.norm(u,axis=2)
        l2 = cp.linalg.norm(v,axis=2)
        l4 = cp.linalg.norm(w,axis=2)
        cos12 = (u[:,:,0]*v[:,:,0] + u[:,:,1]*v[:,:,1] + u[:,:,2]*v[:,:,2])/(l1*l2)
        cos14 = (u[:,:,0]*w[:,:,0] + u[:,:,1]*w[:,:,1] + u[:,:,2]*w[:,:,2])/(l1*l4)
        sin12 = cp.sqrt(1.0 - cos12*cos12)
        sin14 = cp.sqrt(1.0 - cos14*cos14)
        mapped_surf_nodes_pos[:,:,1,0] = l1
        mapped_surf_nodes_pos[:,:,2,0] = l1 + l2*cos12
        mapped_surf_nodes_pos[:,:,2,1] = l2*sin12
        mapped_surf_nodes_pos[:,:,3,0] = l4*cos14
        mapped_surf_nodes_pos[:,:,3,1] = l4*sin14

        Jac = cp.matmul(self.Bip_sur,mapped_surf_nodes_pos[:,:,cp.newaxis,:,:].repeat(4,axis=2))
        surf_detJac = cp.linalg.det(Jac)
        ele_surf_area = surf_detJac.sum(axis=2)

        # critical time step
        ele_length = ele_vol/ele_surf_area.max(axis=1)
        self.dt = 1e10
        for i in range(self.mat_num):
            l = ele_length[self.element_mat==i+1].min()
            dt = self.ele_min_Cp_Rho_overCond[i]*l**2/2.0 *self.defaultFac
            self.dt = min(self.dt,dt)
        self.dt = self.dt.get().item()

class heat_solve_mgr():
    def __init__(self,domain):
        ##### modification needed, from files
        self.domain = domain
        self.ambient = 300
        self.r_beam = 1.5
        self.q_in = 250
        self.h_conv = 0.00005
        self.h_rad = 0.2
        
        # initialization
        self.temperature = self.ambient*cp.ones(self.domain.nodes.shape[0])
        self.current_step = 0
        self.rhs = cp.zeros(self.domain.nN)
        self.m_vec = cp.zeros(self.domain.nN)
        self.density_Cp_Ip = cp.zeros([domain.nE,8])
        self.Cond_Ip = cp.zeros([domain.nE,8])
        self.melt_depth = 0
        
    def update_cp_cond(self):
        domain=self.domain
        elements = domain.elements
        temperature_nodes = self.temperature[elements]
        temperature_ip = (domain.Nip_ele[:,cp.newaxis,:]@temperature_nodes[:,cp.newaxis,:,cp.newaxis].repeat(8,axis=1))[:,:,0,0]
        
        self.density_Cp_Ip *= 0
        self.Cond_Ip *= 0
        ##### temp-dependent, modification needed, from files
        solidus1 = 1533.15
        liquidus1 = 1609.15
        latent1 = 272/(liquidus1-solidus1)
        mat1 = domain.element_mat ==1
        thetaIp = temperature_ip[domain.active_elements*mat1]
        self.density_Cp_Ip[domain.active_elements*mat1] += domain.density[0]*latent1 * (thetaIp>solidus1)*(thetaIp<liquidus1)
        thetaIp = cp.clip(thetaIp,self.ambient,solidus1)
        self.density_Cp_Ip[domain.active_elements*mat1] += domain.density[0]*(0.36024 + 2.6e-5*thetaIp - 4e-9*thetaIp**2)
        self.Cond_Ip[domain.active_elements*mat1] += 5.6e-4 + 2.9e-5 * thetaIp - 7e-9*thetaIp**2;

        solidus2 = 1648.15
        liquidus2 = 1673.15
        latent2 = 272.5/(liquidus1-solidus2)
        mat2 = domain.element_mat ==2
        thetaIp = temperature_ip[domain.active_elements*mat2]
        self.density_Cp_Ip[domain.active_elements*mat2] += domain.density[1]*latent2*(thetaIp>solidus2)*(thetaIp<liquidus2)
        self.density_Cp_Ip[domain.active_elements*mat2] += domain.density[1]*0.5000
        self.Cond_Ip[domain.active_elements*mat2] += 0.0214

   
    def update_mvec_stifness(self):
        nodes = self.domain.nodes
        elements = self.domain.elements[self.domain.active_elements]
        Bip_ele = self.domain.Bip_ele
        Nip_ele = self.domain.Nip_ele
        temperature_nodes = self.temperature[elements]
        
        detJac = self.domain.ele_detJac[self.domain.active_elements]
        density_Cp_Ip = self.density_Cp_Ip[self.domain.active_elements]
        mass = cp.sum((density_Cp_Ip * detJac)[:,:,cp.newaxis,cp.newaxis] 
                      * Nip_ele[:,:,cp.newaxis]@Nip_ele[:,cp.newaxis,:],axis=1)
        lump_mass= cp.sum(mass,axis=2)

        gradN = self.domain.ele_gradN[self.domain.active_elements]
        Cond_Ip = self.Cond_Ip[self.domain.active_elements]
        stiffness = cp.sum((Cond_Ip * detJac)[:,:,cp.newaxis,cp.newaxis] * gradN.transpose([0,1,3,2])@gradN,axis = 1)
        stiff_temp = stiffness@temperature_nodes[:,:,cp.newaxis]
        
        self.rhs *= 0
        self.m_vec *= 0

        scatter_add(self.rhs,elements.flatten(),-stiff_temp.flatten())
        scatter_add(self.m_vec,elements.flatten(),lump_mass.flatten())
        

    def update_fluxes(self):
        surface = self.domain.surface[self.domain.active_surface]
        nodes = self.domain.nodes
        Nip_sur = self.domain.Nip_sur
        Bip_sur = self.domain.Bip_sur
        surface_node_coords = self.domain.surface_node_coords[self.domain.active_surface]
        surface_xy  = self.domain.surface_xy[self.domain.active_surface]
        surface_flux = self.domain.surface_flux[self.domain.active_surface]

        q_in = self.q_in
        h_conv =self.h_conv
        ambient = self.ambient
        h_rad = self.h_rad
        r_beam = self.r_beam
        laser_loc = self.laser_loc
        laser_state = self.laser_state
        
        ip_pos = self.domain.surf_ip_pos[self.domain.active_surface]
    
        r2 = cp.square(cp.linalg.norm(ip_pos-laser_loc,axis=2))
        qmov = 3.0 * q_in * laser_state /(cp.pi * r_beam**2)*cp.exp(-3.0 * r2 / (r_beam**2)) * surface_xy 

        temperature_nodes = self.temperature[surface]
        temperature_ip = Nip_sur@temperature_nodes[:,:,cp.newaxis]

        qconv = -1 * h_conv * (temperature_ip - ambient)
        qconv = qconv[:,:,0]*surface_flux
        
        qrad = -1 * 5.6704e-14 * h_rad * (temperature_ip**4 - ambient**4)
        qrad = qrad [:,:,0]*surface_flux

        detJac = self.domain.surf_detJac[self.domain.active_surface]
        q = ((qmov+qrad+qconv)*detJac)[:,:,cp.newaxis].repeat(4,axis=2)*Nip_sur
        scatter_add(self.rhs,surface.flatten(),q.sum(axis=1).flatten())

    def time_integration(self):
        domain = self.domain
        self.current_step += 1
        domain.current_time += domain.dt
        domain.update_birth()

        self.update_cp_cond()
        self.update_mvec_stifness()

        self.laser_loc = domain.toolpath[self.current_step,0:3]
        self.laser_state = domain.toolpath[self.current_step,3]
        self.update_fluxes()

        self.temperature[domain.active_nodes] += domain.dt*self.rhs[domain.active_nodes]/self.m_vec[domain.active_nodes]
        self.temperature[cp.where(domain.nodes[:,2]==-20)]=300
    
    def calculate_melt(self):
        domain = self.domain
        elements = domain.elements[domain.active_elements]
        temperature_ele_nodes = self.temperature[elements]

        temperature_ele_max = temperature_ele_nodes.max(axis = 1)
        solidus = 1533.15
        elements = elements[temperature_ele_nodes[:,4:8].max(axis=1)>=solidus]
        temperature_ele_nodes = self.temperature[elements]
        elements = elements[temperature_ele_nodes[:,0:4].max(axis=1)<=solidus]
        temperature_ele_nodes = self.temperature[elements]
        
        self.melt_depth = 0
        if elements.shape[0]>0:
            
            ele_nodes_pos = domain.nodes[elements]


            temperature_ele_top = temperature_ele_nodes[:,4:8].max(axis=1)
            temperature_ele_bot = temperature_ele_nodes[:,0:4].max(axis=1)
            z_top = ele_nodes_pos[:,4,2]
            z_bot = ele_nodes_pos[:,0,2]
            melt_z =  z_bot + (z_top - z_bot) * (solidus - temperature_ele_bot) / (temperature_ele_top - temperature_ele_bot);

            self.melt_depth = self.laser_loc[2] - melt_z.min()
