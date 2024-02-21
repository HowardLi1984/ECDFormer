
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:30:18 2020

@author: Licheng Xu and Shuoqing Zhang

The original SPMS calculation module
"""

from rdkit import Chem
import numpy as np
from copy import deepcopy
precision = 8
class SPMS():
    def __init__(self,sdf_file,key_atom_num=None,sphere_radius=None,desc_n=40,desc_m=40,
                 orientation_standard=True,first_point_index_list=None,second_point_index_list=None,third_point_index_list=None):
        
        self.sdf_file = sdf_file
        self.sphere_radius = sphere_radius
        if key_atom_num != None:
            key_atom_num = list(np.array(key_atom_num,dtype=np.int)-1)
            self.key_atom_num = key_atom_num
        else:
            self.key_atom_num = []
        self.desc_n = desc_n
        self.desc_m = desc_m
        self.orientation_standard = orientation_standard
        self.first_point_index_list = first_point_index_list
        self.second_point_index_list = second_point_index_list
        self.third_point_index_list = third_point_index_list


        rdkit_period_table = Chem.GetPeriodicTable()

        mol = Chem.MolFromMolFile(sdf_file,removeHs=False,sanitize=False)
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        atoms = mol.GetAtoms()
        atom_types = [atom.GetAtomicNum() for atom in atoms]
        atom_symbols = [rdkit_period_table.GetElementSymbol(item) for item in atom_types]
        atom_weights = [atom.GetMass() for atom in atoms]
        
        atom_weights = np.array([atom_weights,atom_weights,atom_weights]).T

        weighted_pos = positions*atom_weights

        weight_center = np.round(weighted_pos.sum(axis=0)/atom_weights.sum(axis=0)[0],decimals=precision)

        radius = np.array([rdkit_period_table.GetRvdw(item) for item in atom_types]) # van der Waals radius
        volume = 4/3*np.pi*pow(radius,3)
        self.positions = positions
        self.weight_center = weight_center
        self.radius = radius
        self.volume = volume
        self.atom_types = atom_types
        self.atom_symbols = atom_symbols
        self.rdkit_period_table = rdkit_period_table
        self.atom_weight = atom_weights
    def _Standarlize_Geomertry_Input(self,origin_positions):
        
        if self.key_atom_num == []:
            key_atom_position = deepcopy(self.weight_center)
            key_atom_position = key_atom_position.reshape(1,3)
            distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
            farest_atom_index = np.argmax(distmat_from_key_atom)
            distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
            nearest_atom_index = np.argmin(distmat_from_key_atom)
            second_key_atom_index = nearest_atom_index
            third_key_atom_index = farest_atom_index
            second_atom_position = deepcopy(origin_positions[second_key_atom_index])
            second_atom_position = second_atom_position.reshape(1,3)
            third_atom_position = deepcopy(origin_positions[third_key_atom_index])
            third_atom_position = third_atom_position.reshape(1,3)
            append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

            
        else:
            key_atom_num = self.key_atom_num
            if len(key_atom_num) == 1:
                key_atom_position = deepcopy(origin_positions[key_atom_num[0]])
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

            elif len(key_atom_num) >= 2:
                key_atom_position = origin_positions[key_atom_num].mean(axis=0)
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

                
        OldCoord = np.c_[append_positions, np.ones(len(append_positions))]
        
        first_atom_coord = OldCoord[-3][0:3]
        
        second_atom_coord = OldCoord[-2][0:3]
        Xv =  second_atom_coord-first_atom_coord
        Xv_xy = Xv.copy()
        Xv_xy[2] = 0
        X_v = np.array([Xv[0],0,0])
        Z_v = np.array([0,0,1])
        alpha = np.arccos(Xv_xy[0:3].dot(
                X_v[0:3])/(np.sqrt(Xv_xy[0:3].dot(Xv_xy[0:3]))*np.sqrt(X_v[0:3].dot(X_v[0:3]))))
        beta = np.arccos(Xv[0:3].dot(
                Z_v)/(np.sqrt(Xv[0:3].dot(Xv[0:3]))*np.sqrt(Z_v.dot(Z_v))))
        
        if Xv_xy[1]*Xv_xy[0] > 0:
            alpha = -alpha
        if Xv[0] < 0:
            beta = -beta    
        def T_M(a):
            T_M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [a[0], a[1], a[2], 1]])
            return T_M
        
        def RZ_alpha_M(alpha):
            RZ_alpha_M = np.array([[np.cos(alpha), np.sin(
                alpha), 0, 0], [-np.sin(alpha), np.cos(alpha), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            return RZ_alpha_M
        
        def RY_beta_M(beta):
            RY_beta_M = np.array([[np.cos(beta), 0, np.sin(beta), 0], [
                                 0, 1, 0, 0], [-np.sin(beta), 0, np.cos(beta), 0], [0, 0, 0, 1]])
            return RY_beta_M
        
        a = -first_atom_coord

         
        new_xyz_coord1 = OldCoord.dot(T_M(a)).dot(
            RZ_alpha_M(alpha)).dot(RY_beta_M(beta))    

                
        third_atom_coord = new_xyz_coord1[-1][0:3]

        second_atom_coord = new_xyz_coord1[-2][0:3]
        Xy = third_atom_coord - second_atom_coord
        Y_v = np.array([0, 1, 0])
        gamma = np.arccos(Xy.dot(Y_v)/(np.sqrt(Xy.dot(Xy))*np.sqrt(Y_v.dot(Y_v))))

        if Xy[0] < 0:
            gamma = -gamma
        NewCoord = new_xyz_coord1.dot(RZ_alpha_M(gamma))
        
        third_atom_coord = NewCoord[-1][0:3]
        third_XY = third_atom_coord[0:2]
        axis_y_2d = np.array([0,1])
        sita = np.arccos(third_XY.dot(axis_y_2d)/(np.sqrt(third_XY.dot(third_XY))*np.sqrt(axis_y_2d.dot(axis_y_2d))))

        if third_XY[0]*third_XY[1] < 0:
            sita = -sita
        NewCoord0 = NewCoord.dot(RZ_alpha_M(sita))
        NewCoord1 = np.around(np.delete(NewCoord0, 3, axis=1), decimals=precision)   
        
        NewCoord2 = NewCoord1[:-3]
        New3Points = NewCoord1[-3:]

        return NewCoord2,New3Points        
        
    def _Customized_Coord_Standard(self,positions,first_point_index_list,second_point_index_list,third_point_index_list):
        def T_M(a):            ### translation
            T_M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [a[0], a[1], a[2], 1]])
            return T_M
        
        def RZ_alpha_M(alpha):
            RZ_alpha_M = np.array([[np.cos(alpha), np.sin(
                alpha), 0, 0], [-np.sin(alpha), np.cos(alpha), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            return RZ_alpha_M
        
        def RY_beta_M(beta):
            RY_beta_M = np.array([[np.cos(beta), 0, np.sin(beta), 0], [
                                 0, 1, 0, 0], [-np.sin(beta), 0, np.cos(beta), 0], [0, 0, 0, 1]])
            return RY_beta_M
        def RX_gamma_M(gamma):
            RX_gamma_M = np.array([[1,0,0,0],[0,np.cos(gamma),np.sin(gamma),0],[0,-np.sin(gamma),np.cos(gamma),0],[0,0,0,1]])
            return RX_gamma_M
        
        first_point_index_list = [item-1 for item in first_point_index_list]
        second_point_index_list = [item-1 for item in second_point_index_list]
        third_point_index_list = [item-1 for item in third_point_index_list]
        OldCoord = np.c_[positions, np.ones(len(positions))]
        first_point_coord = np.mean(OldCoord[first_point_index_list],axis=0)[0:3]
        second_point_coord = np.mean(OldCoord[second_point_index_list],axis=0)[0:3]
        Xv =  second_point_coord-first_point_coord
        Xv_xy = Xv.copy()
        Xv_xy[2] = 0
        Y_v_neg = np.array([0,-1,0])
        Y_v_pos = np.array([0,1,0])
        alpha = np.arccos(Xv_xy[0:3].dot(Y_v_neg[0:3])/((np.sqrt(Xv_xy[0:3].dot(Xv_xy[0:3])))*(np.sqrt(Y_v_neg[0:3].dot(Y_v_neg[0:3])))))
        
        a = -first_point_coord
        
        if Xv_xy[0] > 0:
            alpha = -alpha
        new_xyz_coord = OldCoord.dot(T_M(a))     ### translation done
        new_xyz_coord1 = new_xyz_coord.dot(RZ_alpha_M(alpha))
        first_point_coord1 = np.mean(new_xyz_coord1[first_point_index_list],axis=0)[0:3]
        second_point_coord1 = np.mean(new_xyz_coord1[second_point_index_list],axis=0)[0:3]
        Xv1 =  second_point_coord1-first_point_coord1
        Xv1_yz = Xv1.copy()
        Xv1_yz[0] = 0
        gamma = np.pi-np.arccos(Xv1_yz[0:3].dot(Y_v_pos)/((np.sqrt(Xv1_yz[0:3].dot(Xv1_yz[0:3])))*(np.sqrt(Y_v_pos[0:3].dot(Y_v_pos[0:3])))))
        if Xv1[2] < 0:
            gamma = -gamma
        new_xyz_coord2 = new_xyz_coord1.dot(RX_gamma_M(gamma))    ### put one point at the negative y axis
        
        ### rotate around y axis
        third_point_coord = np.mean(new_xyz_coord2[third_point_index_list],axis=0)[0:3]
        Xv3 = third_point_coord.copy()
        Xv3_xz = Xv3.copy()
        Xv3_xz[1] = 0
        X_v_pos = np.array([1,0,0])
        beta = np.arccos(Xv3_xz[0:3].dot(X_v_pos[0:3])/((np.sqrt(Xv3_xz[0:3].dot(Xv3_xz[0:3])))*(np.sqrt(X_v_pos[0:3].dot(X_v_pos[0:3])))))
        if Xv3[2] > 0:
            beta = -beta
        new_xyz_coord3 = new_xyz_coord2.dot(RY_beta_M(beta))
        return new_xyz_coord3[:,0:3]
    
    
    
    def _Standarlize_Geomertry(self):
        if self.orientation_standard == True:
            new_positions,new_3points = self._Standarlize_Geomertry_Input(self.positions)
            if self.key_atom_num != None:
                bias_move = np.array([0.000001,0.000001,0.000001])
                new_positions += bias_move
                new_3points += bias_move
            new_geometric_center,new_weight_center,new_weight_center_2 = new_3points[0],new_3points[1],new_3points[2]
            self.new_positions = new_positions
            self.new_geometric_center,self.new_weight_center,self.new_weight_center_2 = new_geometric_center,new_weight_center,new_weight_center_2
        ############
        # Check this part carefully
        elif self.orientation_standard == False:
            new_positions = self.positions
            self.new_positions = self.positions
        ############
        
        elif self.orientation_standard == "Customized":
            new_positions = self._Customized_Coord_Standard(self.positions,self.first_point_index_list,self.second_point_index_list,self.third_point_index_list)
            self.new_positions = new_positions
        distances = np.sqrt(np.sum(new_positions**2,axis=1))
        self.distances = distances
        distances_plus_radius = distances + self.radius
        sphere_radius = np.ceil(distances_plus_radius.max())
        if self.sphere_radius == None:
            self.sphere_radius = sphere_radius
        
    def _polar2xyz(self,r,theta,fi):
        x = r*np.sin(theta)*np.cos(fi)
        y = r*np.sin(theta)*np.sin(fi)
        z = r*np.cos(theta)
        return np.array([x,y,z])
    def _xyz2polar(self,x,y,z):
        # theta 0-pi
        # fi 0-2pi
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arcsin(np.sqrt(x**2+y**2)/r)
        fi = np.arctan(y/x)
        if z < 0:
            theta = np.pi - theta
        if x < 0 and y > 0:
            fi = np.pi + fi
        elif x < 0 and y < 0:
            fi = np.pi + fi
        elif x > 0 and y < 0:
            fi = 2*np.pi + fi
        return np.array([r,theta,fi])
    def Writegjf(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_types = self.atom_types
        coord_string = ''
        for at_ty,pos in zip(atom_types,new_positions):
            coord_string += '%10d %15f %15f %15f \n'%(at_ty,pos[0],pos[1],pos[2])
        string = '#p\n\nT\n\n0 1\n' + coord_string + '\n'
        with open(file_path,'w') as fw:
            fw.writelines(string)
            
            
    def Writexyz(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_symbols = self.atom_symbols
        
        atom_num = len(atom_symbols)
        coord_string = '%d\ntitle\n'%atom_num
        for at_sy,pos in zip(atom_symbols,new_positions):
            coord_string += '%10s %15f %15f %15f \n'%(at_sy,pos[0],pos[1],pos[2])
        with open(file_path,'w') as fw:
            fw.writelines(coord_string)
    def GetSphereDescriptors(self):
        self._Standarlize_Geomertry()
        new_positions = self.new_positions
        
        radius = self.radius
        sphere_radius = self.sphere_radius

        N = self.desc_n
        M = self.desc_m
        delta_theta = 1/N * np.pi
        delta_fi = 1/M * np.pi
        theta_screenning = np.array([item*delta_theta for item in range(1,N+1)])
        self.theta_screenning = theta_screenning
        fi_screenning = np.array([item*delta_fi for item in range(1,M*2+1)])
        self.fi_screenning = fi_screenning
        PHI, THETA = np.meshgrid(fi_screenning, theta_screenning)

        x = sphere_radius*np.sin(THETA)*np.cos(PHI)
        y = sphere_radius*np.sin(THETA)*np.sin(PHI)
        z = sphere_radius*np.cos(THETA)
        mesh_xyz = np.array([[x[i][j],y[i][j],z[i][j]] for i in range(theta_screenning.shape[0]) for j in range(fi_screenning.shape[0])])
        self.mesh_xyz = mesh_xyz
        psi = np.linalg.norm(new_positions,axis=1)
        atom_vec = deepcopy(new_positions)
        self.psi = psi
        all_cross = []
        for j in range(atom_vec.shape[0]): #######################
            all_cross.append(np.cross(atom_vec[j].reshape(-1,3),mesh_xyz,axis=1)) #############
        all_cross = np.array(all_cross)
        all_cross = all_cross.transpose(1,0,2)
        self.all_cross = all_cross
        mesh_xyz_h = np.linalg.norm(all_cross,axis=2)/sphere_radius

        dot = np.dot(mesh_xyz,atom_vec.T)
        atom_vec_norm = np.linalg.norm(atom_vec,axis=1).reshape(-1,1)
        mesh_xyz_norm = np.linalg.norm(mesh_xyz,axis=1).reshape(-1,1)
        self.mesh_xyz_norm = mesh_xyz_norm
        self.atom_vec_norm = atom_vec_norm
        
        orthogonal_mesh = dot/np.dot(mesh_xyz_norm,atom_vec_norm.T)
        
        self.mesh_xyz_h = mesh_xyz_h
        
        self.orthogonal_mesh = orthogonal_mesh
        
        #cross_det
        cross_det = mesh_xyz_h <= radius
        #orthogonal_det
        orthogonal_det = np.arccos(orthogonal_mesh) <= np.pi*0.5
        double_correct = np.array([orthogonal_det,cross_det]).all(axis=0)
        double_correct_index = np.array(np.where(double_correct==True)).T
        self.double_correct_index = double_correct_index
        d_1 = np.zeros(mesh_xyz_h.shape)
        d_2 = np.zeros(mesh_xyz_h.shape)
        for item in double_correct_index:
            
            d_1[item[0]][item[1]] = max( (psi[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2) ,0)**0.5
            d_2[item[0]][item[1]]=(radius[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2)**0.5
        self.d_1 = d_1
        self.d_2 = d_2
        
        sphere_descriptors = sphere_radius - d_1 - d_2
        sphere_descriptors_compact = sphere_descriptors.min(1)
        sphere_descriptors_reshaped = sphere_descriptors_compact.reshape(PHI.shape)
        sphere_descriptors_reshaped = sphere_descriptors_reshaped.round(precision)
        
        if len(self.key_atom_num) == 1:
            sphere_descriptors_init = np.zeros((theta_screenning.shape[0],fi_screenning.shape[0])) + sphere_radius - self.radius[self.key_atom_num[0]]
            sphere_descriptors_final = np.min(np.concatenate([sphere_descriptors_reshaped.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1),sphere_descriptors_init.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1)],axis=2),axis=2)
        else:
            sphere_descriptors_final = sphere_descriptors_reshaped
        
        self.PHI = PHI
        self.THETA = THETA
        self.sphere_descriptors = sphere_descriptors_final

