from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import random
from scipy.spatial import distance_matrix
import math
import copy
import prody


def get_random_rotation(max_angle=60):

    angle = np.random.uniform(-max_angle, max_angle)
    axis = np.random.uniform(-1, 1, size=3)
    axis /= np.linalg.norm(axis)

    c = math.cos(angle * math.pi / 180)
    s = math.sin(angle * math.pi / 180)
    
    rotation_matrix = np.array([[c + axis[0]**2 * (1 - c), axis[0] * axis[1] * (1 - c) - axis[2] * s, axis[0] * axis[2] * (1 - c) + axis[1] * s],
                                [axis[0] * axis[1] * (1 - c) + axis[2] * s, c + axis[1]**2 * (1 - c), axis[1] * axis[2] * (1 - c) - axis[0] * s],
                                [axis[0] * axis[2] * (1 - c) - axis[1] * s, axis[1] * axis[2] * (1 - c) + axis[0] * s, c + axis[2]**2 * (1 - c)]])

    return rotation_matrix


def calc_energy(ligand_coords, pocket_coords):
    dist = distance_matrix(ligand_coords, pocket_coords)
    # 能量定义
    # 斥力随着距离越近越大
    repulsion_strength = 100
    vdw_radius = 3.4
    # 距离大于10的不计算
    repulsion_energy = np.sum(4 * (pow(vdw_radius / (dist + 1e-6), 12) - pow(vdw_radius / (dist + 1e-6), 6)) * (dist < 5))
    
    # 电荷力随着距离越近越大,但远小于斥力
    # 距离大于10的不计算
    charge_strength = 2
    charge_energy = 0
    # charge_energy = np.sum(charge_strength / (dist + 1e-6)** 2 * (dist < 6))

    total_energy = repulsion_energy - charge_energy
    return total_energy


def get_best_ligand_mol(ori_ligand_mol, best_energy, protein_coords, all_steps=100, max_translation=5, max_rotation=90):
    best_ligand_mol = Chem.Mol(ori_ligand_mol)
    for step in range(all_steps):
        new_ligand_mol = Chem.Mol(ori_ligand_mol)
        ligand_coords = new_ligand_mol.GetConformer().GetPositions()
        # 随机旋转和平移
        rotation_matrix = get_random_rotation(max_rotation)
        translation = np.random.uniform(-max_translation, max_translation, size=3)
        center_coord = copy.deepcopy(np.mean(ligand_coords, axis=0))
        new_coords = ligand_coords - center_coord

        new_coords = np.array([np.dot(rotation_matrix, coord) for coord in new_coords])
        new_coords += center_coord + translation

        new_energy = calc_energy(new_coords, protein_coords)
        if new_energy < best_energy:
            best_energy = new_energy

            conformer = new_ligand_mol.GetConformer()
            for j in range(conformer.GetNumAtoms()):
                x, y, z = new_coords[j]
                conformer.SetAtomPosition(j, (x, y, z))

            best_ligand_mol = Chem.Mol(new_ligand_mol)
    return best_ligand_mol, best_energy


def get_pocket_coords(pocket_path):
    pocket = prody.parsePDB(pocket_path)
    pocket_coords = pocket.select('protein and not hydrogen').getCoords()
    return pocket_coords


def remove_clashes(
        ligand_mol, pocket_path,
        max_translation=5, max_rotation=150,
        num_trys=16, num_steps=4, start_fold=1, add_fold=2
    ):
    ligand_coords = ligand_mol.GetConformer().GetPositions()
    protein_coords = get_pocket_coords(pocket_path)

    # rotation_limit_list = [max_rotation, max_rotation/2, max_rotation/3, max_rotation/4, max_rotation/5]
    # translation_list = [max_translation, max_translation/2, max_translation/3, max_translation/4, max_translation/5]
        
    rotation_limit_list = [
        max_rotation / (start_fold + add_fold * 0), 
        max_rotation / (start_fold + add_fold * 1), 
        max_rotation / (start_fold + add_fold * 2), 
        max_rotation / (start_fold + add_fold * 3)
    ]
    translation_list = [
        max_translation / (start_fold + add_fold * 0), 
        max_translation / (start_fold + add_fold * 1), 
        max_translation / (start_fold + add_fold * 2), 
        max_translation / (start_fold + add_fold * 3)
    ]
    min_steps_list = [60, 50, 40, 30]
    best_energy = calc_energy(ligand_coords, protein_coords)

    # 每100个steps后挑取一个最好的，然后缩小范围，再进行100个steps，直到最后一个mol
    # 循环进行6次，得到6个最终的mol
    # 6个mol进行排序，取最好的一个
    ret_mol_list = []

    for i in range(num_trys):

        j = 0
        best_energy_in_try = best_energy
        best_ligand_mol_in_try = Chem.Mol(ligand_mol)

        while j < num_steps:
            
            best_ligand_mol_in_steps, best_energy_in_steps = get_best_ligand_mol(
                best_ligand_mol_in_try,
                best_energy_in_try,
                protein_coords,
                all_steps=min_steps_list[j],
                max_translation=translation_list[j],
                max_rotation=rotation_limit_list[j]
            )
            if best_energy_in_steps < best_energy_in_try:
                best_energy_in_try = best_energy_in_steps
                best_ligand_mol_in_try = Chem.Mol(best_ligand_mol_in_steps)

            j += 1
        
        ret_mol_list.append((best_ligand_mol_in_try, best_energy_in_try))
    
    ret_mol_list = sorted(ret_mol_list, key=lambda x: x[1])
    return ret_mol_list[0][0]

