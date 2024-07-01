import os
import traceback
import os
import subprocess
from rdkit import Chem
import numpy as np
from rdkit import Chem, RDLogger
import subprocess
import sys
from vina import Vina
import math
import prody

RDLogger.DisableLog('rdApp.*')

# 设置环境变量
os.environ['MGLPYTHON'] = '/mnt/cfs/users/luohao/software/mgltool/bin/pythonsh'
os.environ['PREPARE_RECEPTOR'] = '/mnt/cfs/users/luohao/software/mgltool/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'
os.environ['PREPARE_LIGAND'] = '/mnt/cfs/users/luohao/software/mgltool/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'


class Vscore_Reward(object):
    def __init__(
            self, 
            score_save_dir, 
            pocket_pdb_path, 
            ligand_mol, 
            pocket_name='14gs',
            need_init_score=False,
            need_clean=False
        ):
        self.save_dir = score_save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.pocket_name = pocket_name
        self.pocket_pdb_path = pocket_pdb_path
        self.ligand_mol = ligand_mol

        self.pocket_pdbqt_path = os.path.join(self.save_dir, self.pocket_name + "_pocket.pdbqt")
        self.ligand_sdf_path = os.path.join(self.save_dir, self.pocket_name + "_ligand.sdf")
        self.ligand_mol2_path = os.path.join(self.save_dir, self.pocket_name + "_ligand.mol2")
        self.ligand_pdbqt_path = os.path.join(self.save_dir, self.pocket_name + "_ligand.pdbqt")
        
        if need_clean:
            self.clean_old_data()
        else:
            self.cp_pdbqt()
        if pocket_pdb_path:
            self.prepare_pocket()
        if ligand_mol:
            self.prepare_ligand(self.ligand_mol)
        if need_init_score:
            self.score = self.get_xscore()
            # print('init score', self.score, flush=True)

    def prepare_pocket(self):
        command = '$MGLPYTHON $PREPARE_RECEPTOR -r {protein} -o {protein_pdbqt}'.format(
            protein=self.pocket_pdb_path,
            protein_pdbqt=self.pocket_pdbqt_path
        )

        if os.path.exists(self.pocket_pdbqt_path):
            pass
        else:
            os.system(command)
            # proc = subprocess.run(
            #     command, 
            #     shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            # )
  
    def prepare_ligand(self, state):
        state = Chem.RemoveHs(state)
        writer = Chem.SDWriter(self.ligand_sdf_path)
        writer.write(state)
        writer.close()

        command = 'obabel -isdf ' + self.ligand_sdf_path + ' -omol2 -O ' + self.ligand_mol2_path
        proc = subprocess.run(
                command, 
                shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        
        command = '''$MGLPYTHON $PREPARE_LIGAND -l {lig_mol2} \
                -A hydrogens -o {qt_file} \
                '''.format(lig_mol2=self.ligand_mol2_path, qt_file=self.ligand_pdbqt_path)
        # ret = os.system(command)
        proc = subprocess.run(
            command, 
            shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def get_xscore(self):
        pocket_center, box_size = self.get_box_from_sdf()
        try:
            score = self.vina_s(pocket_center, box_size)
            mini_score = self.vina_s(pocket_center, box_size, mode='minimize')
            print(score, mini_score)
        except:
            score = 10
            mini_score = 10
        return score

    def vina_s(self, pocket_center, box_size, seed=0, mode='score_only'):
        v = Vina(sf_name='vina', seed=seed, verbosity=0)
        v.set_receptor(self.pocket_pdbqt_path)
        v.set_ligand_from_file(self.ligand_pdbqt_path)
        v.compute_vina_maps(center=pocket_center, box_size=box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        return score
    
    def get_box_from_sdf(self, buffer=20):

        atoms = []
        for atom in self.ligand_mol.GetAtoms():
            pos = self.ligand_mol.GetConformer().GetAtomPosition(atom.GetIdx())
            atoms.append((pos.x, pos.y, pos.z))

        xs = [atom[0] for atom in atoms]
        ys = [atom[1] for atom in atoms]
        zs = [atom[2] for atom in atoms]
        pocket_center = [(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2]
        box_size = [buffer, buffer, buffer]
        return np.array(pocket_center).tolist(), np.array(box_size).tolist()
    
    def clean_old_data(self):
        try:
            os.remove(self.pocket_pdbqt_path)
        except:
            traceback.print_exc()
    
    def cp_pdbqt(self):
        cmd = 'cp {sourcepdbqt} {targetpdbqt}'.format(
            sourcepdbqt = os.path.join(self.save_dir, '..', self.pocket_name + "_pocket.pdbqt"),
            targetpdbqt = self.pocket_pdbqt_path
        )
        os.system(cmd)
