import os
import numpy as np
from Bio.PDB import *
from Bio import BiopythonWarning
from scipy.spatial import distance_matrix
from numpy.linalg import norm
from rdkit import Chem
import scipy
import pymesh
import warnings
warnings.simplefilter('ignore', BiopythonWarning)
g_eps = 1.0e-6


class Surface(object):
    def __init__(self, config, radii=None):
        self.msms_bin = config['surface_msms_bin']
        self.surface_save_dir = config['surface_save_dir']

        self.radii = {"N": "1.540000", "O": "1.400000", "F": "1.420000", "C": "1.740000", "H": "1.200000", "S": "1.800000", "P": "1.800000", "Z": "1.39", "X": "0.770000", "B": "2.00", "Cl": "1.90"}
        if radii:
            self.radii = radii
        
        self.polarHydrogens = {"ALA": ["H"], "GLY": ["H"], "SER": ["H", "HG"], "THR": ["H", "HG1"], "LEU": ["H"], "ILE": ["H"], "VAL": ["H"], "ASN": ["H", "HD21", "HD22"], 
            "GLN": ["H", "HE21", "HE22"], "ARG": ["H", "HH11", "HH12", "HH21", "HH22", "HE"], "HIS": ["H", "HD1", "HE2"], "TRP": ["H", "HE1"], "PHE": ["H"], "TYR": ["H", "HH"], 
            "GLU": ["H"], "ASP": ["H"], "LYS": ["H", "HZ1", "HZ2", "HZ3"], "PRO": [], "CYS": ["H"], "MET": ["H"]}
        self.hbond_std_dev = np.pi / 3
        self.donorAtom = {"H": "N", "HH11": "NH1", "HH12": "NH1", "HH21": "NH2", "HH22": "NH2", "HE": "NE", "HD21": "ND2", "HD22": "ND2", "HE21": "NE2", "HE22": "NE2", 
            "HD1": "ND1", "HE2": "NE2", "HE1": "NE1", "HZ1": "NZ", "HZ2": "NZ", "HZ3": "NZ", "HH": "OH", "HG": "OG", "HG1": "OG1"}
        self.acceptorAngleAtom = {"O": "C",  "O1": "C", "O2": "C", "OXT": "C", "OD1": "CG", "OD2": "CG", "OD2": "CB", "OE1": "CD", "OE2": "CD", "ND1": "CE1", "NE2": "CE1",
            "OH": "CZ", "OG": "CB", "OG1": "CB", }
        self.acceptorPlaneAtom = {"O": "CA", "OD1": "CB", "OE1": "CG", "OE2": "CG", "ND1": "NE2", "NE2": "ND1", "OH": "CE1", "OH": "CE1"}

    def calc_pocket_vertice(self, pocket_path=None, cube_size=1.5):
        pocket_ply_dict = self.calc_ply(pocket_path, init=False, cube_size=cube_size)

        return pocket_ply_dict

    def calc_ply(self, pocket_in_path, if_ligand=False, init=False, cube_size=1.5):
        # 计算msms 表面

        vertices1, faces1, normals1, names1, areas1, errno = self.computeMSMS(pocket_in_path, protonate=True, if_ligand=True, cube_size=cube_size)
        if errno:
            return {'vertices': None, 'error_no': -1}

        return {'vertices': vertices1, 'error_no': 0}

    def computeMSMS(self, pdb_file_path,  protonate=True, if_ligand=False, cube_size=1.5):
        out_xyzrn_path = os.path.join(self.surface_save_dir, "tmp.xyzrn")
        if protonate:
            output_pdb_as_xyzrn(pdb_file_path, out_xyzrn_path, if_ligand, radii=self.radii, polarHydrogens=self.polarHydrogens)

        cmd = "{0} -density 3.0 -hdensity 3.0 -probe 1.5 -if {1} -of {2} -af {3} > {4} 2>&1".format(
                self.msms_bin, 
                out_xyzrn_path, 
                self.surface_save_dir, 
                self.surface_save_dir,
                os.path.join(self.surface_save_dir, 'msms.log'),
            )
        ret = os.system(cmd)
        if ret:
            print('********msms_bin error', ret, flush=True)
            return None, None, None, None, None, -1

        # try:
        vertices, faces, _, _ = read_msms(self.surface_save_dir)
        # print('be vertices', len(vertices))
        # vertices = cubefy(vertices, 1.7)
        if cube_size == 0:
            pass
        else:
            vertices = cubefy(vertices, cube_size)
        # print('baf vertices', len(vertices))
        # mesh = pymesh.form_mesh(vertices, faces)
        # regular_mesh = fix_mesh(mesh, self.config['mesh_threshold'])
        # except:
        #     print('********read_msms error', flush=True)
        #     return None, None, None, None, None, -1
        
        return vertices, None, None, None, None, 0

def cubefy(vert, cube_size=1.5):
    # 使用scipy的KDTree进行快速的空间查询
    kdtree = scipy.spatial.cKDTree(vert)
    # 选择一个合适的立方体大小
    cube_size = cube_size  # 1.5

    # 在每个立方体中选择一个点
    sparse_data = []
    for point in vert:
        if len(sparse_data) == 0 or np.min(scipy.spatial.distance.cdist([point], sparse_data)) >= cube_size:
            sparse_data.append(point)
    sparse_data = np.array(sparse_data)

    return sparse_data


def calc_available_pocket(pocket_ply, ligand_ply, min_dist=1, expand_count=3):
    # 产生ligand-pocket距离矩阵A
    ret_dict = {'vertices': []}
    pocket_vertices_set = set()
    
    dist_mat = distance_matrix(ligand_ply['vertices'], pocket_ply['vertices'])
    # 遍历A，找到小于min_dist的pocket点，如果没有小于min_dist，则加入三个最近的点
    for index_ligand, dist_list in enumerate(dist_mat):
        if_have_min_vert = False
        for index_pocket, dist in enumerate(dist_list):
            if dist < min_dist:
                pocket_vertices_set.add(index_pocket)
                if_have_min_vert = True
        if not if_have_min_vert:
            range_list = list(range(len(dist_list)))
            tmp_dict = dict(zip(dist_list, range_list))
            new_dict = sorted(tmp_dict.items(), reverse=False)
            count = 0
            for dist_index_pair in new_dict:
                if count >= expand_count:
                    break
                pocket_vertices_set.add(dist_index_pair[1])
                count += 1
    # 遍历pocket index set 组成对应vert list 和 charge list
    for pocket_vert_index in pocket_vertices_set:
        ret_dict['vertices'].append(pocket_ply['vertices'][pocket_vert_index].tolist())

    return ret_dict


def output_pdb_as_xyzrn(pdb_file_path, out_xyzrn_path, if_ligand=False, radii=None, polarHydrogens=None):
    """
        pdb_file_path: input pdb filename
        out_xyzrn_path: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdb_file_path, pdb_file_path)
    outfile = open(out_xyzrn_path, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if not if_ligand and residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if if_ligand or ( atomtype in radii and resname in polarHydrogens ):
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")


def read_msms(file_root):
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    # Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id


def fix_mesh(mesh, resolution, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    
    target_len = resolution
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.01)

    count = 0;
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);
    # mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.01)
    
    return mesh