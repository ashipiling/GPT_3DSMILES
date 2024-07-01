from multiprocessing import Pool
import traceback


def coords_str2coords(coord_str):
    coord = []
    coord_str_list = coord_str.split('{')[1:]
    for item in coord_str_list:
        l = item.split(',')
        tmp_l = []
        for i in l:
            xyz = float(i) / 10
            tmp_l.append(xyz)
        coord.append(tmp_l)

    return coord


def reorder_coords(coords):
    new_coords = sorted(coords)
    return new_coords


def encode_number(num):
    if num > 199.9 or num < -199.9:
        print('coords 太大')
        return -1, None

    num = int(round(num * 10, 0))
    prefix = ''
    if abs(num) != num:
        prefix = '-'
    num_str = prefix + '0' * (4 - len(str(abs(num)))) + str(abs(num))

    return 0, num_str


def coords_2_str(coords):
    three_dimension_smiles = ''
    for idx in range(len(coords)):
        ret, x = encode_number(coords[idx][0])
        if ret:
            return None
        ret, y = encode_number(coords[idx][1])
        if ret:
            return None
        ret, z = encode_number(coords[idx][2])
        if ret:
            return None

        three_dimension_smiles += '{' + str(x) + ',' + str(y) + ',' + str(z)

    return three_dimension_smiles


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    res = []
    for i in range(n):
        l = lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        res.append({'lst': l, 'process_id': i})
    return res


def main(dic):
    ret_list = []
    for index, line in enumerate(dic['lst']):
        try:
            coords_str = line.strip().split('|')[0]
            coords = coords_str2coords(coords_str)
            new_coords = reorder_coords(coords)
            new_coords_str = coords_2_str(new_coords)

            new_string = new_coords_str + '|' + '|'.join(line.strip().split('|')[1:])
            ret_list.append(new_string)
        except:
            pass

        if index % 1000 == 0:
            print('index', index)
        
    return ret_list


lines = open('../data/3dsmiles_msms_alignxyz_40.txt', 'r').readlines()
new_path = '../data/3dsmiles_msms_alignxyz_40_reorderpocket.txt'
n_worker = 50
# 启用多进程

with Pool(processes=n_worker) as pool:
    results = pool.map(main, split_list(lines, n_worker))

ret_list = [item for sublist in results for item in sublist]

# 保存3dsmiles
with open(new_path, 'w') as f:
    for txt in ret_list:
        f.write(txt + '\n')
