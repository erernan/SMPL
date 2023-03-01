import numpy as np
import pickle
import sys
from chumpy import ch

output_path = './model.pkl'

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Error: expected source model path.')
    exit(-1)
  src_path = sys.argv[1]
  with open(src_path, 'rb') as f:
    src_data = pickle.load(f, encoding="latin1")
  model = {
    # beta 2 joint mapping, 24 * 6890
    'J_regressor': src_data['J_regressor'],
    # blend skinning 6890 * 24
    'weights': np.array(src_data['weights']),
    # pose 2 shape 6890 * 3 * 207
    'posedirs': np.array(src_data['posedirs']),
    # mean template shape 6890 * 3
    'v_template': np.array(src_data['v_template']),
    # beta 2 shape 6890 * 3 * 10
    'shapedirs': np.array(src_data['shapedirs']),
    # faces 13776 * 3
    'f': np.array(src_data['f']),
    # 运动学树
    'kintree_table': src_data['kintree_table']
  }
  if 'cocoplus_regressor' in src_data.keys():
    model['joint_regressor'] = src_data['cocoplus_regressor']
  with open(output_path, 'wb') as f:
    pickle.dump(model, f)
