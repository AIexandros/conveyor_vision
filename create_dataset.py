import pickle
from pathlib import Path
import json


def transform_data(data_dir, get_feature_vec):
    data = []
    labels = []
    for dir_ in data_dir.iterdir():
        if dir_.is_dir():
            for img_path in dir_.glob('*.jpg'):
                feature_vec = get_feature_vec(img_path)
                if feature_vec is not None:
                    data.append(feature_vec)
                    labels.append(dir_.name)

    return {'data': data, 'labels': labels}


def create_dataset(data_dir):
    with open(data_dir/'info.json','r') as f:
        info = json.load(f)
        task = info['task']
        
    from neuralnet_features import get_feature_vec
    print('Dataset created succesfully!')
    dataset = transform_data(data_dir,get_feature_vec)

    with open(data_dir/'features.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    return dataset


if __name__=='__main__':
    data_dir = Path('./data')
    dataset = create_dataset(data_dir)
    print(f"Created a {len(dataset['data'])}x{len(dataset['data'][0])} dataset.")