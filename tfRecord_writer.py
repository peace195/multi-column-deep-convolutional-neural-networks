import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image as pil_image
import random

def load_image(addr):
    image = pil_image.open(addr)
    image = np.asarray(image.resize((227, 227), pil_image.ANTIALIAS), dtype=np.float32)
    return image

def load_block_image(flower_dir, leaf_dir, branch_dir, entire_dir):
    block_img = np.zeros((4, 227, 227, 3), dtype=np.float32)
    if flower_dir != 'None':
        flower_img = load_image(flower_dir)
        if len(flower_img.shape) != 3 or flower_img.shape[2] != 3:
            return np.zeros((227, 227, 3))
        block_img[0] = flower_img
    if leaf_dir != 'None':
        leaf_img = load_image(leaf_dir)
        if len(leaf_img.shape) != 3 or leaf_img.shape[2] != 3:
            return np.zeros((227, 227, 3))
        block_img[1] = leaf_img
    if branch_dir != 'None':
        branch_img = load_image(branch_dir)
        if len(branch_img.shape) != 3 or branch_img.shape[2] != 3:
            return np.zeros((227, 227, 3))
        block_img[2] = branch_img
    if entire_dir != 'None':
        entire_img = load_image(entire_dir)
        if len(entire_img.shape) != 3 or entire_img.shape[2] != 3:
            return np.zeros((227, 227, 3))
        block_img[3] = entire_img
    return block_img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


data_dir = '../50_species_simple_collection/'
plant_list = ['flower', 'leaf', 'branch', 'entire']
image_size = 227
label_id = 0
species_dict = {}
data = {}

for sub_dir_plant in os.listdir(data_dir):
    img_dir = []
    img_labels = []
    for sub_dir_species in os.listdir(data_dir + sub_dir_plant + '/Training/'):
        try:
            for img in os.listdir(data_dir + sub_dir_plant + '/Training/' + sub_dir_species):
                if not sub_dir_species in species_dict:
                    species_dict[sub_dir_species] = label_id
                    label_id = label_id + 1

                img_dir.append(os.path.join(data_dir + sub_dir_plant + '/Training/' + sub_dir_species, img))
                img_labels.append(species_dict[sub_dir_species])
        except OSError:
            continue
        
    data[sub_dir_plant + '_train'] = img_dir
    data[sub_dir_plant + '_train_labels'] = img_labels

for sub_dir_plant in os.listdir(data_dir):
    img_dir = []
    img_labels = []
    for sub_dir_species in os.listdir(data_dir + sub_dir_plant + '/Testing/'):
        try:
            for img in os.listdir(data_dir + sub_dir_plant + '/Testing/' + sub_dir_species):
                img_dir.append(os.path.join(data_dir + sub_dir_plant + '/Testing/' + sub_dir_species, img))
                img_labels.append(species_dict[sub_dir_species])
        except OSError:
            continue

    data[sub_dir_plant + '_test'] = img_dir
    data[sub_dir_plant + '_test_labels'] = img_labels

for sub_dir_plant in os.listdir(data_dir):
    img_dir = []
    img_labels = []
    for sub_dir_species in os.listdir(data_dir + sub_dir_plant + '/SvmInput'):
        try:
            for img in os.listdir(data_dir + sub_dir_plant + '/SvmInput/' + sub_dir_species):
                img_dir.append(os.path.join(data_dir + sub_dir_plant + '/SvmInput/' + sub_dir_species, img))
                img_labels.append(species_dict[sub_dir_species])
        except OSError:
            continue

    data[sub_dir_plant + '_svm'] = img_dir
    data[sub_dir_plant + '_svm_labels'] = img_labels

for organ in plant_list:
    for dataset_type in ['test', 'svm']:
        f = open(organ + '_' + dataset_type, 'r')
        img_dir = []
        img_labels = []
        for line in f:
            img_dir.append(line.strip().split(';')[0])
            img_labels.append(species_dict[line.strip().split(';')[1]])

        data['my_' + organ + '_' + dataset_type] = img_dir
        data['my_' + organ + '_' + dataset_type + '_labels'] = img_labels
        f.close()

label_count = len(species_dict)


img_dir = []
img_labels = []
for i in xrange(len(plant_list) - 1):
    for j in xrange(i + 1, len(plant_list)):
        for m in xrange(len(data[plant_list[i] + '_train_labels'])):
            for n in xrange(len(data[plant_list[j] + '_train_labels'])):
                if data[plant_list[i] + '_train_labels'][m] == data[plant_list[j] + '_train_labels'][n]:
                    img_labels.append(data[plant_list[i] + '_train_labels'][m])
                    img_dir.append((data[plant_list[i] + '_train'][m], data[plant_list[j] + '_train'][n]))

pi = np.random.permutation(len(img_dir))
data['pair_images'] = np.array(img_dir)[pi]
data['pair_labels'] = np.array(img_labels)[pi]

writer = tf.python_io.TFRecordWriter('mcdnn_train.tfrecords')
train_addrs = data['pair_images']
train_labels = data['pair_labels']
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
        sys.stdout.flush()
    # Load the image
    if 'flower' in train_addrs[i][0]:
        if 'leaf' in train_addrs[i][1]:
            img = load_block_image(train_addrs[i][0], train_addrs[i][1], 'None', 'None')
        elif 'branch' in train_addrs[i][1]:
            img = load_block_image(train_addrs[i][0], 'None', train_addrs[i][1], 'None')
        else:
            img = load_block_image(train_addrs[i][0], 'None', 'None', train_addrs[i][1])
    elif 'leaf' in train_addrs[i][0]:
        if 'branch' in train_addrs[i][1]:
            img = load_block_image('None', train_addrs[i][0], train_addrs[i][1], 'None')
        else:
            img = load_block_image('None', train_addrs[i][0], 'None', train_addrs[i][1])
    elif 'branch' in train_addrs[i][0]:
        img = load_block_image('None', 'None', train_addrs[i][0], train_addrs[i][1])
    if img.shape[0] != 4:
        continue
    label = train_labels[i]
    # Create a feature
    feature = {'mcdnntrain/label': _int64_feature(label),
               'mcdnntrain/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
'''
organ1 = data['my_flower_test']
organ2 = data['my_leaf_test']
labels = data['my_flower_test_labels']
writer = tf.python_io.TFRecordWriter('flower_leaf.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image(organ1[i], organ2[i], 'None', 'None')
    if img.shape[0] != 4:
        print "okk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

organ1 = data['my_flower_test']
organ2 = data['my_branch_test']
labels = data['my_flower_test_labels']
writer = tf.python_io.TFRecordWriter('flower_branch.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image(organ1[i],  'None', organ2[i], 'None')
    if img.shape[0] != 4:
        print "okkk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

organ1 = data['my_flower_test']
organ2 = data['my_entire_test']
labels = data['my_flower_test_labels']
writer = tf.python_io.TFRecordWriter('flower_entire.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image(organ1[i], 'None', 'None', organ2[i])
    if img.shape[0] != 4:
        print "okkkk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

organ1 = data['my_leaf_test']
organ2 = data['my_branch_test']
labels = data['my_leaf_test_labels']
writer = tf.python_io.TFRecordWriter('leaf_branch.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image('None', organ1[i], organ2[i], 'None')
    if img.shape[0] != 4:
        print "okkkkk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

organ1 = data['my_leaf_test']
organ2 = data['my_entire_test']
labels = data['my_leaf_test_labels']
writer = tf.python_io.TFRecordWriter('leaf_entire.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image('None', organ1[i], 'None', organ2[i])
    if img.shape[0] != 4:
        print "okkkkkk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

organ1 = data['my_branch_test']
organ2 = data['my_entire_test']
labels = data['my_branch_test_labels']
writer = tf.python_io.TFRecordWriter('branch_entire.tfrecords')
for i in range(len(organ1)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'mcdnn data: {}/{}'.format(i, len(organ1))
        sys.stdout.flush()
    # Load the image
    img = load_block_image('None', 'None', organ1[i], organ2[i])
    if img.shape[0] != 4:
        print "okkkkkkk"
        continue
    label = labels[i]
    # Create a feature
    feature = {'mcdnntest/label': _int64_feature(label),
               'mcdnntest/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

for organ in plant_list:
    train_addrs = data[organ + '_train']
    train_labels = data[organ + '_train_labels']
    train_filename = organ + '_train.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print 'Train data: {}/{}'.format(i, len(train_addrs))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue

        label = train_labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

for organ in plant_list:
    train_addrs = data[organ + '_test']
    train_labels = data[organ + '_test_labels']
    train_filename = organ + '_test.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print 'Test data: {}/{}'.format(i, len(train_addrs))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue

        label = train_labels[i]
        # Create a feature
        feature = {'test/label': _int64_feature(label),
                   'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

for organ in plant_list:
    train_addrs = data[organ + '_svm']
    train_labels = data[organ + '_svm_labels']
    train_filename = organ + '_svm.tfrecords'  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print 'Svm data: {}/{}'.format(i, len(train_addrs))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue
        label = train_labels[i]
        # Create a feature
        feature = {'svm/label': _int64_feature(label),
                   'svm/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

for organ in plant_list:
    for dataset_type in ['test', 'svm']:
        train_addrs = data['my_' + organ + '_' + dataset_type]
        train_labels = data['my_' +  organ + '_' + dataset_type + '_labels']
        train_filename = 'my_' + organ + '_' + dataset_type + '.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(train_filename)
        for i in range(len(train_addrs)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print 'My data: {}/{}'.format(i, len(train_addrs))
                sys.stdout.flush()
            # Load the image
            img = load_image(train_addrs[i])
            if len(img.shape) != 3 or img.shape[2] != 3:
                continue

            label = train_labels[i]
            # Create a feature
            feature = {'my/label': _int64_feature(label),
                       'my/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            
        writer.close()
        sys.stdout.flush()
'''



