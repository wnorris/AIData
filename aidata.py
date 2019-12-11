import tensorflow as tf
import pickle
import os
import io
import PIL
import hashlib
from lxml import etree

tf.enable_eager_execution()

def _bytes_list_feature(values):
  """Returns a bytes_list from a string / byte."""
  for i in range(len(values)):
    if isinstance(values[i], type(tf.constant(0))):
      values[i] = values[i].numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_list_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_list_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def convert_python_dict_to_encoded_tf_example(python_dict):
  temp_python_dict_of_features = {}
  for k, v in python_dict.items():
    if len(v) <= 0:
      print("Error: Each key provided in python dictionary must have at least one value. The following key has none: {}".format(k))
    else:
      if isinstance(v[0], str):
        temp_python_dict_of_features[k] = _bytes_list_feature(list(map(lambda x: x.encode(), v)))
      elif isinstance(v[0], bytes) or isinstance(v[0], bytearray) or isinstance(v[0], memoryview):
        temp_python_dict_of_features[k] = _bytes_list_feature(v)
      elif isinstance(v[0], float) or isinstance(v[0], complex):
        temp_python_dict_of_features[k] = _float_list_feature(v)
      elif isinstance(v[0], bool) or isinstance(v[0], int):
        temp_python_dict_of_features[k] = _int64_list_feature(v)
      else:
        print("Error: The following key has an entry with an unsupported type. Key: {}, Type: {}, Value: {}".format(k, type(v[0]), v[0]))
  example_proto = tf.train.Example(features=tf.train.Features(feature=temp_python_dict_of_features))
  return example_proto.SerializeToString()

def convert_encoded_tf_example_to_python_dict(encoded_tf_example):
  output_dict = {}
  example = tf.train.Example()
  example.ParseFromString(encoded_tf_example)
  for key in example.features.feature:
    if example.features.feature[key].HasField("bytes_list"):
      output_dict[key] = list(example.features.feature[key].bytes_list.value)
    if example.features.feature[key].HasField("float_list"):
      output_dict[key] = list(example.features.feature[key].float_list.value)
    if example.features.feature[key].HasField("int64_list"):
      output_dict[key] = list(example.features.feature[key].int64_list.value)
  return output_dict

def read_tf_records(filepaths):
  raw_dataset = tf.data.TFRecordDataset(filepaths)
  records = raw_dataset.take(-1)
  return list(map(lambda x: x.numpy(), records))

def write_tf_records(filepath, tf_examples):
  writer = tf.python_io.TFRecordWriter(filepath)
  for tf_example in tf_examples:
    writer.write(tf_example)
  writer.close()

def read_pickled_python_dicts(filepaths):
  if not isinstance(filepaths, list):
    filepaths = [filepaths]
  full_python_dict_array = []
  for filepath in filepaths:
    f = open(filepath, "rb")
    python_dict_array = pickle.load(f)
    f.close()
    full_python_dict_array = full_python_dict_array + python_dict_array
  return full_python_dict_array

def write_pickled_python_dicts(filepath, python_dict_array):
  f = open(filepath, "wb")
  pickle.dump(python_dict_array, f)
  f.close()

def read_pascal_voc(filepath="testdata/VOCdevkit/VOC2012/", setname="trainval"):
  python_dict_array = []

  examples_path = os.path.join(filepath, 'ImageSets', 'Main',
                               'aeroplane_' + setname + '.txt')
  annotations_dir = os.path.join(filepath, "Annotations")
  with tf.gfile.GFile(examples_path) as fid:
    lines = fid.readlines()
  examples_list = [line.strip().split(' ')[0] for line in lines]
  for idx, example in enumerate(examples_list):
    #if idx % 100 == 0:
    #  print('On image {} of {}'.format(idx, len(examples_list)))
    path = os.path.join(annotations_dir, example + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = _recursive_parse_xml_to_dict(xml)['annotation']

    with tf.gfile.GFile(os.path.join(filepath, "JPEGImages", data['filename']), 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
      raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    if 'object' in data:
      for obj in data['object']:
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        #classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    python_obj = {
      'image/height': [height],
      'image/width': [width],
      'image/filename': [data['filename'].encode('utf8')],
      'image/source_id': [data['filename'].encode('utf8')],
      'image/key/sha256': [key.encode('utf8')],
      'image/encoded': [encoded_jpg],
      'image/format': ['jpeg'.encode('utf8')],
      'image/object/bbox/xmin': xmin,
      'image/object/bbox/xmax': xmax,
      'image/object/bbox/ymin': ymin,
      'image/object/bbox/ymax': ymax,
      'image/object/class/text': classes_text,
      #'image/object/class/label': classes,
      'image/object/truncated': truncated,
      'image/object/view': poses
    }

    python_dict_array.append(python_obj)
  return python_dict_array

def _recursive_parse_xml_to_dict(xml):
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = _recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def write_pascal_voc(filepath, python_dict_array):
  return

def print_python_dicts(python_dicts):
  if not isinstance(python_dicts, list):
    python_dicts = [python_dicts]
  for python_obj in python_dicts:
    for k, v in python_obj.items():
      print_type = str(type(v[0]))
      print_value = list(map(lambda x: (str(x)[:30] + '...') if len(str(x)) > 30 else str(x), v)) 
      print("{} [{}]: {}".format(k, print_type, print_value))

