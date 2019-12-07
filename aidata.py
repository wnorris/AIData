import tensorflow as tf
import pickle

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

