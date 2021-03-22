# AIData
A simple tool to quickly inspect and convert computer vision datasets between
various formats.

Data formats include "tfrecord" (TF Examples encoded on disk into TF Record
blocks) and "pickle" which is a list of python dicts in a similar format to TF
Examples that are encoded to disk using the pickle library.

Example python dict declaration:
{
  "encoded/image": b"1234567890",
  "image/height": [200], "image/width": [400],
  "image/object/class/text": ["cat", "dog"],
  "image/object/bbox/xmin": [0.0, 0.8],
  "image/object/bbox/ymin": [0.0, 0.8],
  "image/object/bbox/xmax": [0.2, 1.0],
  "image/object/bbox/ymax": [0.2, 1.0]
}

Example usage:
python main.py pickle:dataset.pickle tfrecord:dataset.tfrecord true

