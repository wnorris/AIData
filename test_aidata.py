#!/usr/bin/env python
import unittest
from aidata import convert_python_dict_to_encoded_tf_example
from aidata import convert_encoded_tf_example_to_python_dict
from aidata import read_tf_records
from aidata import write_tf_records


class TestAIData(unittest.TestCase):
    #def setUp(self):

    #def tearDown(self):

    def test_dict_to_example(self):
        python_dict = {
          "image/height": [200], "image/width": [400],
          "image/object/class/text": ["cat", "dog"],
          "image/object/bbox/xmin": [0.0, 0.8],
          "image/object/bbox/ymin": [0.0, 0.8],
          "image/object/bbox/xmax": [0.2, 1.0],
          "image/object/bbox/ymax": [0.2, 1.0]
        }
        tf_example = convert_python_dict_to_encoded_tf_example(python_dict)
        output_dict = convert_encoded_tf_example_to_python_dict(tf_example)
        self.assertEqual(python_dict["image/height"][0], output_dict["image/height"][0])
        self.assertEqual(python_dict["image/width"][0], output_dict["image/width"][0])
        self.assertEqual(python_dict["image/object/class/text"][0],
            output_dict["image/object/class/text"][0].decode())
        self.assertEqual(python_dict["image/object/class/text"][1],
            output_dict["image/object/class/text"][1].decode())
        keys = ["image/object/bbox/xmin", "image/object/bbox/xmax", "image/object/bbox/ymin", "image/object/bbox/ymax"]
        for key in keys:
            self.assertEqual(python_dict[key][0],
                round(output_dict[key][0], 4))
            self.assertEqual(python_dict[key][1],
                round(output_dict[key][1], 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
