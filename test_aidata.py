#!/usr/bin/env python
import unittest
import os
from aidata import convert_python_dict_to_encoded_tf_example
from aidata import convert_encoded_tf_example_to_python_dict
from aidata import read_tf_records
from aidata import write_tf_records
from aidata import read_pickled_python_dicts
from aidata import write_pickled_python_dicts
from aidata import read_pascal_voc


class TestAIData(unittest.TestCase):

    def generate_test_example(self):
        python_dict = {
          "encoded/image": b"1234567890",
          "image/height": [200], "image/width": [400],
          "image/object/class/text": ["cat", "dog"],
          "image/object/bbox/xmin": [0.0, 0.8],
          "image/object/bbox/ymin": [0.0, 0.8],
          "image/object/bbox/xmax": [0.2, 1.0],
          "image/object/bbox/ymax": [0.2, 1.0]
        }
        return python_dict

    def verify_bytes_or_string_equal(self, a, b):
        if isinstance(a, str):
            a = str.encode(a)
        if isinstance(b, str):
            b = str.encode(b)
        self.assertEqual(a, b)

    def verify_python_dicts_equal(self, python_dict, output_dict):
        self.assertEqual(python_dict["image/height"][0], output_dict["image/height"][0])
        self.assertEqual(python_dict["image/width"][0], output_dict["image/width"][0])
        for i in range(len(output_dict["image/object/class/text"])):
          self.verify_bytes_or_string_equal(python_dict["image/object/class/text"][i],
              output_dict["image/object/class/text"][i])
        keys = ["image/object/bbox/xmin", "image/object/bbox/xmax", "image/object/bbox/ymin", "image/object/bbox/ymax"]
        for key in keys:
            for i in range(len(output_dict[key])):
                self.assertAlmostEqual(python_dict[key][i],
                    output_dict[key][i])

    def test_dict_to_example(self):
        python_dict = self.generate_test_example()
        tf_example = convert_python_dict_to_encoded_tf_example(python_dict)
        output_dict = convert_encoded_tf_example_to_python_dict(tf_example)
        self.verify_python_dicts_equal(python_dict, output_dict)

    def test_tf_record_read_and_write(self):
        python_dict = self.generate_test_example()
        tf_example = convert_python_dict_to_encoded_tf_example(python_dict)
        write_tf_records("testdata/tmp_output.record", [tf_example, tf_example, tf_example])
        output_tf_examples = read_tf_records("testdata/tmp_output.record")
        self.assertEqual([tf_example, tf_example, tf_example], output_tf_examples)
        output_dict = convert_encoded_tf_example_to_python_dict(output_tf_examples[0])
        self.verify_python_dicts_equal(python_dict, output_dict)
        os.remove("testdata/tmp_output.record")

    def test_pickled_python_dict_read_and_write(self):
        python_dict = self.generate_test_example()
        write_pickled_python_dicts("testdata/tmp_output.pkl", [python_dict, python_dict, python_dict])
        output_python_dicts = read_pickled_python_dicts(["testdata/tmp_output.pkl"])
        self.assertEqual([python_dict, python_dict, python_dict], output_python_dicts)
        self.verify_python_dicts_equal(python_dict, output_python_dicts[0])
        os.remove("testdata/tmp_output.pkl")

    def test_read_pascal_voc_format(self):
        python_dicts = read_pickled_python_dicts(["testdata/pascal_voc_three_record_sample.pickle"])
        output_python_dicts = read_pascal_voc("testdata/VOCdevkit/VOC2012/")
        self.verify_python_dicts_equal(python_dicts[0], output_python_dicts[0])
        self.verify_python_dicts_equal(python_dicts[1], output_python_dicts[1])
        self.verify_python_dicts_equal(python_dicts[2], output_python_dicts[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
