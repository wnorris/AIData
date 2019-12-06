#!/usr/bin/env python
import unittest
import os
from aidata import convert_python_dict_to_encoded_tf_example
from aidata import convert_encoded_tf_example_to_python_dict
from aidata import read_tf_records
from aidata import write_tf_records


class TestAIData(unittest.TestCase):
    #def setUp(self):

    #def tearDown(self):

    def generate_test_example(self):
        python_dict = {
          "image/height": [200], "image/width": [400],
          "image/object/class/text": ["cat", "dog"],
          "image/object/bbox/xmin": [0.0, 0.8],
          "image/object/bbox/ymin": [0.0, 0.8],
          "image/object/bbox/xmax": [0.2, 1.0],
          "image/object/bbox/ymax": [0.2, 1.0]
        }
        return python_dict

    def verify_python_dicts_equal(self, python_dict, output_dict):
        self.assertEqual(python_dict["image/height"][0], output_dict["image/height"][0])
        self.assertEqual(python_dict["image/width"][0], output_dict["image/width"][0])
        self.assertEqual(python_dict["image/object/class/text"][0],
            output_dict["image/object/class/text"][0].decode())
        self.assertEqual(python_dict["image/object/class/text"][1],
            output_dict["image/object/class/text"][1].decode())
        keys = ["image/object/bbox/xmin", "image/object/bbox/xmax", "image/object/bbox/ymin", "image/object/bbox/ymax"]
        for key in keys:
            self.assertAlmostEqual(python_dict[key][0],
                output_dict[key][0])
            self.assertAlmostEqual(python_dict[key][1],
                output_dict[key][1])

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

if __name__ == '__main__':
    unittest.main(verbosity=2)
