import sys
from aidata import convert_python_dict_to_encoded_tf_example
from aidata import convert_encoded_tf_example_to_python_dict
from aidata import read_tf_records
from aidata import write_tf_records
from aidata import read_pickled_python_dicts
from aidata import write_pickled_python_dicts
from aidata import print_python_dicts
import pickle

def main():
  num_args = len(sys.argv) - 1
  if num_args != 3 and num_args != 2 and num_args != 1:
    print("Usage: main.py input [output] [pretty_print]")
    sys.exit()

  full_python_dicts = []
  input_splits = sys.argv[1].split(",")
  for input_split in input_splits:
    if len(input_split.split(":", 1)) != 2:
      print("Type specifier required before input.")
      sys.exit()
    
    input_type = input_split.split(":", 1)[0]
    input_paths = input_split.split(":", 1)[1].split(",")  
    if input_type == "tfrecord":
      encoded_tf_examples = read_tf_records(input_paths)
      python_dicts = list(map(convert_encoded_tf_example_to_python_dict, encoded_tf_examples))
      full_python_dicts = full_python_dicts + python_dicts
    elif input_type == "pickle":
      python_dicts = read_pickled_python_dicts(input_paths)
      full_python_dicts = full_python_dicts + python_dicts
    else:
      print("Unsupported input type: {}".format(input_type))
      sys.exit()

  if num_args < 2:
    sys.exit()

  if len(sys.argv[2].split(":", 1)) != 2:
    print("Type specifier required before output.")
    sys.exit()
  output_type = sys.argv[2].split(":", 1)[0]
  output_path = sys.argv[2].split(":", 1)[1]
  if output_type == "tfrecord":
    encoded_tf_examples = list(map(convert_python_dict_to_encoded_tf_example, full_python_dicts))
    write_tf_records(output_path, encoded_tf_examples)
  elif output_type == "pickle":
    write_pickled_python_dicts(output_path, full_python_dicts)
  else:
    print("Unsupported output type: {}".format(output_type))
    sys.exit()

  if num_args < 3:
    sys.exit()

  print_python_dicts(python_dicts)

if __name__ == '__main__':
  main()

