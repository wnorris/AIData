import sys
from aidata import convert_python_dict_to_encoded_tf_example
from aidata import convert_encoded_tf_example_to_python_dict
from aidata import read_tf_records
from aidata import write_tf_records
from aidata import print_python_dicts
import pickle

def main():
  num_args = len(sys.argv) - 1
  if num_args != 2 and num_args != 1:
    print("Usage: main.py input output")
    sys.exit()

  encoded_tf_examples = read_tf_records([sys.argv[1]])
  output_examples = []
  python_dicts = []
  for raw_record in encoded_tf_examples:
    python_obj = convert_encoded_tf_example_to_python_dict(raw_record)
    encoded_tf_example = convert_python_dict_to_encoded_tf_example(python_obj)
    output_examples.append(encoded_tf_example)
    python_dicts.append(python_obj)

  print_python_dicts(python_dicts)

  if num_args == 2:
    f = open("{}.pkl".format(sys.argv[2]), "wb")
    pickle.dump(python_dicts, f)
    f.close()
    write_tf_records("{}.record".format(sys.argv[2]), output_examples)

if __name__ == '__main__':
  main()

