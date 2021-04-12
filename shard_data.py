import pickle
import os
from statistics import mean
import math
import sys

HOSTS = 10
BYTES_PER_MB = 1000000
MIN_SHARD_SIZE = 100 * BYTES_PER_MB

def main():
  num_args = len(sys.argv) - 1
  if num_args != 1:
    print("Usage: shard_data.py input")
    sys.exit()

  input_path = sys.argv[1]
  with open(input_path, "rb") as f:
      dataset = pickle.load(f)
    
  dataset_size = os.path.getsize(input_path)
  average_record_size = dataset_size / len(dataset)
    
  print("Dataset size in MB: {}".format(dataset_size / BYTES_PER_MB))
  print("Number of records: {}".format(len(dataset)))
  print("Average record size: {} MB".format(average_record_size / BYTES_PER_MB))
    
  min_records_per_shard = math.ceil(MIN_SHARD_SIZE / average_record_size)
  max_shards = math.floor(len(dataset) / min_records_per_shard)
    
  ideal_number_of_shards = 10*HOSTS
   
  num_shards = min(max_shards, ideal_number_of_shards)
    
  if num_shards < ideal_number_of_shards:
      print("Limited to minimum shard size.")
  else:
      print("Not limited to minimum shard size.")
    
  print("Number of shards: {}".format(num_shards))
  print("Estimated size per shard: {} MB".format((len(dataset) / num_shards) * (average_record_size / BYTES_PER_MB)))
    
  records_per_shard = math.ceil(len(dataset)/num_shards)
  for i in range(num_shards):
      pickle.dump(dataset[i*records_per_shard:(i+1)*records_per_shard],
              open("{}-{}-of-{}".format(input_path, i, num_shards), "wb"))

if __name__ == '__main__':
  main()

