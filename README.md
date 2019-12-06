# AIData
A simple tool to quickly inspect and convert computer vision datasets between
various formats.

Data formats in memory: Python Dict, TF Example. Data formats on disk: Python
Dict, TF Example, PASCAL VOC. For Python Dict we store individual entries as
pickled data on disk. For TF Example we support TF Record, individual entries
as encoded TF Example, and individual entries as uncompressed text protos.
