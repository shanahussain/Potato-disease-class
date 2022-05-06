from PIL import Image
import imagehash
hash0 = imagehash.average_hash(Image.open(r'D:\Potato_disease\data\Healthy\0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG'))
hash1 = imagehash.average_hash(Image.open(r'D:\Potato_disease\data\Early_blight\0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG'))
cutoff = 5  # maximum bits that could be different between the hashes.

if hash0 - hash1 < cutoff:
  print('Medium')
else:
  print('High')