import pickle
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
with open('sample_000000000000.data.pickle', 'rb') as f:
    data = pickle.load(f)

def imbytes2arr(b):
    return np.array(Image.open(io.BytesIO(b)))

step = data['steps'][0]
print("Instruction:", step['observation']['natural_language_instruction'].decode())

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
titles = ['image', 'hand_image', 'image_with_depth']
keys   = ['image', 'hand_image', 'image_with_depth']
for ax, t, k in zip(axs, titles, keys):
    img = imbytes2arr(step['observation'][k])
    ax.imshow(img)
    ax.set_title(t)
    ax.axis('off')
plt.tight_layout()
plt.savefig('step0_views.png', dpi=120)   # 保存到文件
print('Saved -> step0_views.png')