import numpy as np
from tensorboardX import SummaryWriter
# 以后的模板
writer = SummaryWriter(comment='base_scalar')
for epoch in range(100):
    writer.add_scalar('scalar/test', np.random.rand(), epoch)
    writer.add_scalars('scalar/scalars_test', {'xsinx':epoch * np.sin(epoch), 'xcosx':epoch * np.cos(epoch)}, epoch)

writer.close()