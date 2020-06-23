import argparse
import os
import models
import numpy as np
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/a2b')
# parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/a2b/result_lpips.txt')
parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/b2a')
parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/SOTAsDemos/DRIT/DRIT/outputs/001_portrait/b2a/result_lpips.txt')


parser.add_argument('--use_gpu', default=True, help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=opt.use_gpu)

subdirs = [os.path.join(opt.dir, x) for x in os.listdir(opt.dir) if os.path.isdir(os.path.join(opt.dir, x))]
subdirs.sort()
# print(subdirs)
# exit()

# crawl directories
f = open(opt.out, 'w')
dist_all_files = []

for subdir in subdirs:
    files = [x for x in os.listdir(subdir) if x[:6] == 'output']
    dists = []
    for (ff, file0) in enumerate(files[:-1]):
        img0 = util.im2tensor(util.load_image(os.path.join(subdir, file0)))  # RGB image from [-1,1]
        if opt.use_gpu:
            img0 = img0.cuda()

        for (gg, file1) in enumerate(files[ff + 1:]):
            img1 = util.im2tensor(util.load_image(os.path.join(subdir, file1)))
            if opt.use_gpu:
                img1 = img1.cuda()

            # Compute distance
            dist01 = model.forward(img0, img1).item()
            dists.append(dist01)
            # print('(%s, %s): %.3f' % (file0, file1, dist01))
            # f.writelines('(%s, %s): %.3f' % (file0, file1, dist01))

    dist_mean = np.mean(np.array(dists))
    print('%s: %.3f' % (subdir, dist_mean))
    # f.writelines('Mean: %.3f' % dist_mean)

    dist_all_files.append(dist_mean)

dist_mean_all = np.mean(np.array(dist_all_files))
print('Mean of all: %.3f' % dist_mean_all)
f.write('Mean of all: %.3f\n' % dist_mean_all)
f.close()
