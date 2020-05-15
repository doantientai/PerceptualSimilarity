import argparse
import os
import models
import numpy as np
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use_gpu', default=True, help='turn on flag to use GPU')

# 012_MUNIT_origin_cityscapes_64_cyc
# parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/a2b/a2b')
# parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/a2b/a2b/result_lpips.txt')
parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/b2a/b2a')
parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/b2a/b2a/result_lpips.txt')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
f = open(opt.out,'w')

# files = os.listdir(opt.dir)
subdirs = [x for x in os.listdir(opt.dir) if x[0] == "_"]
# subdirs.remove("results.txt")

first_subdir = subdirs[0]
file_names = os.listdir(os.path.join(opt.dir, first_subdir))

# exit()
dist_all_files = []
for file_name in file_names:
	print(file_name)
	dists = []
	for (ff, subdir0) in enumerate(subdirs[:-1]):
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir, subdir0, file_name))) # RGB image from [-1,1]
		if opt.use_gpu:
			img0 = img0.cuda()

		for (gg, subdir1) in enumerate(subdirs[ff + 1:]):
			img1 = util.im2tensor(util.load_image(os.path.join(opt.dir, subdir1, file_name)))
			if opt.use_gpu:
				img1 = img1.cuda()

			# Compute distance
			dist01 = model.forward(img0, img1).item()
			dists.append(dist01)
			# print('(%s, %s): %.3f' % (subdir0, subdir1, dist01))
			# f.writelines('(%s, %s): %.3f' % (subdir0, subdir1, dist01))

	dist_mean = np.mean(np.array(dists))
	print('Mean of %s: %.3f' % (file_name, dist_mean))
	f.writelines('Mean of %s: %.3f' % (file_name, dist_mean))

	dist_all_files.append(dist_mean)

dist_mean_all = np.mean(np.array(dist_all_files))
print('Mean of all: %.3f' % dist_mean)
f.writelines('Mean of all: %.3f' % dist_mean)

f.close()
