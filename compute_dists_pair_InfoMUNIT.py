import argparse
import os
import models
import numpy as np
from util import util

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use_gpu', default=True, help='turn on flag to use GPU')

# # 012_MUNIT_origin_cityscapes_64_cyc
# # parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/a2b/a2b')
# # parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/a2b/a2b/result_lpips.txt')
# parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/b2a/b2a')
# parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/012_MUNIT_origin_cityscapes_64_cyc/tests/test_batch/b2a/b2a/result_lpips.txt')

# 015_cityscapes_64_cyc
# parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/a2b/a2b')
# parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/a2b/a2b/result_lpips.txt')
# parser.add_argument('-d', '--dir', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/b2a/b2a')
# parser.add_argument('-o', '--out', type=str, default='/media/tai/6TB/Projects/InfoMUNIT/Models/ver_workshop/015_cityscapes_64_cyc/tests/test_batch/b2a/b2a/result_lpips.txt')

# # 018_MUNIT_origin_dog2catDRIT_64
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_500k/b2a/b2a/')
# parser.add_argument('-o', '--out', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_500k/b2a/b2a/result_lpips.txt')

# test 200 images for edge-datasets ----------------------------
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/tests/test_batch_200i/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/003_edge2bag_64/tests/test_batch_200i/b2a/b2a/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/tests/test_batch_200i/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/004_edge2shoe_64/tests/test_batch_200i/b2a/b2a/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/tests/test_batch_200i/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/005_MUNIT_origin_edge2bag_64/tests/test_batch_200i/b2a/b2a/')

# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/tests/test_batch_200i/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/006_MUNIT_origin_edge2shoe_64/tests/test_batch_200i/b2a/b2a/')

# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_batch_300k/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/018_MUNIT_origin_dog2catDRIT_64/tests/test_batch_300k/b2a/b2a/')

# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_300k/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/019_InfoMUNIT_dog2catDRIT_64/tests/test_batch_300k/b2a/b2a/')

# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/tests/test_batch_500k/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/020_MUNIT_portrait_64/tests/test_batch_500k/b2a/b2a/')

# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/tests/test_batch_500k/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/022_InfoMUNIT_portrait_64/tests/test_batch_500k/b2a/b2a/')

### Experiments after ECCV before WACV
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/tests/test_batch_800k/a2b/a2b/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/024_InfoMUNIT_infoLen_4/tests/test_batch_800k/b2a/b2a/')
# parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/026_InfoMUNIT_infoLen_8/tests/test_batch_800k/a2b/a2b/')
parser.add_argument('-d', '--dir', type=str, default='/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models/026_InfoMUNIT_infoLen_8/tests/test_batch_800k/b2a/b2a/')

opt = parser.parse_args()
opt.out = os.path.join(opt.dir, 'result_lpips.txt')

## Initializing the model
model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=opt.use_gpu)

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
	f.write('Mean of %s: %.3f\n' % (file_name, dist_mean))

	dist_all_files.append(dist_mean)

dist_mean_all = np.mean(np.array(dist_all_files))
print('Mean of all: %.3f' % dist_mean_all)
f.write('Mean of all: %.3f\n' % dist_mean_all)

f.close()
