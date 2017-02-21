import os
import glob
import processOneImage as pImg
import scipy.io as sio

dataDir = '/Users/xikangzhang/research/data/ucf_sports_actions'
outputDir = './ucf_pose'

seq = [s for s in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir,s))]
print(seq)
if not os.path.isdir(outputDir):
	os.mkdir(outputDir)
for s in seq:
	subseq = [ss for ss in os.listdir(os.path.join(dataDir, s)) if os.path.isdir(os.path.join(dataDir, s, ss))]
	if not os.path.isdir(os.path.join(outputDir, s)):
		os.mkdir(os.path.join(outputDir, s))
	for ss in subseq:
		print (ss)
		if not os.path.isdir(os.path.join(outputDir, s, ss)):
			os.mkdir(os.path.join(outputDir, s, ss))
# 		files = [f for f in os.listdir(os.path.join(dataDir, s, ss)) if os.path.isfile(os.path.join(dataDir, s, ss, f))]
		files = glob.glob(os.path.join(dataDir, s, ss,'*.jpg'))
# 		print(files)
		for f in files:
			prediction = pImg.processOneImage(f)
			print(prediction)
			sio.savemat(os.path.join(outputDir, s, ss, os.path.basename(f)), {'prediction':prediction})