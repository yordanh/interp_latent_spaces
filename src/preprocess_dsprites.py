#!/usr/bin/env python
"""
title           :preprocess_dsprites.py
description     :Filters and labels the dSprites dataset according to the user preferences and labels in config.json.
				:Adds color.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :05/2018
python_version  :2.7.14
==============================================================================
"""

import numpy as np
import cv2
import math
import numpy
import argparse
import os
import shutil
import itertools
import copy

from config_parser import ConfigParser

parser = argparse.ArgumentParser(description='Process the dSprited dataset.')
parser.add_argument('--image_size', default=100, type=int, help='Width and height of the square patch in px.')
parser.add_argument('--cutoff', default=10000, type=int, help='Cutoff number - max number of images per class extracted')
parser.add_argument('--labels', '-l', default="singular", help='Determined how to treat the labels for the different images')

data = np.load("/home/yordan/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
label_counters = {}

def prep_dir(folder_name):
	print("Prepring " + folder_name)
	if os.path.exists(folder_name):
		print("Cleaning " + folder_name)
		
		map(lambda object_folder : shutil.rmtree(folder_name + object_folder), os.listdir(folder_name))

		print(folder_name + " has been cleaned!")
	else:
		os.makedirs(folder_name)


# extracts images for a single label - e.g big and blue in the label group big_blue
def extract(folder_name=None, labels=None, args=None, latent_spec=None, cutoff=None, image_size=None, verbose=False):
	print("Extracting images for " + folder_name + str(labels))
	indecies = []
	for i, c in enumerate(data['latents_classes']):
		if (c[1] in latent_spec['shape'] and
		    c[2] in latent_spec['scale'] and 
		    c[3] in latent_spec['orientation'] and
		    c[4] in latent_spec['x'] and
		    c[5] in latent_spec['y']):
		    indecies.append(i)

	images = numpy.take(data['imgs'], indecies, axis=0)

	for i, image in enumerate(images):
		if i >= cutoff and cutoff != 0:
			break

		image = cv2.resize(image, (image_size, image_size))

		for bgr_color in latent_spec["color"]:
			image_out = numpy.tile(image.reshape(image_size,image_size,1), (1, 1, 3)) * bgr_color
			
			# singular labels
			if args.labels == "singular":
				for label in labels:
					cv2.imwrite(folder_name + label + "/" + str(label_counters[label]) + ".png", image_out)
					label_counters[label] += 1

			# # composite labels
			elif args.labels == "composite":
				object_folder_name = folder_name + "_".join(labels) + "/"		
				cv2.imwrite(object_folder_name + "/" + str(label_counters["_".join(labels)]) + ".png", image_out)
				label_counters["_".join(labels)] += 1

		if verbose:
			cv2.imshow("image", image_out)
			cv2.waitKey()
		
		if i % 100 == 0:
			print("{0} images have been processed so far.".format(i))
	print(label_counters)


# extracts symbols for a label group - e.g big_blue
def extract_label_groups(label_groups=None, unseen=None, latent_spec=None, mappings=None, folder_name=None, args=None):
	# build up the labels for all objects - take the combinations of the
	# lists in label_groups; color is a special case, since we add it - it is
	# not part of the given latent factors of variation
	object_labels = list(itertools.product(*[label_groups[x] for x in label_groups]))
	
	print(object_labels)
	
	# singular labels
	# extract images for each possible label combination from the given groups and 
	# export in the relevant folders
	if args.labels == "singular":
		for labels in object_labels:
			for label in labels:
				if label not in label_counters.keys():
					label_counters[label] = 0
				object_folder_name = folder_name + label + "/"
				if not os.path.exists(object_folder_name):
					os.makedirs(object_folder_name)

			revised_latent_spec = revise_latent_spec(copy.deepcopy(latent_spec), labels, mappings)
			
			if unseen != None:
				labels = filter(lambda label : label in unseen, labels)
			extract(folder_name=folder_name, labels=labels, args=args, latent_spec=revised_latent_spec, 
					image_size=args.image_size, cutoff=args.cutoff)


	# # composite labels
	# # extract images for each possible label combination from the given groups and 
	# # export in the relevant folders
	elif args.labels == "composite":
		for labels in object_labels:
			object_folder_name = folder_name + "_".join(labels) + "/"
			os.makedirs(object_folder_name)
			revised_latent_spec = revise_latent_spec(copy.deepcopy(latent_spec), labels, mappings)
			label_counters["_".join(labels)] = 0
			extract(folder_name=folder_name, labels=labels, args=args, latent_spec=revised_latent_spec, 
					image_size=args.image_size, cutoff=args.cutoff)


# revise the latent class specification, depending on the 
# given labels; we know what labels map to what classes
# across the different factors of variation
def revise_latent_spec(latent_spec, label, mappings):

	# color is special because it is added by us and if it is not a label
	# it won't be mapped to the necessary values
	# therefore we preemptively map it to an array of numeric values here 
	# and if it ends up being a label we map it again to a single value after that
	colors = latent_spec["color"]
	latent_spec["color"] = []
	for color in colors:
		latent_spec["color"] += mappings["color"][color]
	
	mappings_keys = mappings.keys()

	for key in label:
		for mkey in mappings_keys:
			if key in mappings[mkey].keys():
				new_value = mappings[mkey][key]
				if isinstance(new_value, list):
					latent_spec[mkey] = new_value
				else:
					latent_spec[mkey] = [new_value]
				break
		
	return latent_spec


if __name__ == "__main__":
	
	args = parser.parse_args()
	config_parser = ConfigParser("config/config.json")

	# in order to be able to refine out latent specs wrt to
	# user-defined labels we need to know what do these
	# labels mean wrt to the latent factors
	mappings = {}
	mappings['color'] = {'white': [numpy.array([255, 255, 255])],
						 'red' : [numpy.array([0, 64, 192])],
						 'red1' : [numpy.array([0, 64, 192])],
						 'ured' : [numpy.array([127, 0, 235]), numpy.array([12, 16, 225])],
						 'uyellow': [numpy.array([10, 240, 250]), numpy.array([10, 208, 228])],
						 'yellow' : [numpy.array([111, 217, 228])],
						 'ugreen' : [numpy.array([20, 240, 10]), numpy.array([137, 180, 62])],
						 'green' : [numpy.array([60, 107, 10])],
						 'blue': [numpy.array([200, 127, 0])],
						 'blue1' : [numpy.array([255, 127, 0])],
						 'ublue' : [numpy.array([250, 245, 13]), numpy.array([240, 21, 7])],
						 'pink': [numpy.array([255, 0, 255])]
						 }

	mappings['shape'] = {'square': 0,
				   		 'ellipse': 1,
				   		 'heart': 2}

	mappings['scale'] = {'small': [0],
						 'usmall': [0],
						 'medium': [2],
						 'umedium': [2],
				   		 'big' : [5],
				   		 'ubig' : [5]}

	mappings['orientation'] = {'rotated': [4,14,24,34],
						 	   'flat': [0,10,20,39]}

	# describes the specification wrt to which we filter the 
	# images, depending on their latent factor classes
	# the spec is refined once we are given labels
	# 
	# Given Latent Classes
	# [0] Color: white
	# [1] Shape: square, ellipse, heart
	# [2] Scale: 6 values linearly spaced in [0.5, 1]
	# [3] Orientation: 40 values in [0, 2 pi]
	# [4] Position X: 32 values in [0, 1]
	# [5] Position Y: 32 values in [0, 1]
	latent_spec = {'color': ['white', 'red', 'yellow', 'green', 'blue', 'pink'],
				   'shape': [0],#range(3),
				   'scale': [0,5],#range(6),
				   'orientation': [0,10,20,39,4,14,24,34],
				   'x': [13, 14, 15, 16, 17],
				   'y': [13, 14, 15, 16, 17]}

	# delete any previous object folders
	folder_name = "data/dSprites/"
	prep_dir(folder_name)

	specs = config_parser.parse_specs()

	extract_label_groups(label_groups=specs["train"], folder_name=folder_name+"train/", latent_spec=latent_spec, 
						 mappings=mappings, args=args)
	for unseen_spec in specs["unseen"]:
		extract_label_groups(label_groups=unseen_spec[1], unseen=unseen_spec[0], folder_name=folder_name+"unseen/", 
							 latent_spec=latent_spec, mappings=mappings, args=args)
