import json

class ConfigParser(object):
	def __init__(self, filename):
		file = open(filename, "r")
		self.config = json.load(file)

	def parse_specs(self):
		specs = {'train' : [], 'unseen' : []}

		specs['train'] = self.config['data_generation']['train']['spec']
		for record in self.config['data_generation']['unseen']:
			specs['unseen'].append((record['label'], record['spec']))

		return specs

	def parse_labels(self):
		labels = {'train' : [], 'unseen' : []}
		
		labels['train'] = self.config["labels"]["train"]
		labels['unseen'] = self.config["labels"]["unseen"]
		
		return labels

	def parse_groups(self):
		groups = self.config["groups"]

		return groups