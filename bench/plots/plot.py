#!/usr/bin/env python3

'''
@Author: Óscar Vera López
'''
import os
import json
import pandas
import seaborn 
import matplotlib.pyplot
import matplotlib.pyplot as plt

seaborn.set(font_scale = 1.5)
seaborn.set_style("whitegrid")

###############################################################################
# 							JSON Load Functions								  #
###############################################################################

def load_json(file: str) -> dict:
	'''
	Load a JSON file and return its content

	Parameters:
		file (str): Path to the JSON file

	Returns:
		dict: JSON content
	'''

	with open(file, 'r') as f:
		data = json.load(f)

	return data

###############################################################################
# 							Auxiliary Functions								  #
###############################################################################

def pretty_print(param: dict):
	'''
	Pretty print a dictionary.

	Parameters:
		param (dict): Dictionary to print.
	'''
	return '{' + ', '.join(['{}: {}'.format(k, v) for k, v in param.items()]) + '}'

###############################################################################
# 							Data Loading Functions						  #
###############################################################################

def load_all(files: list) -> tuple[dict, pandas.DataFrame]:
	'''
	Load all provided JSON files with benchmark results. The results of all files are concatenated.

	Parameters:
		files (list): List of paths to JSON files with benchmark results.

	Returns:
		tuple: Tuple with the following lists:
			- parameters: Dictionary by fixture with the parameters used.
			- benchmarks: DataFrame with the benchmark results.
	'''

	# Process all files and append the results.
	parameters = {}
	benchmarks = pandas.DataFrame()

	for file in files:
		p, b = load(file)

		# Parameter compliancy check. All benchmarks must have the same indexed parameters.
		for key in p.keys():
			if key not in parameters:
				parameters[key] = p[key]
			else:
				for i, value in enumerate(p[key]):
					if i < len(parameters[key]):
						assert value == parameters[key][i], 'For fixture {} parameter missmatch: {} != {} at index {}'.format(key, value, parameters[key][i], i)
					else:
						parameters[key].append(value)
		
		benchmarks = pandas.concat([benchmarks, b], ignore_index=True)

	return parameters, benchmarks

def load(file: str) -> tuple[dict, pandas.DataFrame]:
	'''
	Load a JSON file with benchmark results.

	Parameters:
		file (str): Path to the JSON file with benchmark results.

	Returns:
		tuple: Tuple with the following lists:
			- parameters: Dictionary by fixture with the parameters used.
			- benchmarks: DataFrame with the benchmark results.
	'''

	# Load JSON file
	data = load_json(file)
	parameters, name, fixture, fixture_param, iterations, real_time, limbs, batch, ntt = load_raw(data)

	# Need to create a list with the Platform used for each benchmark.
	Platform = [file.split('/')[-1].split('.')[0]] * len(name)

	# Create a DataFrame to store the data.
	benchmarks = pandas.DataFrame({
		'name': name,
		'fixture': fixture,
		'fixture_param': fixture_param,
		'iterations': iterations,
		'real_time': real_time,
		'used_limbs': limbs,
		'batch': batch,
		'ntt': ntt,
		'Platform': Platform
	})

	return parameters, benchmarks

def load_raw(data: dict) -> tuple:
	'''
	Load raw JSON data.
	
	Parameters:
		data (dict): JSON content loaded from a benchmark results file.

	
	Expected JSON file format:

		{
			"context": 
			{
				"<name>Fixture:<index>": "<param_name>: <value>, ...",
				...
			}
			"benchmarks": 
			[
				{
					# Mandatory fields
					"name": "fixture_name/benchmark_name/fixture_param/...",
					"real_time": float,
					"iterations": int,
					"time_unit": str,
					# Optional fields
					"p_limbs": value,
					"p_batch": value,
					"p_ntt": value,
					# Other optional fields not used but useful
					"skipped": bool, 
				},
				...
			],
			
		}

	Returns:
		tuple: Tuple with the following dictionary and lists:
			- parameters: Dictionary by fixture with the parameters used.
			- name: List of names of the benchmark.
			- fixture: List of fixtures attached to each benchmark.
			- fixture_param: List of fixture parameters used.
			- iterations: List of number of iterations.
			- real_time: List of real times in nanoseconds.
			- limbs: List of number of limbs used.
			- batch: List of batch sizes used.
			- ntt: List of booleans indicating if Number-theoretic transform is used.
	'''

	# Every results file has a context and benchmarks.
	context = data['context']
	benchmarks = data['benchmarks']

	# Process context and benchmarks.
	parameters = load_context(context)
	name, fixture, fixture_param, iterations, real_time, limbs, batch, ntt = load_benchmarks(benchmarks)

	return parameters, name, fixture, fixture_param, iterations, real_time, limbs, batch, ntt

def load_context(context: dict) -> dict:
	'''
	Load context data with a more manageable format.
	
	Parameters:
		context (dict): Context data dictionary from a benchmark results file.
	
	Returns:
		dict: Processed context data. Format:
			{ 
				'<name>Fixture': [
					{ '<param_name>': '<value>', ... },
					{ '<param_name>': '<value>', ... },
					...
				],
				...
			}
	'''

	# Dictorionary to store the parameters used on each fixture.
	parameters = {}

	for key in context.keys():
		# Search for keys named '<name>Fixture:<index>'.
		if 'Fixture:' in key:
			name, index = key.split(':')
			# Each key maps to a string with the format 'key1: value1, key2: value2, ...'.
			raw_value = context[key].split(',')
			value = {}
			for v in raw_value:
				k, v = v.split(':')
				value[k.strip()] = v.strip()
			# Insert the parameters in the correct index and fixture name.
			if name not in parameters: parameters[name] = []
			parameters[name].insert(int(index), value)
	
	return parameters

def load_benchmarks(benchmarks: dict) -> tuple:
	'''
	Load benchmark data with a more manageable format.

	Parameters:
		benchmarks (dict): Benchmark data dictionary from a benchmark results file.
	
	Returns:
		tuple: Tuple with the following lists consistent by index:
			- name: Name of the benchmark.
			- fixture: Fixture attached to the benchmark.
			- fixture_param: Fixture parameters.
			- iterations: Number of iterations.
			- real_time: Real time in nanoseconds.
			- limbs: Number of limbs used.
			- batch: Batch size.
			- ntt: If Number-theoretic transform is used.
	'''

	# General properties.
	name = []
	fixture = []
	fixture_param = []
	iterations = []
	real_time = []
	# Specific properties.
	limbs = []
	batch = []
	ntt = []

	for benchmark in benchmarks:

		# Check if skipped.
		if 'skipped' in benchmark:
			continue

		# General properties.
		name.append(benchmark['name'].split('/')[1])
		fixture.append(benchmark['name'].split('/')[0])
		fixture_param.append(benchmark['name'].split('/')[2])
		iterations.append(benchmark['iterations'])

		# Time normalization to ns.
		if benchmark['time_unit'] == 'ns':
			real_time.append(benchmark['real_time'])
		elif benchmark['time_unit'] == 'us':
			real_time.append(benchmark['real_time'] * 1e3)
		elif benchmark['time_unit'] == 'ms':
			real_time.append(benchmark['real_time'] * 1e6)
		elif benchmark['time_unit'] == 's':
			real_time.append(benchmark['real_time'] * 1e9)
		
		# Specific properties.
		if 'p_limbs' in benchmark:
			limbs.append(benchmark['p_limbs'])
		else:
			limbs.append(None)
		if 'p_batch' in benchmark:
			batch.append(benchmark['p_batch'])
		else:
			batch.append(None)
		if 'p_ntt' in benchmark:
			ntt_used = True
			if benchmark['p_ntt'] == 0:
				ntt_used = False
			ntt.append(ntt_used)
		else:
			ntt.append(None)

	return name, fixture, fixture_param, iterations, real_time, limbs, batch, ntt

###############################################################################
# 							Data Processing Functions						  #
###############################################################################

def partition_data(data: pandas.DataFrame) -> dict:
	'''
	Partition the data by benchmark.

	Parameters:
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		dict: Dictionary with the data partitioned by fixture.
	'''

	partitioned_data = {}

	for name in data['name'].unique():
		partitioned_data[name] = clear_irrelevant_data(data[data['name'] == name])

	return partitioned_data

def clear_irrelevant_data(data: pandas.DataFrame) -> pandas.DataFrame:

	'''
	Clear irrelevant data from the benchmark result DataFrame.

	Parameters:
		data (DataFrame): DataFrame with a single benchmark results.

	Returns:
		DataFrame: DataFrame with the irrelevant data removed.
	'''

	# Drop all columns with all None values.
	data = data.dropna(axis=1, how='all')
	
	cols = ['iterations', 'real_time', 'Platform', 'fixture', 'fixture_param']
	# For ContextCreation benchmarks retain only relevant columns.
	if 'ContextCreation' in data['name'].unique():
		data = data.drop(columns=[col for col in data.columns if col not in cols])
	# Drop all columns with all equal values except from fixture, fixture_param, Platform and real_time.
	data = data.drop(columns=[col for col in data.columns if col not in cols and data[col].nunique() == 1])

	return data

def derive_data(params: dict, data: pandas.DataFrame) -> pandas.DataFrame:
	'''
	Derive new data from the provided data.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		DataFrame: DataFrame with the derived data.
	'''

	# Compute elements processed per second.
	data = compute_used_limbs(params, data)
	data = compute_processed_limbs(params, data)
	data = compute_elements_per_second(params, data)
	data = compute_slots_per_second(params, data)

	return data

def compute_processed_limbs(params: dict, data: pandas.DataFrame) -> pandas.DataFrame:

	'''
	Compute the number of limbs processed on benchmarks using the limb count from the parameters.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		DataFrame: DataFrame updated with the number of limbs processed.
	'''

	fideslib_fixture_params = params['FIDESlibFixture']
	general_fixture_params = params['GeneralFixture']
	for index, row in data.iterrows():
		if row['fixture'] == 'GeneralFixture':
			data.at[index, 'processed_limbs'] = int(params['GeneralFixture'][int(row['fixture_param'])]['multDepth']) + 1 - row['used_limbs']
		if row['fixture'] == 'FIDESlibFixture':
			# NOTE: FIDESlib fixtures outputs processed limbs as a used limbs.
			#data.at[index, 'processed_limbs'] = int(params['FIDESlibFixture'][int(row['fixture_param'])]['L']) + 1 - row['used_limbs']
			data.at[index, 'processed_limbs'] = row['used_limbs']
	return data

def compute_elements_per_second(params: dict, data: pandas.DataFrame) -> pandas.DataFrame:
	'''
	Compute the elements processed per second for each benchmark.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		DataFrame: DataFrame updated with the elements processed per second.
	'''

	# For the FIDESlibFixture fixture, the number of elements is 2^logN * used_limbs.
	fideslib_fixture_params = params['FIDESlibFixture']
	# For the GeneralFixture fixture, the number of elements is ringDim * used_limbs.
	general_fixture_params = params['GeneralFixture']
	for index, row in data.iterrows():
		if row['fixture'] == 'GeneralFixture':
			data.at[index, 'elements'] = row['processed_limbs'] * int(params['GeneralFixture'][int(row['fixture_param'])]['ringDim'])
		if row['fixture'] == 'FIDESlibFixture':
			data.at[index, 'elements'] = row['processed_limbs'] * (2 ** int(params['FIDESlibFixture'][int(row['fixture_param'])]['logN']))
	
	data['elements_per_second'] = data['elements'] / (data['real_time'] / 1e9)

	return data

def compute_slots_per_second(params: dict, data: pandas.DataFrame) -> pandas.DataFrame:
	'''
	Compute the slots processed per second for each benchmark.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		DataFrame: DataFrame updated with the slots processed per second.
	'''

	# For the FIDESlibFixture fixture, the number of slots is 2^logN / 2.
	fideslib_fixture_params = params['FIDESlibFixture']
	N = [2 ** int(p['logN']) / 2 for p in fideslib_fixture_params]
	slots_list = data.loc[data['fixture'] == 'FIDESlibFixture','fixture_param'].apply(lambda x: N[int(x)])
	data['slots'] = slots_list

	# For the GeneralFixture fixture, the number of slots is ringDim / 2.
	general_fixture_params = params['GeneralFixture']
	for index, row in data.iterrows():
		if row['fixture'] == 'GeneralFixture':
			data.at[index, 'slots'] = int(params['GeneralFixture'][int(row['fixture_param'])]['ringDim']) / 2

	data['slots_per_second'] = data['slots'] / (data['real_time'] / 1e9)
	data['time_per_slot'] = (data['real_time']) / data['slots']

	return data

def compute_used_limbs(params: dict, data: pandas.DataFrame) -> pandas.DataFrame:
	'''

	Compute the number of limbs used on benchmarks using the limb count from the parameters.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		data (DataFrame): DataFrame with the benchmark results.

	Returns:
		DataFrame: DataFrame updated with the number of limbs used.
	'''

	for index, row in data.iterrows():
		if row['fixture'] == 'FIDESlibFixture' and float(row['used_limbs']).is_integer() == False:
			limbs = int(params['FIDESlibFixture'][int(row['fixture_param'])]['L'])+1
			data.at[index, 'used_limbs'] = limbs

	data['name'] = data.apply(lambda x: x['name'].replace('ContextLimbCount', ''), axis=1)	

	return data


###############################################################################
# 							Plotting Functions								  #
###############################################################################

def plot_elements_per_second(params: dict, bench: str, data: pandas.DataFrame):
	'''
	Plot the elements processed per second for each benchmark.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		bench (str): Name of the benchmark.
		data (DataFrame): DataFrame with the benchmark results.
	'''

	if 'elements_per_second' not in data.columns:
		return
	if 'processed_limbs' not in data.columns:
		return

	figure_size = (10, 6)

	for index, fixture_param in enumerate(data['fixture_param'].unique()):
		data_subset = data.copy()[data['fixture_param'] == fixture_param]

		if 'ntt' in data_subset.columns:
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Plot the mean elements per second.
			seaborn.lineplot(data=data_subset, x='processed_limbs', y='elements_per_second', hue='Platform', style='ntt', markers=True, estimator='mean', errorbar=('ci', 0), ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Elements Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/elements_per_second/mean-ntt', exist_ok=True)
			fig.savefig('figures/elements_per_second/mean-ntt/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

		if 'batch' in data_subset.columns:
			# Mean elements per second with best and worst batch size lines.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Find best and worst batch size for each used limb. Work on a copy of the data.
			best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['elements_per_second'].idxmax()
			best_batch_data = data_subset.copy().loc[best_batch]
			best_batch_data['type'] = 'best_batch'
			worst_batch = data_subset.groupby(['processed_limbs', 'Platform'])['elements_per_second'].idxmin()
			worst_batch_data = data_subset.copy().loc[worst_batch]
			worst_batch_data['type'] = 'worst_batch'
			tmp = pandas.concat([best_batch_data, worst_batch_data])
			# Concatenate the data with the best and worst batch size.
			plot_data = data_subset.copy()
			plot_data['type'] = 'mean'
			plot_data = pandas.concat([plot_data, tmp])
			seaborn.lineplot(data=plot_data, x='processed_limbs', y='elements_per_second', hue='Platform', style='type', markers=True, estimator='mean', errorbar=('ci', 0), ax=ax, linewidth=2, style_order=['best_batch', 'mean', 'worst_batch'],hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Elements Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/elements_per_second/limits-batch', exist_ok=True)
			fig.savefig('figures/elements_per_second/limits-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

			# Mean elements per second with confidence interval.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Plot the mean elements per second with a 95% confidence interval.
			seaborn.lineplot(data=data_subset, x='processed_limbs', y='elements_per_second', hue='Platform', estimator='mean',markers=True, errorbar=('ci', 95), ax=ax, style='Platform',hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Elements Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/elements_per_second/mean-batch', exist_ok=True)
			fig.savefig('figures/elements_per_second/mean-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

			# Best batch size for each limb count and Platform.
			best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['elements_per_second'].idxmax()
			best_batch_data = data_subset.copy().loc[best_batch]
			# For each Platform, compute most repeated best batch size.
			best_batch_data = best_batch_data.groupby(['Platform', 'batch']).size().reset_index(name='count')
			# Get only data that has the maximum count.
			best_batch_data = best_batch_data.loc[best_batch_data.groupby('Platform')['count'].idxmax()]
			# From the data, filter for the best batch size.
			best_batch_data = best_batch_data[['Platform', 'batch']]
			# Filter data for the pairs of Platform and batch size.
			best_batch_data = data_subset.merge(best_batch_data, on=['Platform', 'batch'])
			# Plot the best batch size.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			seaborn.lineplot(data=best_batch_data, x='processed_limbs', y='elements_per_second', hue='Platform', markers=True, style='Platform', ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Elements Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/elements_per_second/best-batch', exist_ok=True)
			fig.savefig('figures/elements_per_second/best-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')	

		fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
		# Plot the mean elements per second.
		seaborn.lineplot(data=data_subset, x='processed_limbs', y='elements_per_second', hue='Platform', estimator='mean',markers=True, ax=ax, errorbar=('ci', 0), linewidth=2, style='Platform',hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
		# Set the title and labels.

		ax.set_xlabel('Procesed Limbs')
		ax.set_ylabel('Elements Processed per Second')
		ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
		ax.minorticks_on()
		ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
		# Save the figure.
		os.makedirs('figures/elements_per_second/mean-all', exist_ok=True)
		fig.savefig('figures/elements_per_second/mean-all/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
		matplotlib.pyplot.close('all')

	return

def plot_slots_per_second(params: dict, bench: str, data: pandas.DataFrame):
	'''
	Plot the slots processed per second for each benchmark.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		bench (str): Name of the benchmark.
		data (DataFrame): DataFrame with the benchmark results.
	'''

	if 'slots_per_second' not in data.columns:
		return
	if 'processed_limbs' not in data.columns:
		return
	
	figure_size = (10, 6)

	for index, fixture_param in enumerate(data['fixture_param'].unique()):
		data_subset = data.copy()[data['fixture_param'] == fixture_param]

		if 'ntt' in data_subset.columns:
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Plot the mean slots per second.
			seaborn.lineplot(data=data_subset, x='processed_limbs', y='slots_per_second', hue='Platform', style='ntt', markers=True, estimator='mean', errorbar=('ci', 0), ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Slots Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/slots_per_second/mean-ntt', exist_ok=True)
			fig.savefig('figures/slots_per_second/mean-ntt/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

		if 'batch' in data_subset.columns:
			# Mean slots per second with best and worst batch size lines.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Find best and worst batch size for each used limb. Work on a copy of the data.
			best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['slots_per_second'].idxmax()
			best_batch_data = data_subset.copy().loc[best_batch]
			best_batch_data['type'] = 'best_batch'
			worst_batch = data_subset.groupby(['processed_limbs', 'Platform'])['slots_per_second'].idxmin()
			worst_batch_data = data_subset.copy().loc[worst_batch]
			worst_batch_data['type'] = 'worst_batch'
			tmp = pandas.concat([best_batch_data, worst_batch_data])
			# Concatenate the data with the best and worst batch size.
			plot_data = data_subset.copy()
			plot_data['type'] = 'mean'
			plot_data = pandas.concat([plot_data, tmp])
			seaborn.lineplot(data=plot_data, x='processed_limbs', y='slots_per_second', hue='Platform', style='type', markers=True, estimator='mean', errorbar=('ci', 0), ax=ax, linewidth=2, style_order=['best_batch', 'mean', 'worst_batch'],hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Slots Processed per Second')
			ax.set_yscale('log')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/slots_per_second/limits-batch', exist_ok=True)
			fig.savefig('figures/slots_per_second/limits-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

			# Mean slots per second with confidence interval.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Plot the mean slots per second with a 95% confidence interval. Also plot points.
			seaborn.lineplot(data=data_subset, x='processed_limbs', y='slots_per_second', hue='Platform', markers = True, estimator='mean', errorbar=('ci', 95), ax=ax, style='Platform',hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Slots Processed per Second')
			ax.set_yscale('log')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/slots_per_second/mean-batch', exist_ok=True)
			fig.savefig('figures/slots_per_second/mean-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')

			# Best batch size for each limb count and Platform.
			best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['slots_per_second'].idxmax()
			best_batch_data = data_subset.copy().loc[best_batch]
			# For each Platform, compute most repeated best batch size.
			best_batch_data = best_batch_data.groupby(['Platform', 'batch']).size().reset_index(name='count')
			# Get only data that has the maximum count.
			best_batch_data = best_batch_data.loc[best_batch_data.groupby('Platform')['count'].idxmax()]
			# From the data, filter for the best batch size.
			best_batch_data = best_batch_data[['Platform', 'batch']]
			# Filter data for the pairs of Platform and batch size.
			best_batch_data = data_subset.merge(best_batch_data, on=['Platform', 'batch'])
			# Plot the best batch size.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			seaborn.lineplot(data=best_batch_data, x='processed_limbs', y='slots_per_second', hue='Platform', markers=True, style='Platform', ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.

			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Slots Processed per Second')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/slots_per_second/best-batch', exist_ok=True)
			fig.savefig('figures/slots_per_second/best-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')	
				
			# Best batch size for each limb count and Platform.
			best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['time_per_slot'].idxmin()
			best_batch_data = data_subset.copy().loc[best_batch]
			# For each Platform, compute most repeated best batch size.
			best_batch_data = best_batch_data.groupby(['Platform', 'batch']).size().reset_index(name='count')
			# Get only data that has the maximum count.
			best_batch_data = best_batch_data.loc[best_batch_data.groupby('Platform')['count'].idxmax()]
			# From the data, filter for the best batch size.
			best_batch_data = best_batch_data[['Platform', 'batch']]
			# Filter data for the pairs of Platform and batch size.
			best_batch_data = data_subset.merge(best_batch_data, on=['Platform', 'batch'])
			# Plot the best batch size.
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			seaborn.lineplot(data=best_batch_data, x='processed_limbs', y='time_per_slot', hue='Platform', markers=True, style='Platform', ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
			# Set the title and labels.
			ax.set_xlabel('Processed Limbs')
			ax.set_ylabel('Time per processed slot (ns)')
			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/time_per_slot/best-batch', exist_ok=True)
			fig.savefig('figures/time_per_slot/best-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
			matplotlib.pyplot.close('all')	

		fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
		# Plot the mean slots per second.
		seaborn.lineplot(data=data_subset, x='processed_limbs', y='slots_per_second', hue='Platform', estimator='mean', markers=True, ax=ax, errorbar=('ci', 0), linewidth=2, style='Platform',hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
		# Set the title and labels.

		ax.set_xlabel('Processed Limbs')
		ax.set_ylabel('Slots Processed per Second')
		ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
		ax.minorticks_on()
		ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
		# Save the figure.
		os.makedirs('figures/slots_per_second/mean-all', exist_ok=True)
		fig.savefig('figures/slots_per_second/mean-all/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
		matplotlib.pyplot.close('all')

	return

def plot_time_performance(params: dict, bench: str, data: pandas.DataFrame):
	'''
	Plot the time performance for each benchmark.

	Parameters:
		params (dict): Dictionary with the parameters used in the benchmarks.
		bench (str): Name of the benchmark.
		data (DataFrame): DataFrame with the benchmark results.
	'''


	if 'real_time' not in data.columns:
		return

	figure_size = (10, 6)

	if 'FIDESlibFixture' == data['fixture'].unique()[0]:

		data_copy = data.copy()
		data_copy['real_time'] = data_copy['real_time'] / 1e3
		
		# Plot mean time by limb count.
		if 'processed_limbs' in data_copy.columns:
			for fixture_param in data_copy['fixture_param'].unique():
				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				# Plot the mean time by limb count.
				seaborn.lineplot(data=data_subset, x='processed_limbs', y='real_time', hue='Platform', estimator='mean', ax=ax, errorbar=('ci', 0), linewidth=2, style='Platform', markers=True,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
				# Add grid.
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
				# Set the title and labels.

				ax.set_xlabel('Processed Limbs')
				ax.set_ylabel('Time (us)')
				# Save the figure.
				os.makedirs('figures/time_performance/mean-limbs', exist_ok=True)
				fig.savefig('figures/time_performance/mean-limbs/{}-{}-{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
				matplotlib.pyplot.close('all')

		# Plot the mean time by batch size.
		if 'batch' in data_copy.columns:
			for fixture_param in data_copy['fixture_param'].unique():
				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				# Plot the mean time by batch size.
				seaborn.lineplot(data=data_subset, x='batch', y='real_time', hue='Platform', estimator='mean', ax=ax, errorbar=('ci', 0), linewidth=2, style='Platform', markers=True,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
				# Set the title and labels.

				ax.set_xlabel('Batch Size')
				ax.set_ylabel('Time (us)')
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
				# Save the figure.
				os.makedirs('figures/time_performance/mean-batch', exist_ok=True)
				fig.savefig('figures/time_performance/mean-batch/{}-{}-{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
				matplotlib.pyplot.close('all')


		if 'batch' in data_copy.columns and 'processed_limbs' in data_copy.columns:
			for fixture_param in data_copy['fixture_param'].unique():
				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				# Best batch size for each limb count and Platform.
				best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['real_time'].idxmin()
				best_batch_data = data_subset.copy().loc[best_batch]
				# For each Platform, compute most repeated best batch size.
				best_batch_data = best_batch_data.groupby(['Platform', 'batch']).size().reset_index(name='count')
				# Get only data that has the maximum count.
				best_batch_data = best_batch_data.loc[best_batch_data.groupby('Platform')['count'].idxmax()]
				# From the data, filter for the best batch size.
				best_batch_data = best_batch_data[['Platform', 'batch']]
				# Filter data for the pairs of Platform and batch size.
				best_batch_data = data_subset.merge(best_batch_data, on=['Platform', 'batch'])
				# Plot the best batch size.
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				seaborn.lineplot(data=best_batch_data, x='processed_limbs', y='real_time', hue='Platform', markers=True, style='Platform', ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
				# Set the title and labels.

				ax.set_xlabel('Processed Limbs')
				ax.set_ylabel('Time (us)')
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='gray')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
				# Save the figure.
				os.makedirs('figures/time_performance/best-batch', exist_ok=True)
				fig.savefig('figures/time_performance/best-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
				matplotlib.pyplot.close('all')	

		# Plot mean time for all fixture parameters and Platforms.
		data_copy['fixture_param'] = data_copy['fixture_param'].apply(lambda x: pretty_print(params['FIDESlibFixture'][int(x)]))
		fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
		# Plot the mean time
		seaborn.barplot(data=data_copy, x='fixture_param', y='real_time', hue='Platform', legend=True, ax=ax)
		# Set the title and labels.s

		ax.set_xlabel('Platform')
		ax.set_ylabel('Time (us)')
		ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
		ax.minorticks_on()
		ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
		# Save the figure.
		os.makedirs('figures/time_performance/mean-all', exist_ok=True)
		fig.savefig('figures/time_performance/mean-all/{}-{}.pdf'.format(bench, data['fixture'].unique()[0]))
		matplotlib.pyplot.close('all')

	elif 'GeneralFixture' == data['fixture'].unique()[0]:

		data_copy = data.copy()
		data_copy['real_time'] = data_copy['real_time'] / 1e3

		# Plot mean time by limb count.
		if 'processed_limbs' in data_copy.columns:
			for fixture_param in data_copy['fixture_param'].unique():
				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				# Plot the mean time by limb count.
				seaborn.lineplot(data=data_subset, x='processed_limbs', y='real_time', hue='Platform', estimator='mean', ax=ax, errorbar=('ci', 0), linewidth=2, style='Platform', markers=True,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
				# Set the title and labels.

				ax.set_xlabel('Processed Limbs')
				ax.set_ylabel('Time (us)')
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
				# Save the figure.
				os.makedirs('figures/time_performance/mean-limbs', exist_ok=True)
				fig.savefig('figures/time_performance/mean-limbs/{}-{}-{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
				matplotlib.pyplot.close('all')

		if 'batch' in data_copy.columns and 'processed_limbs' in data_copy.columns:
			for fixture_param in data_copy['fixture_param'].unique():
				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				data_subset['max'] = data_subset.groupby(['batch', 'Platform'])['processed_limbs'].transform('max')
				data_subset = data_subset[data_subset['processed_limbs'] == data_subset['max']]
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				# Plot the mean time by batch size.
				seaborn.lineplot(data=data_subset, x='batch', y='real_time', hue='Platform', estimator='mean', ax=ax,
								 errorbar=('ci', 0), linewidth=2, style='Platform', markers=True,
								 hue_order=['4060Ti', 'A4500', 'V100', '4090'], dashes=False)
				# Set the title and labels.
				ax.set_xlabel('Batch Size')
				ax.set_ylabel('Time (us)')
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='gray')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
				# Save the figure.
				os.makedirs('figures/time_performance/mean-batch', exist_ok=True)
				fig.savefig(
					'figures/time_performance/mean-batch/{}-{}-{}.pdf'.format(bench, data_subset['fixture'].unique()[0],
																			  fixture_param))
				matplotlib.pyplot.close('all')

				data_subset = data_copy.copy()[data_copy['fixture_param'] == fixture_param]
				# Best batch size for each limb count and Platform.
				best_batch = data_subset.groupby(['processed_limbs', 'Platform'])['real_time'].idxmin()
				best_batch_data = data_subset.copy().loc[best_batch]
				# For each Platform, compute most repeated best batch size.
				best_batch_data = best_batch_data.groupby(['Platform', 'batch']).size().reset_index(name='count')
				# Get only data that has the maximum count.
				best_batch_data = best_batch_data.loc[best_batch_data.groupby('Platform')['count'].idxmax()]
				# From the data, filter for the best batch size.
				best_batch_data = best_batch_data[['Platform', 'batch']]
				# Filter data for the pairs of Platform and batch size.
				best_batch_data = data_subset.merge(best_batch_data, on=['Platform', 'batch'])
				# Plot the best batch size.
				fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
				seaborn.lineplot(data=best_batch_data, x='processed_limbs', y='real_time', hue='Platform', markers=True, style='Platform', ax=ax, linewidth=2,hue_order=['4060Ti','A4500','V100','4090'],dashes=False)
				# Set the title and labels.

				ax.set_xlabel('Processed Limbs')
				ax.set_ylabel('Time (us)')
				ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='gray')
				ax.minorticks_on()
				ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
				# Save the figure.
				os.makedirs('figures/time_performance/best-batch', exist_ok=True)
				fig.savefig('figures/time_performance/best-batch/{}-{}-param{}.pdf'.format(bench, data_subset['fixture'].unique()[0], fixture_param))
				matplotlib.pyplot.close('all')

			data_subset = data_copy.copy()
			data_subset['max'] = data_subset.groupby(['fixture_param', 'Platform'])['processed_limbs'].transform('max')
			data_subset = data_subset[data_subset['processed_limbs'] == data_subset['max']]
			f = data_subset.groupby(['fixture_param', 'Platform'])['real_time'].idxmin()
			data_subset = data_subset.loc[f]
			fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
			# Plot the mean time
			seaborn.barplot(data=data_subset, x='fixture_param', y='real_time', hue='Platform', legend=True, ax=ax,
							errorbar=('ci', 95), hue_order=['4060Ti','A4500','V100','4090'])
			# Set the title and labels.
			ax.set_xlabel('Parameter set')
			ax.set_ylabel('Time (us)')
			ax.set_yscale('log')
			lista = [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000]
			ax.set_yticks(lista, [f"{x}" for x in lista])

			ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
			ax.minorticks_on()
			ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
			# Save the figure.
			os.makedirs('figures/time_performance/mean-0-level-param', exist_ok=True)
			fig.savefig('figures/time_performance/mean-0-level-param/{}-{}.pdf'.format(bench, data['fixture'].unique()[0]))
			matplotlib.pyplot.close('all')

		
		# Plot mean time for all fixture parameters and Platforms.
		#data_copy['fixture_param'] = data_copy['fixture_param'].apply(lambda x: pretty_print(params['GeneralFixture'][int(x)]))
		fig, ax = matplotlib.pyplot.subplots(figsize=figure_size)
		# Plot the mean time
		seaborn.barplot(data=data_copy, x='fixture_param', y='real_time', hue='Platform', legend=True, ax=ax, errorbar=('ci', 95))
		# Set the title and labels.s

		ax.set_xlabel('Parameter set')
		ax.set_ylabel('Time (us)')
		ax.grid(True, which='both', linestyle='--', linewidth='0.5', color='black')
		ax.minorticks_on()
		ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
		
		# Save the figure.
		os.makedirs('figures/time_performance/mean-all', exist_ok=True)
		fig.savefig('figures/time_performance/mean-all/{}-{}.pdf'.format(bench, data['fixture'].unique()[0]))
		matplotlib.pyplot.close('all')

	return


###############################################################################
# 								Main Function								  #
###############################################################################

def main():
	'''
	Main function
	'''

	# Obtain absolute path to all JSON files in the data directory.
	files = [os.path.abspath(os.path.join('data', f)) for f in os.listdir('data') if f.endswith('.json')]
	p, b = load_all(files)

	# Filter data.

	b = derive_data(p, b)
	bs = partition_data(b)

	# Plot data.
	for key in bs.keys():
		#plot_elements_per_second(p, key, bs[key])
		#plot_slots_per_second(p, key, bs[key])
		plot_time_performance(p, key, bs[key])
	return

if __name__ == '__main__':
	main()