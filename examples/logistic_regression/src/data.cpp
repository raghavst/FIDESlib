#include "data.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <ranges>
#include <vector>

#include "helper.hpp"

bool load_csv(const std::string &filename, std::vector<std::vector<std::string>> &data) {

	data.clear();

	std::ifstream file(filename);
	if (!file.is_open()) {
		return false;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::vector<std::string> row;
		std::string cell;
		for (char c: line) {
			if (c == ',') {
				row.push_back(cell);
				cell.clear();
			} else {
				cell += c;
			}
		}
		row.push_back(cell);
		data.push_back(row);
	}

	return true;
}

std::tuple<size_t, size_t> parse_data(const std::vector<std::vector<std::string>> &raw_data,
									  std::vector<std::vector<double>> &data, std::vector<double> &results,
									  const size_t result_index) {
	data.clear();
	results.clear();

	// Now process the raw data to convert the categorical columns to numerical columns.
	for (const auto &raw_row: raw_data) {
		std::vector<double> row;
		for (size_t j = 0; j < raw_row.size(); ++j) {
			if (j == result_index) {
				double res = std::stod(raw_row[j]);
				results.push_back(res);
			} else {
				row.push_back(std::stod(raw_row[j]));
			}
		}
		data.push_back(row);
	}

	return std::make_tuple(data[0].size(), data.size());
}

void generate_weights(const size_t num_features, std::vector<double> &weights) {
	weights.clear();
	weights.resize(num_features);

	for (size_t i = 0; i < num_features; ++i) {
		weights[i] = 0;
	}

	std::cout << "Generated weights for " << num_features << " features" << std::endl;
}

bool check_data(const std::vector<std::vector<double>> &data, const std::vector<double> &results) {

	if (data.size() != results.size()) {
		return false;
	}
	return true;
}

void save_weights(const std::string &filename, const std::vector<double> &weights) {
	std::ofstream file(filename, std::ofstream::out | std::ofstream::trunc);
	if (!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < weights.size(); ++i) {
		file << weights[i];
		if (i != weights.size() - 1) {
			file << ",";
		}
	}
}

void load_weights(const std::string &filename, const size_t num_features, std::vector<double> &weights) {
	weights.clear();

	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string line;
	std::getline(file, line);
	std::vector<std::string> row;
	std::string cell;
	for (char c: line) {
		if (c == ',') {
			row.push_back(cell);
			cell.clear();
		} else {
			cell += c;
		}
	}
	row.push_back(cell);

	for (const auto &i: row) {
		weights.push_back(std::stod(i));
	}

	if (weights.size() != num_features) {
		std::cerr << "Incorrect number of weights" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void print_times(const std::string &path, std::vector<iteration_time_t> &times, size_t iterations, bool accelerated,
				 size_t fun_activation, double accuracy, const exec_t exec) {

	std::ofstream file(path, std::ios_base::app);

	for (auto &[fst, snd]: times) {
		std::cout << iterations << "," << fst.count() << "," << snd.count() << "," << fun_activation << ","
				  << accelerated;
		if (exec == VALIDATION) {
			std::cout << "," << accuracy;
		}
		std::cout << std::endl;
	}

	file.close();
}
