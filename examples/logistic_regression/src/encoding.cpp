#include "encoding.hpp"

#include <bit>
#include <iostream>

#include "fhe.hpp"

std::tuple<size_t, size_t, size_t> pack_data(const std::vector<std::vector<double>> &data,
											 std::vector<std::vector<double>> &data_fhe, const size_t num_slots) {

	const size_t num_data_samples = data.size();
	const size_t num_features = data[0].size();

	const size_t num_features_fhe = std::bit_ceil(num_features);

	if (num_features_fhe > num_slots) {
		std::cerr << "Error: num_features_fhe > num_slots" << std::endl;
		exit(EXIT_FAILURE);
	}

	const size_t data_samples_per_ciphertext = num_slots / num_features_fhe;
	const size_t ciphertext_count = (num_data_samples + data_samples_per_ciphertext - 1) / data_samples_per_ciphertext;


	size_t last_ciphertext_datum_count = 0;

	data_fhe.clear();
	data_fhe.resize(ciphertext_count);

	for (size_t ciphertext_idx = 0; ciphertext_idx < ciphertext_count; ++ciphertext_idx) {
		data_fhe[ciphertext_idx].reserve(num_slots);
		last_ciphertext_datum_count = 0;
		for (size_t datum_idx = 0; datum_idx < data_samples_per_ciphertext; ++datum_idx) {
			if (ciphertext_idx * data_samples_per_ciphertext + datum_idx >= num_data_samples) {
				data_fhe[ciphertext_idx].resize(num_slots);
				break;
			};
			auto datum = data[ciphertext_idx * data_samples_per_ciphertext + datum_idx];
			datum.resize(num_features_fhe, 0.0);
			data_fhe[ciphertext_idx].insert(data_fhe[ciphertext_idx].end(), datum.begin(), datum.end());
			last_ciphertext_datum_count++;
		}
	}

	return std::make_tuple(num_features_fhe, data_samples_per_ciphertext, last_ciphertext_datum_count);
}

std::tuple<size_t, size_t, size_t> pack_results(const std::vector<double> &data,
												std::vector<std::vector<double>> &data_fhe, const size_t num_columns) {

	const size_t num_data_samples = data.size();
	const size_t num_features = num_columns;

	const size_t num_features_fhe = std::bit_ceil(num_features);
	const size_t data_samples_per_ciphertext = num_slots / num_features_fhe;
	const size_t ciphertext_count = (num_data_samples + data_samples_per_ciphertext - 1) / data_samples_per_ciphertext;

	size_t last_ciphertext_datum_count = 0;

	data_fhe.clear();
	data_fhe.resize(ciphertext_count);

	for (size_t ciphertext_idx = 0; ciphertext_idx < ciphertext_count; ++ciphertext_idx) {
		data_fhe[ciphertext_idx].resize(num_slots, 0.0);
		last_ciphertext_datum_count = 0;
		for (size_t datum_idx = 0; datum_idx < data_samples_per_ciphertext; ++datum_idx) {
			if (ciphertext_idx * data_samples_per_ciphertext + datum_idx >= num_data_samples) {
				data_fhe[ciphertext_idx].resize(num_slots);
				break;
			};
			const auto datum = data[ciphertext_idx * data_samples_per_ciphertext + datum_idx];
			data_fhe[ciphertext_idx][datum_idx * num_columns] = datum;
			last_ciphertext_datum_count++;
		}
	}

	return std::make_tuple(num_features_fhe, data_samples_per_ciphertext, last_ciphertext_datum_count);
}

bool pack_weights(const std::vector<double> &weights, std::vector<double> &weights_fhe, const size_t columns,
				  const size_t rows) {

	if (weights.size() > columns) {
		return false;
	}

	weights_fhe.clear();
	weights_fhe.resize(rows * columns, 0.0);

	for (size_t idx = 0; idx < weights.size(); ++idx) {
		for (size_t col_idx = idx; col_idx < columns * rows; col_idx += columns) {
			weights_fhe[col_idx] = weights[idx];
		}
	}

	return true;
}

void unpack_data(const std::vector<std::vector<double>> &data_fhe, std::vector<std::vector<double>> &data,
				 const size_t num_rows, const size_t num_cols, const size_t last_cipher_num_rows,
				 const size_t num_features) {

	data.clear();

	for (size_t matrix_idx = 0; matrix_idx < data_fhe.size(); ++matrix_idx) {
		for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
			if (matrix_idx == data_fhe.size() - 1 && row_idx > last_cipher_num_rows - 1) {
				break;
			}
			std::vector<double> row(num_features, 0.0);
			for (size_t col_idx = 0; col_idx < num_features; ++col_idx) {
				row[col_idx] = data_fhe[matrix_idx][row_idx * num_cols + col_idx];
			}
			data.push_back(row);
		}
	}
}

bool unpack_weights(const std::vector<double> &weights_fhe, std::vector<double> &weights, size_t number_features) {
	weights.clear();
	weights.resize(number_features, 0.0);
	for (size_t idx = 0; idx < number_features; ++idx) {
		weights[idx] = weights_fhe[idx];
	}

	return true;
}
