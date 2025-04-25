#ifndef DATA_HPP
#define DATA_HPP

#include <chrono>
#include <string>
#include <vector>

#include "helper.hpp"

/**
 * Load a CSV file into memory.
 * @param filename CSV file path.
 * @param data Matrix where the data will be stored.
 * @return True if successful.
 * @note Each row of the matrix maps to a row in the CSV.
 */
bool load_csv(const std::string &filename, std::vector<std::vector<std::string> > &data);

/**
 * Categorization and normalization for any data. Parse data to a LR friendly format.
 * @param raw_data Matrix of raw data to be parsed. Last column is expected to be the expected result column.
 * @param data Matrix of parsed data. Each original raw data matrix row is parsed into a row in this matrix.
 * @param results Parsed results for each data sample.
 * @param result_index Index of the column with the result label.
 * @return Tuple that contains: 1º Number of features of each datum. 2º Number of data samples.
 */
std::tuple<size_t, size_t> parse_data(const std::vector<std::vector<std::string> > &raw_data,
                std::vector<std::vector<double> > &data,
                std::vector<double> &results,
                size_t result_index);


/**
 * Weight generation. Generate weights for the quantity of features specified.
 * @param num_features Number of features of the data samples.
 * @param weights Vector of zero initialized weights generated to be used by the model to train on the given data.
 */
void generate_weights(size_t num_features, std::vector<double> &weights);

/**
 * Check if a data matrix and a result matrix have the LR correct format.
 * @param data Data matrix.
 * @param results Result matrix.
 * @return True if successful.
 */
bool check_data(const std::vector<std::vector<double> > &data, const std::vector<double> &results);

/**
 * Save a weight vector on a file.
 * @param filename File path.
 * @param weights Weight vector to be saved.
 */
void save_weights(const std::string &filename, const std::vector<double> &weights);

/**
 * Load weights from a file.
 * @param filename File path.
 * @param num_features Number of features.
 * @param weights Weight vector destination.
 */
void load_weights(const std::string &filename, size_t num_features, std::vector<double> &weights);

/**
 * Alias the unit of time we are using.
 */
typedef std::chrono::microseconds time_unit_t;
/**
 * Alias for a pair of durations to measure total and bootstrapping times.
 */
typedef std::pair<time_unit_t, time_unit_t> iteration_time_t;

/**
 * Print times to a file.
 * @param path File path.
 * @param times Times.
 * @param iterations Iterations.
 * @param accelerated Used accelerated learning.
 * @param fun_activation Id of activation function.
 * @param accuracy Obtained accuracy.
 * @param exec Execution type.
 */
void print_times (const std::string &path, std::vector<iteration_time_t> &times, size_t iterations, bool accelerated, size_t fun_activation, double accuracy, exec_t exec);

#endif // DATA_HPP
