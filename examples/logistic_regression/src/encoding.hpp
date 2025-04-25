#ifndef ENCODING_HPP
#define ENCODING_HPP

#include <vector>
#include <tuple>

/**
 * Adapt loaded data to be FHE packed.
 * @param data Data matrix of size n * f. (n: number of data samples, f: number of features of each data sample)
 * @param data_fhe Encoded matrix adapted to the slots requirements. Each data sample (row) is padded with 0's to
 * approximate the number of features to the next power of 2. Each row of data_fhe is filled with these augmented datums
 * until num_slots is reached or there is no more data left (then it is 0 filled).
 * @param num_slots Number of slots (power of 2) we will be working with.
 * @return Tuple that contains:
 * 1º Approximated number of features for each datum.
 * 2º Number of datum encoded on each row of data_fhe.
 * 3º Number of datum encoded on the last row of data_fhe (needed if data.size() is not power of 2).
 */
std::tuple<size_t, size_t, size_t> pack_data(const std::vector<std::vector<double> > &data,
                                                 std::vector<std::vector<double>> &data_fhe,
                                                 size_t num_slots);

/**
 * Pack the results of the dataset on a FHE friendly format.
 * @param data Data to be packed.
 * @param data_fhe Destination matrix.
 * @param num_columns Number of columns of the destination matrix.
 * @return Tuple that contains:
 * 1º Approximated number of features for each datum.
 * 2º Number of datum encoded on each row of data_fhe.
 * 3º Number of datum encoded on the last row of data_fhe (needed if data.size() is not power of 2).
 */
std::tuple<size_t, size_t, size_t> pack_results(const std::vector<double> &data,
                                                std::vector<std::vector<double>> &data_fhe,
                                                size_t num_columns);

/**
 * Replicate and pack the given weights in an FHE friendly matrix mode.
 * @param weights Weights to be packed.
 * @param weights_fhe Result weight matrix.
 * @param columns Number of columns of the final matrix. (Must be equal or greater than number of weights)
 * @param rows Number of rows of the final matrix.
 * @return True if successful.
 */
bool pack_weights(const std::vector<double> &weights, std::vector<double> &weights_fhe, size_t columns, size_t rows);

/**
 * Unpacks the given data from FHE friendly format to the natural data sample per row format.
 * @param data_fhe FHE friendly vector of data matrix.
 * @param data Resulting vector of data samples.
 * @param num_rows Number of data samples per data matrix.
 * @param num_cols Number of approximated features on each data sample.
 * @param last_cipher_num_rows Number of data samples on the last matrix.
 * @param num_features Real number of features of each data sample.
 */
void unpack_data(const std::vector<std::vector<double> > &data_fhe, std::vector<std::vector<double>> &data, size_t num_rows, size_t num_cols, size_t last_cipher_num_rows, size_t num_features);

/**
 * Given the replicated weights on FHE friendly matrix format, get the generator vector of that weight matrix.
 * @param weights_fhe Weight matrix.
 * @param weights Result weight vector.
 * @param number_features Number of important weight elements.
 * @return True if successful.
 */
bool unpack_weights(const std::vector<double> &weights_fhe, std::vector<double> &weights, size_t number_features);

#endif //ENCODING_HPP
