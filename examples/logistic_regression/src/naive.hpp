#ifndef NAIVE_HPP
#define NAIVE_HPP

#include <vector>
#include "data.hpp"

/**
 * Perform LR Training on CPU (no FHE).
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @return Vector with the times of each iteration.
 */
std::vector<iteration_time_t> logistic_regression_naive_train(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   std::vector<double> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations);

/**
 * Perform LR Inference on CPU (no FHE).
 * @param data Ciphertext of encrypted data matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @return Vector with the times of each inference.
 */
std::vector<iteration_time_t> logistic_regression_naive_inference(std::vector<std::vector<double>> &data,
                                   const std::vector<double> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext);

/**
 * Perform LR Training on CPU (no FHE) (with NAG).
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @return Vector with the times of each iteration.
 */
std::vector<iteration_time_t> logistic_regression_naive_train_accelerated(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   std::vector<double> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations);

#endif //NAIVE_HPP