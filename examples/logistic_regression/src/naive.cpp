#include "naive.hpp"
#include "data.hpp"
#include "fhe.hpp"

#include <chrono>
#include <iostream>
#include <vector>

/**
 * Approximation of the sigmoid function.
 * @param mat Matrix of data.
 */
void activation_function_naive(std::vector<double> &mat) {
    for (double & i : mat) {
        auto i3 = i * i * i;
        i *= 0.15;
        i3 *= -0.0015;
        i += 0.5;
        i += i3;
    }
}

/**
 * Accumulate the values of each row on the first column of the ciphertext matrix.
 * @param mat Matrix where to perform the accumulation.
 * @param num_columns Number of columns of the matrix.
 */
void row_accumulate(std::vector<double> &mat, const size_t num_columns) {
    for (size_t j = 1; j < num_columns; j <<= 1) {
        std::vector<double> rot (mat.size(), 0.0);
        for (size_t i = 0; i < rot.size(); i += 1) {
            rot[i] = mat[(i+j)%mat.size()];
        }
        for (size_t i = 0; i < mat.size(); i += 1) {
            mat[i] += rot[i];
        }
    }
}

/**
 * Propagate the values of the first column to the rest columns of the ciphertext matrix.
 * @param mat Matrix where to perform the propagation.
 * @param num_columns Number of columns of the matrix.
 */
void row_propagate(std::vector<double> &mat, const size_t num_columns) {
    for (size_t j = 1; j < num_columns; j <<= 1) {
        std::vector<double> rot (mat.size(), 0.0);
        for (size_t i = 0; i < rot.size(); i += 1) {
            size_t dest_idx = i - j;
            if (dest_idx >= mat.size()) { // overflow
                dest_idx = mat.size() - j + i;
            }
            rot[i] = mat[dest_idx];
        }
        for (size_t i = 0; i < mat.size(); i += 1) {
            mat[i] += rot[i];
        }
    }
}


/**
 * Accumulate the values by column. Each column ends with the same value on all rows.
 * @param mat Matrix where to perform the accumulation.
 * @param num_rows Number of rows of the matrix.
 * @param num_columns Number of columns of the matrix.
 */
void column_accumulate(std::vector<double> &mat, const size_t num_rows, const size_t num_columns) {
    for (size_t j = num_columns ; j < num_rows*num_columns; j <<= 1) {
        std::vector<double> rot (mat.size(), 0.0);
        for (size_t i = 0; i < rot.size(); i += 1) {
            rot[i] = mat[(i+j)%mat.size()];
        }
        for (size_t i = 0; i < mat.size(); i += 1) {
            mat[i] += rot[i];
        }
    }
}

/**
 * Perform an iteration of LR Training (no FHE).
 * @param data Data matrix.
 * @param results Results matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @return Iteration times.
 */
iteration_time_t logistic_regression_naive_train_iteration(const std::vector<double> &data,
                                             const std::vector<double> &results,
                                             std::vector<double> &weights,
                                             const size_t rows,
                                             const size_t columns,
                                             const size_t batch_size,
                                             const double learning_rate) {

    const auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> ct (data.size(), 0.0);

    /// Step 1. Multiply weight matrix by data matrix.
    for (size_t i = 0; i < data.size(); i += 1) {
        ct[i] = data[i]*weights[i];
    }

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_naive(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    for (size_t i = 0; i < ct.size(); i += 1) {
        if (i % columns != 0) {
            ct[i] = 0.0;
        }
    }

    /// Step 5. Compute loss (ours - expected).
    for (size_t i = 0; i < results.size(); i += 1) {
        ct[i] -= results[i];
    }

    /// Step 6. Propagation of first column value to the rest of the columns.
    row_propagate(ct, columns);

    /// Step 7. Multiply the result by the original data.
    for (size_t i = 0; i < ct.size(); i += 1) {
        ct[i] *= data[i];
    }

    /// Step 8. Compute the gradient loss across all data rows.
    column_accumulate(ct, rows, columns);

    /// Step 9. Adjust to learning rate and batch configuration.
    for (size_t i = 0; i < ct.size(); i += 1) {
        ct[i] *= (learning_rate)/static_cast<double>(batch_size);
    }

    /// Step 10. Update original weights.
    for (size_t i = 0; i < weights.size(); i += 1) {
        weights[i] -= ct[i];
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed =  std::chrono::duration_cast<time_unit_t>(end_time - start_time);
    return std::make_pair(elapsed, time_unit_t::zero());
}

std::vector<iteration_time_t> logistic_regression_naive_train(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   std::vector<double> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext,
                                   const size_t training_iterations) {
    std::vector<iteration_time_t> times (training_iterations);
    std::cout << "Doing " << training_iterations << " training iterations" << std::endl;
    for (size_t it = 0; it < training_iterations; ++it) {
        const size_t data_idx = it % data.size();
        const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
        const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
        times[it] = logistic_regression_naive_train_iteration(data[data_idx], results[data_idx], weights, rows, columns, batch_size, learning_rate);
    }
    return times;
}

/**
 * Perform an iteration of LR Inference (no FHE).
 * @param data Data matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @return Iteration times.
 */
iteration_time_t logistic_regression_naive_inference_iteration(std::vector<double> &data,
                                       const std::vector<double> &weights,
                                       const size_t rows,
                                       const size_t columns,
                                       const size_t batch_size) {

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<double> ct (data.size(), 0.0);

    /// Step 1. Multiply weight matrix by data matrix.
    for (size_t i = 0; i < data.size(); i += 1) {
        ct[i] = data[i]*weights[i];
    }

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_naive(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    for (size_t i = 0; i < ct.size(); i += 1) {
        if (i % columns != 0) {
            ct[i] = 0.0;
        }
    }

    data = ct;

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed =  std::chrono::duration_cast<time_unit_t>(end_time - start_time);
    return std::make_pair(elapsed, time_unit_t::zero());
}

std::vector<iteration_time_t> logistic_regression_naive_inference(std::vector<std::vector<double>> &data,
                                   const std::vector<double> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext) {
    std::vector<iteration_time_t> times (data.size());
    for (size_t it = 0; it < data.size(); ++it) {
        const size_t batch_size = it == data.size() - 1 ? samples_last_ciphertext : rows;
        times[it] = logistic_regression_naive_inference_iteration(data[it], weights, rows, columns, batch_size);
    }
    return times;
}

/**
 * Perform an iteration of LR Training (no FHE) (with NAG).
 * @param data Data matrix.
 * @param results Results matrix.
 * @param weights Weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @return Iteration times.
 */
iteration_time_t logistic_regression_naive_train_iteration_accelerated(const std::vector<double> &data,
                                             const std::vector<double> &results,
                                             std::vector<double> &weights,
                                             const size_t rows,
                                             const size_t columns,
                                             const size_t batch_size,
                                             const double learning_rate,
                                             const double momentum) {

    const auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<double> ct (data.size(), 0.0);

    /// Step 1. Multiply weight matrix by data matrix.
    for (size_t i = 0; i < data.size(); i += 1) {
        ct[i] = data[i]*weights[i];
    }

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_naive(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    for (size_t i = 0; i < ct.size(); i += 1) {
        if (i % columns != 0) {
            ct[i] = 0.0;
        }
    }

    /// Step 5. Compute loss (ours - expected).
    for (size_t i = 0; i < results.size(); i += 1) {
        ct[i] -= results[i];
    }

    /// Step 6. Propagation of first column value to the rest of the columns.
    row_propagate(ct, columns);

    /// Step 7. Multiply the result by the original data.
    for (size_t i = 0; i < ct.size(); i += 1) {
        ct[i] *= data[i];
    }

    /// Step 8. Compute the gradient loss across all data rows.
    column_accumulate(ct, rows, columns);

    /// Step 9. Adjust to learning rate and batch configuration.
    for (size_t i = 0; i < ct.size(); i += 1) {
        ct[i] *= (learning_rate)/static_cast<double>(batch_size);
    }

    static std::vector<double> phi (weights.size(), 0.0);
    static std::vector<double> phi_prev (weights.size(), 0.0);

    // Step 10. Calculate current momentum.
    for (size_t i = 0; i < phi.size(); i += 1) {
        phi[i] = weights[i] - ct[i];
    }
    // Step 11. Update weights based on momentum.
    for (size_t i = 0; i < phi.size(); i += 1) {
        weights[i] = phi[i] + momentum*(phi[i] - phi_prev[i]);
    }
    // Step 12. Save momentum for next iteration.
    for (size_t i = 0; i < phi.size(); i += 1) {
        phi_prev[i] = phi[i];
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed =  std::chrono::duration_cast<time_unit_t>(end_time - start_time);
    return std::make_pair(elapsed, time_unit_t::zero());
}

std::vector<iteration_time_t> logistic_regression_naive_train_accelerated(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   std::vector<double> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext,
                                   const size_t training_iterations) {

    std::vector<iteration_time_t> times (training_iterations);
    std::cout << "Doing " << training_iterations << " training iterations (NAG)" << std::endl;
    for (size_t it = 0; it < training_iterations; ++it) {
        const size_t data_idx = it % data.size();
        const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
        const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
        const double momentum = 1.0 / static_cast<double>(training_iterations);
        times[it] = logistic_regression_naive_train_iteration_accelerated(data[data_idx], results[data_idx], weights, rows, columns, batch_size, learning_rate, momentum);
    }
    return times;
}
