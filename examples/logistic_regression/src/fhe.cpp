#include "fhe.hpp"

#include <openfhe.h>

#include "crypt.hpp"
#include "data.hpp"

/**
 * Global OpenFHE context.
 */
lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_cpu = nullptr;
/**
 * Global OpenFHE mask for first column elements of a ciphertext matrix.
 */
lbcrypto::Plaintext first_column_mask = nullptr;
/**
 * Momentum vector for accelerated gradient descent.
 */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> phi = nullptr;
/**
 * Previous momentum.
 */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> phi_prev = nullptr;

/**
* Do bootstrapping every 2 iterations
*/
bool bootstrap_every_two = false;

/**
 * Activation to be used.
 */
size_t activation_function = 1;

void create_cpu_context(bool accelerated) {

    // Parameter selection.
    constexpr uint32_t scale_mod_size = 59;
    constexpr uint32_t first_mod = 60;
    constexpr uint32_t num_large_digits = 3;

    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
    parameters.SetScalingModSize(scale_mod_size);
	parameters.SetFirstModSize(first_mod);
    parameters.SetRingDim(ring_dim);
    parameters.SetBatchSize(num_slots); 
    parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
    parameters.SetKeySwitchTechnique(lbcrypto::HYBRID);
    parameters.SetSecretKeyDist(lbcrypto::UNIFORM_TERNARY);
    parameters.SetNumLargeDigits(num_large_digits);

    // Bootstrapping parameters.
    uint32_t levelsAvailableAfterBootstrap;
    if (bootstrap_every_two) levelsAvailableAfterBootstrap = accelerated ? 18 : 16;
    else levelsAvailableAfterBootstrap = accelerated ? 9 : 8;
    const usint depth = levelsAvailableAfterBootstrap + lbcrypto::FHECKKSRNS::GetBootstrapDepth(level_budget, parameters.GetSecretKeyDist());
    parameters.SetMultiplicativeDepth(depth);

    std::cout << "Multiplicative depth of " << parameters.GetMultiplicativeDepth() << std::endl;

    // Context creation.
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(lbcrypto::FHE);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::ADVANCEDSHE);

    if (cc_cpu != nullptr) {
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalAutomorphismKeys();
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalMultKeys();
        lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>::ClearEvalSumKeys();
    }
    cc_cpu = cc;
}

void prepare_cpu_context(const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys, const size_t matrix_cols, const size_t matrix_rows) {
    // Safety check
    if (matrix_cols*matrix_rows != num_slots) {
        std::cerr << "Matrix size is different from number of slots" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Multiplication keys.
    cc_cpu->EvalMultKeyGen(keys.secretKey);
    // Rotation keys for same row value propagation and accumulation by rows and columns.
    std::vector<int> col_rot_idx;
    for (size_t j = 1; j < matrix_cols; j <<= 1) {
        col_rot_idx.push_back(static_cast<int>(j));
        col_rot_idx.push_back(-static_cast<int>(j));
    }
    for (size_t i = matrix_cols; i < matrix_cols*matrix_rows; i <<= 1) {
        col_rot_idx.push_back(static_cast<int>(i));
    }
    cc_cpu->EvalRotateKeyGen(keys.secretKey, col_rot_idx);
    // First column matrix mask creation.
    auto naive_1st_col_mask = std::vector<double>(matrix_cols*matrix_rows, 0);
    auto naive_phi_vector = std::vector<double>(matrix_cols*matrix_rows, 0);
    for (size_t i = 0; i < matrix_cols*matrix_rows; i+=matrix_cols) {
        naive_1st_col_mask[i] = 1;
    }
    first_column_mask = cc_cpu->MakeCKKSPackedPlaintext(naive_1st_col_mask);
    const auto phi_plaintext = cc_cpu->MakeCKKSPackedPlaintext(naive_phi_vector);
    phi = cc_cpu->Encrypt(phi_plaintext, keys.publicKey);
    phi_prev = cc_cpu->Encrypt(phi_plaintext, keys.secretKey);

    // Bootstrapping config.

    cc_cpu->EvalBootstrapSetup(level_budget, {0,0}, matrix_cols);
    cc_cpu->EvalBootstrapKeyGen(keys.secretKey, matrix_cols);
    cc_cpu->EvalBootstrapPrecompute(matrix_cols);

}

/**
 * Approximation of the sigmoid function.
 * @param ct Matrix of data.
 */
void activation_function_cpu(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) {
    // Ciphertexts for the variables.
    const auto ct2 = cc_cpu->EvalSquare(ct);
    auto ct3 = cc_cpu->EvalMult(ct2, ct);
    // Multiply by the coefficients.
    cc_cpu->EvalMultInPlace(ct, (0.15));
    cc_cpu->EvalMultInPlace(ct3, (-0.0015));
    // Add all together.
    cc_cpu->EvalAddInPlace(ct, 0.5);
    cc_cpu->EvalAddInPlace(ct, ct3);
}

/**
 * Accumulate the values of each row on the first column of the ciphertext matrix.
 * @param ct Ciphertext matrix where to perform the accumulation.
 * @param num_columns Number of columns of the matrix.
 */
void row_accumulate(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct, const size_t num_columns) {
    for (size_t j = 1; j < num_columns; j <<= 1) {
        auto rot = cc_cpu->EvalRotate(ct, static_cast<int>(j));
        cc_cpu->EvalAddInPlace(ct, rot);
    }
}

/**
 * Propagate the values of the first column to the rest columns of the ciphertext matrix.
 * @param ct Ciphertext matrix where to perform the propagation.
 * @param num_columns Number of columns of the matrix.
 */
void row_propagate(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct, const size_t num_columns) {
    for (size_t j = 1; j < num_columns; j <<= 1) {
        auto rot = cc_cpu->EvalRotate(ct, -static_cast<int>(j));
        cc_cpu->EvalAddInPlace(ct, rot);
    }
}

/**
 * Accumulate the values by column. Each column ends with the same value on all rows.
 * @param ct Ciphertext matrix where to perform the accumulation.
 * @param num_rows Number of rows of the matrix.
 * @param num_columns Number of columns of the matrix.
 */
void column_accumulate(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct, const size_t num_rows, const size_t num_columns) {
    for (size_t j = num_columns ; j < num_rows*num_columns; j <<= 1) {
        auto rot = cc_cpu->EvalRotate(ct, static_cast<int>(j));
        cc_cpu->EvalAddInPlace(ct, rot);
    }
}

/**
 * Perform an iteration of LR Training on CPU.
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @return Iteration time
 */
iteration_time_t logistic_regression_cpu_train_iteration(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &data,
                                             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &results,
                                             lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                             const size_t rows,
                                             const size_t columns,
                                             const size_t batch_size,
                                             const double learning_rate) {

    const auto start = std::chrono::high_resolution_clock::now();

    /// Step 1. Multiply weight matrix by data matrix.
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct = cc_cpu->EvalMult(data, weights);

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_cpu(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    ct = cc_cpu->EvalMult(ct, first_column_mask);

    /// Step 5. Compute loss (ours - expected).
    cc_cpu->EvalSubInPlace(ct, results);

    /// Step 6. Propagation of first column value to the rest of the columns.
    row_propagate(ct, columns);

    /// Step 7. Multiply the result by the original data.
    ct = cc_cpu->EvalMult(ct, data);

    /// Step 8. Compute the gradient loss across all data rows.
    column_accumulate(ct, rows, columns);

    /// Step 9. Adjust to learning rate and batch configuration.
    cc_cpu->EvalMultInPlace(ct, (learning_rate)/static_cast<double>(batch_size));

    /// Step 10. Update original weights.
    cc_cpu->EvalSubInPlace(weights, ct);

    const auto boot = std::chrono::high_resolution_clock::now();

    /// Boostrapping
    static bool do_boot = false;
    if (bootstrap_every_two) {
        if (do_boot) {
            weights->SetSlots(columns);
            weights = cc_cpu->EvalBootstrap(weights);
            weights->SetSlots(num_slots);
        }
        do_boot = !do_boot;
    }
    else {
        weights->SetSlots(columns);
        weights = cc_cpu->EvalBootstrap(weights);
        weights->SetSlots(num_slots);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_total = std::chrono::duration_cast<time_unit_t>(end - start);
    auto elapsed_boot = std::chrono::duration_cast<time_unit_t>(end - boot);
    return std::make_pair(elapsed_total, elapsed_boot);
}

std::vector<iteration_time_t> logistic_regression_cpu_train(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext,
                                   const size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {

    std::vector<iteration_time_t> times(training_iterations);
    std::cout << "Doing " << training_iterations << " training iterations" << std::endl;
    for (size_t it = 0; it < training_iterations; ++it) {
        const size_t data_idx = it % data.size();
        const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
        const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
        const auto enc_data = encrypt_data(data[data_idx], pk);
        const auto enc_results = encrypt_data(results[data_idx], pk);
        times[it] = logistic_regression_cpu_train_iteration(enc_data, enc_results, weights, rows, columns, batch_size, learning_rate);
    }
    return times;
}

/**
 * Perform an iteration of LR Inference on CPU.
 * @param data Ciphertext of encrypted data matrix Will contain the inference result.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @return Iteration time.
 */
iteration_time_t logistic_regression_cpu_inference_iteration(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &data,
                                       const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                       const size_t rows,
                                       const size_t columns,
                                       const size_t batch_size) {
    const auto start = std::chrono::high_resolution_clock::now();
    /// Step 1. Multiply weight matrix by data matrix.
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct = cc_cpu->EvalMult(data, weights);

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_cpu(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    data = cc_cpu->EvalMult(ct, first_column_mask);

    const auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_total = std::chrono::duration_cast<time_unit_t>(end - start);
    return std::make_pair(elapsed_total, time_unit_t::zero());
}

std::vector<iteration_time_t> logistic_regression_cpu_inference(std::vector<std::vector<double>> &data,
                                   const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext,
                                   const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys) {
    std::vector<iteration_time_t> times(data.size());
    for (size_t it = 0; it < data.size(); ++it) {
        const size_t batch_size = it == data.size() - 1 ? samples_last_ciphertext : rows;
        auto enc_data = encrypt_data(data[it], keys.publicKey);
        times[it] = logistic_regression_cpu_inference_iteration(enc_data, weights, rows, columns, batch_size);
        data[it] = decrypt_data(enc_data, keys.secretKey);
    }
    return times;
}

/**
 * Perform an iteration of LR Training on CPU (with NAG).
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param batch_size Number of data samples on each data matrix (typically same as rows)
 * @param learning_rate Desired learning rate for the iteration.
 * @param momentum Momentum of accelerated learning.
 * @return Iteration time.
 */
iteration_time_t logistic_regression_cpu_train_iteration_accelerated(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &data,
                                             const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &results,
                                             lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                             const size_t rows,
                                             const size_t columns,
                                             const size_t batch_size,
                                             const double learning_rate,
                                             const double momentum) {
    const auto start = std::chrono::high_resolution_clock::now();
    /// Step 1. Multiply weight matrix by data matrix.
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ct = cc_cpu->EvalMult(weights, data);

    /// Step 2. Accumulate results on the first column (inner product result).
    row_accumulate(ct, columns);

    /// Step 3. Apply the activation function.
    activation_function_cpu(ct);

    /// Step 4. Remove garbage from the ciphertext by masking the last result.
    ct = cc_cpu->EvalMult(ct, first_column_mask);

    /// Step 5. Compute loss (ours - expected).
    cc_cpu->EvalSubInPlace(ct, results);

    /// Step 6. Propagation of first column value to the rest of the columns.
    row_propagate(ct, columns);

    /// Step 7. Multiply the result by the original data.
    ct = cc_cpu->EvalMult(ct, data);

    /// Step 8. Compute the gradient loss across all data rows.
    column_accumulate(ct, rows, columns);

    /// Step 9. Adjust to learning rate and batch configuration.
    cc_cpu->EvalMultInPlace(ct, (learning_rate)/static_cast<double>(batch_size));

    // Step 10. Calculate current momentum.
    phi = cc_cpu->EvalSub(weights, ct);

    // Step 11. Update weights based on momentum.
    cc_cpu->EvalSubInPlace(phi_prev, phi);
    cc_cpu->EvalMultInPlace(phi_prev, momentum);
    weights = cc_cpu->EvalSub(phi, phi_prev);

    // Step 12. Save momentum for next iteration.
    phi_prev = phi;

    const auto boot = std::chrono::high_resolution_clock::now();

    /// Boostrapping
    static bool do_boot = false;
    if (bootstrap_every_two) {
        if (do_boot) {
            weights->SetSlots(columns);
            weights = cc_cpu->EvalBootstrap(weights);
            weights->SetSlots(num_slots);
            phi_prev->SetSlots(columns);
            phi_prev = cc_cpu->EvalBootstrap(phi_prev);
            phi_prev->SetSlots(num_slots);
        }
        do_boot = !do_boot;
    }
    else {
        weights->SetSlots(columns);
        weights = cc_cpu->EvalBootstrap(weights);
        weights->SetSlots(num_slots);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_total = std::chrono::duration_cast<time_unit_t>(end - start);
    auto elapsed_boot = std::chrono::duration_cast<time_unit_t>(end - boot);
    return std::make_pair(elapsed_total, elapsed_boot);
}

std::vector<iteration_time_t> logistic_regression_cpu_train_accelerated(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   const size_t rows,
                                   const size_t columns,
                                   const size_t samples_last_ciphertext,
                                   const size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk) {
    std::vector<iteration_time_t> times(training_iterations);
    std::cout << "Doing " << training_iterations << " training iterations (NAG)" << std::endl;
    for (size_t it = 0; it < training_iterations; ++it) {
        const size_t data_idx = it % data.size();
        const size_t batch_size = data_idx == data.size() - 1 ? samples_last_ciphertext : rows;
        const double learning_rate = 10/(static_cast<double>(it)+1) > 0.005 ? 10/(static_cast<double>(it)+1) : 0.005;
        const double momentum = 1.0 / static_cast<double>(training_iterations);
        const auto enc_data = encrypt_data(data[data_idx], pk);
        const auto enc_results = encrypt_data(results[data_idx], pk);
        times[it] = logistic_regression_cpu_train_iteration_accelerated(enc_data, enc_results, weights, rows, columns, batch_size, learning_rate, momentum);
    }
    return times;
}