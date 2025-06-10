#ifndef FHE_HPP
#define FHE_HPP

#include <openfhe.h>

#include "data.hpp"

/**
 * Bootstrapping level budget.
 */
const std::vector<uint32_t> level_budget = {2, 2};
/**
 * Ring dimension used.
 */
constexpr uint32_t ring_dim = 1 << 16;
/**
 * Number of slots used.
 */
constexpr uint32_t num_slots = ring_dim / 2;
/**
 * Global OpenFHE context.
 */
extern lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_cpu;
/**
 * Momentum vector for accelerated gradient descent.
 */
extern lbcrypto::Ciphertext<lbcrypto::DCRTPoly> phi;
/**
 * Previous momentum.
 */
extern lbcrypto::Ciphertext<lbcrypto::DCRTPoly> phi_prev;

extern lbcrypto::Plaintext first_column_mask;
extern lbcrypto::Plaintext first_column_mask_0;
extern lbcrypto::Plaintext first_column_mask_1;
extern lbcrypto::Plaintext first_column_mask_3;

/**
* Do bootstrapping every 2 iterations
*/
extern bool bootstrap_every_two;

/**
 * Activation to be used.
 */
extern size_t activation_function;

/**
 * Create a global OpenFHE context.
 * @param accelerated Use accelerated learning.
 * @param infeence Is doing inference.
 */
void create_cpu_context(bool accelerated, bool inference);

/**
 * Prepare the global OpenFHE context for a specific LR workload.
 * @param keys Key pair.
 * @param matrix_cols Number of columns of each data sample on a ciphertext. Needed for rotation index key generation.
 * @param matrix_rows Number of data samples (rows) on the ciphertext matrix.
 */
void prepare_cpu_context(const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys, size_t matrix_cols, size_t matrix_rows);

/**
 * Perform LR Training on CPU.
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @param pk Public key.
 * @return Iteration times.
 */
std::vector<iteration_time_t> logistic_regression_cpu_train(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

/**
 * Perform LR Inference on CPU.
 * @param data Data matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param keys Keypair.
 * @return Iteration times.
 */
std::vector<iteration_time_t> logistic_regression_cpu_inference(std::vector<std::vector<double>> &data,
                                   const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys);

/**
 * Perform LR Training on CPU (with NAG).
 * @param data Ciphertext of encrypted data matrix.
 * @param results Ciphertext of encrypted results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @param pk Public key.
 * @return Iteration times.
 */
std::vector<iteration_time_t> logistic_regression_cpu_train_accelerated(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

#endif //FHE_HPP
