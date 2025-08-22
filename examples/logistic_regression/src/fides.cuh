#ifndef FIDES_CUH
#define FIDES_CUH

#include "data.hpp"

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Parameters.cuh"

extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;
extern FIDESlib::CKKS::Parameters params;

/**
 * Prepare a FIDESlib context for a specific LR workload.
 * @param cc_gpu FIDESlib context.
 * @param keys Key pair.
 * @param matrix_cols Number of columns of each data sample on a ciphertext. Needed for rotation index key generation.
 * @param matrix_rows Number of data samples (rows) on the ciphertext matrix.
 * @param boot
 */
void prepare_gpu_context(FIDESlib::CKKS::Context &cc_gpu, const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys,
						 size_t matrix_cols, size_t matrix_rows, bool boot);

/**
 * Move a ciphertext to the GPU.
 * @param cc_gpu FIDESlib context.
 * @param ct Ciphertext to be moved.
 * @return FIDESlib ciphertext on the GPU.
 */
FIDESlib::CKKS::Ciphertext move_ciphertext(FIDESlib::CKKS::Context &cc_gpu, const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct);

/**
 * Move a ciphertexts to the GPU.
 * @param cc_gpu FIDESlib context.
 * @param cts Ciphertexts to be moved.
 * @return FIDESlib ciphertexts on the GPU.
 */
std::vector<FIDESlib::CKKS::Ciphertext> move_ciphertext(FIDESlib::CKKS::Context &cc_gpu, const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &cts);

/**
 * Move back a ciphertext from the GPU.
 * @param cc_gpu FIDESlib context.
 * @param res Destination ciphertext.
 * @param ct Fideslib ciphertext.
 */
void move_back(const FIDESlib::CKKS::Context &cc_gpu, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &res, FIDESlib::CKKS::Ciphertext &ct);

/**
 * Move back ciphertexts from the GPU.
 * @param cc_gpu FIDESlib context.
 * @param res Destination ciphertexts.
 * @param ct Fideslib ciphertexts.
 */
void move_back(const FIDESlib::CKKS::Context &cc_gpu, std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &res, std::vector<FIDESlib::CKKS::Ciphertext> &ct);

/**
 * Perform LR Training on GPU.
 * @param data  Data matrix.
 * @param results Results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @param public_key Public key.
 * @return Vector of iteration times.
 */
std::vector<iteration_time_t> logistic_regression_gpu_train(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   FIDESlib::CKKS::Ciphertext &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &public_key);

/**
* Perform LR Inference on GPU.
 * @param data  Data matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param keys Keypair.
 * @return Vector of iteration times.
 */
std::vector<iteration_time_t> logistic_regression_gpu_inference(std::vector<std::vector<double>> &data,
                                   const FIDESlib::CKKS::Ciphertext &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   const lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys);

/**
* Perform LR Training on GPU (with NAG).
 * @param data  Data matrix.
 * @param results Results matrix.
 * @param weights Ciphertext of encrypted weight matrix.
 * @param rows Matrix row dimension.
 * @param columns Matrix column dimension (i.e. augmented number of features)
 * @param samples_last_ciphertext Number of rows of the last ciphertext from the given ones.
 * @param training_iterations Training iterations.
 * @param pk Public key.
 * @return Vector of iteration times.
 */
std::vector<iteration_time_t> logistic_regression_gpu_train_accelerated(const std::vector<std::vector<double>> &data,
                                   const std::vector<std::vector<double>> &results,
                                   FIDESlib::CKKS::Ciphertext &weights,
                                   size_t rows,
                                   size_t columns,
                                   size_t samples_last_ciphertext,
                                   size_t training_iterations,
                                   const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

#endif //FIDES_CUH