#pragma once

#include <CKKS/Context.cuh>
#include <cassert>
#include <iostream>
#include <optional>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "MatMul.h"
#include "MatMul.cuh"

#include <cuda_runtime.h>

using namespace lbcrypto;

namespace FIDESlib::CKKS {

    struct PtWeights_GPU {
        std::vector<std::vector<FIDESlib::CKKS::Plaintext>> Wk, Wq, Wv, Wo, Wu, Wd, Wln1, Wln2, Wc, Wp;
        std::vector<std::vector<FIDESlib::CKKS::Plaintext>> bk, bq, bv, bo, bu, bd, bln1, bln2, bc, bp;
    };

    struct Weights_GPU {
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Wk, Wq, Wv, Wo, Wu, Wd, Wln1, Wln2, Wc, Wp;
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> bk, bq, bv, bo, bu, bd, bln1, bln2, bc, bp;
    };

PtWeights_GPU GetPtWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                              const std::string& model_path, int layername, int numSlots, int blockSize, int rows,
                              int cols1, int cols2, int level);

Weights_GPU GetWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                          const std::string& model_path, int layername, int numSlots, int blockSize, int rows,
                          int cols1, int cols2, int level);

std::vector<std::vector<lbcrypto::Plaintext>> EncodeMatrix(const std::vector<std::vector<std::vector<double>>>& matrix,
                                                           lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
                                                           int level);

void encodeMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& pt_inputs_gpu,
                       lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, int numSlots,
                       int blockSize, size_t rows, size_t cols, int level = 0, bool if_repeat = false,
                       bool prescale = false);

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> encryptMatrixtoCPU(
    const std::string& filename, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int numSlots, int blockSize,
    size_t rows, size_t cols, bool if_repeat = false);

void encryptMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& inputs_gpu,
                        lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc,
                        int numSlots, int blockSize, size_t rows, size_t cols, int level = 0, bool if_repeat = false);

std::vector<std::vector<double>> decryptGPUMatrix(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& result_gpu,
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& dummy, int numSlots, int blockSize);

    void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots,
                     bool input_prescaled = false);
    void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double scale);

    void MatrixSquare(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, const KeySwitchingKey& keySwitchingKey);

    FIDESlib::CKKS::Ciphertext classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input, 
                                std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight, std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize, int bStepAcc);

    FIDESlib::CKKS::Ciphertext classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input, 
                                std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& weight, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias, int numSlots, int blockSize, int bStepAcc);

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> pooler(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& weight, 
                                        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias, int numSlots, int blockSize) ;
                                            
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> pooler(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input, std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight, 
                                        std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize, int bStepAcc) ;

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bstep, int bStepAcc);

void tokenizer(const std::string& sentence, const std::string& model_name, const std::string& model_path);

void load_weights(const std::string& filename, std::vector<std::vector<double>>& matrix_weights, int rows, int cols);

// void load_bias(const std::string& filename, std::vector<double>& bias, int cols);
void load_bias(const std::string& filename, std::vector<std::vector<double>>& bias_matrix, int rows, int cols);

std::vector<std::vector<double>> readGroundTruth(const std::string& filename);

}  // namespace FIDESlib::CKKS
