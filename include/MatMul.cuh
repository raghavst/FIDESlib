#pragma once

#include <CKKS/Context.cuh>
#include <cassert>
#include <iostream>
#include <optional>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Plaintext.cuh"

#include "MatMul.h"
#include "PolyApprox.cuh"
#include "Transpose.cuh"

#define LOW_MEM true
#define BSGS true

using namespace lbcrypto;

namespace FIDESlib::CKKS {

struct MatrixMatrixProductPrecomputations_GPU {
    int rowSize;
    std::vector<std::vector<Plaintext>> sigmaPlaintexts;
    //std::vector<std::vector<double>> tauVectors;
    std::vector<Plaintext> tauPlaintexts;
    std::vector<std::vector<Plaintext>> phiPlaintexts;

    // std::vector<std::vector<Plaintext>> weightLinearTransform; //

#if BSGS
    int bStep;
    std::vector<Plaintext*> pts_1;
    std::vector<Plaintext*> pts_2;
    std::vector<Plaintext*> pts_3_1;
    std::vector<Plaintext*> pts_3_2;
#endif

    MatrixMatrixProductPrecomputations_GPU(const MatrixMatrixProductPrecomputations_GPU&) = delete;
    MatrixMatrixProductPrecomputations_GPU& operator=(const MatrixMatrixProductPrecomputations_GPU&) = delete;

    MatrixMatrixProductPrecomputations_GPU(MatrixMatrixProductPrecomputations_GPU&&) noexcept = default;
    MatrixMatrixProductPrecomputations_GPU& operator=(MatrixMatrixProductPrecomputations_GPU&&) noexcept = default;

    MatrixMatrixProductPrecomputations_GPU() = default;
    ~MatrixMatrixProductPrecomputations_GPU() = default;
};

void CCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp);

void CCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp);

void CCMMwBias_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
                   const MatrixMatrixProductPrecomputations_GPU& precomp,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias);

void PCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp);

void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias);

MatrixMatrixProductPrecomputations_GPU getMatrixMatrixProductPrecomputations_GPU(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize, int bStep,
    int levelCP, int levelCC, bool fuse_boot_prescale_CCMM, int slots);

std::vector<int> GenerateMatMulRotationIndices_GPU(int rowSize, int bStep);

std::vector<double> getPCMM_bMatrix(std::vector<double> weights, int rowSize);

FIDESlib::CKKS::Ciphertext rotsum_GPU(FIDESlib::CKKS::Ciphertext& in, int blockSize, int padding);

}  // namespace FIDESlib::CKKS
