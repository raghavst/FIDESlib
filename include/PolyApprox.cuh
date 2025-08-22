//
// Created by seyda on 5/19/25.
//
#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "MatMul.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {

void evalFunction(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey,
                  std::vector<double> cheb_coeff, int numSlots, double lower_bound, double upper_bound,
                  bool bts = false);
void evalRelu(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots);
void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots,
              double lower_bound = -1, double upper_bound = 1, bool bts = true);
// void EvalSoftmax(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots, int blockSize,
//                  int bStepAcc, bool bts = true);

void EvalRelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey,
                     int numSlots);
void EvalSoftmax_Matrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
                        const KeySwitchingKey& keySwitchingKey, int numSlots, int blockSize, int bStepAcc, bool bts = false);
void EvalLayerNorm_Matrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, 
                          std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
                          const KeySwitchingKey& keySwitchingKey,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                          int bStepAcc, bool bts = false);
// void EvalLayerNorm_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
//                           const KeySwitchingKey& keySwitchingKey,
//                           std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& weight,
//                           std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias, int numSlots, int blockSize,
//                           int bStepAcc, bool bts = false);

}  // namespace FIDESlib::CKKS