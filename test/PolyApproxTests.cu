//
// Created by seyda on 5/8/25.
//

#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <CKKS/KeySwitchingKey.cuh>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ParametrizedTest.cuh"

#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "MatMul.h"
#include "CKKS/AccumulateBroadcast.cuh"

using namespace std;
using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

    class PolyApproxTests : public GeneralParametrizedTest {};

    TEST_P(PolyApproxTests, SoftmaxMatrix) {

        std::vector<int> accum_indices = GetAccumulateRotationIndices(4, 1, 128);
        // std::vector<int> rotation_indices =  GetbroadcastRotationIndices(4, 128, 128*128);

        std::cout << "accumulate indices: " << std::endl; 
        for (int i=0; i<accum_indices.size(); i++){
            std::cout << accum_indices[i] << ", ";
        }
        // std::cout << std::endl; 
        // std::cout << "broadcast indices: " << std::endl; 
        // for (int i=0; i<rotation_indices.size(); i++){
        //     std::cout << rotation_indices[i] << ", ";
        // }
        // std::cout << std::endl; 

        cc->Enable(lbcrypto::PKE);
        cc->Enable(lbcrypto::KEYSWITCH);
        cc->Enable(lbcrypto::LEVELEDSHE);
        cc->Enable(lbcrypto::ADVANCEDSHE);  // Chebyshev approximation
        cc->Enable(lbcrypto::FHE);          // Bootstrap

        FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
        FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

        int numSlots = cc->GetEncodingParams()->GetBatchSize();
        int blockSize = int(sqrt(numSlots));

        GPUcc.batch = 12;

        auto keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);

        std::vector<int32_t> rotation_indices = {1,2,3,4,8,12,16,32,48,64, -1, -2, -4, -8, -16, -32, -64};
        // std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(blockSize, 4, 4);
        cc->EvalAtIndexKeyGen(keys.secretKey, rotation_indices);

        // Move keys to GPU
        auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
        FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc);
        eval_key_gpu.Initialize(GPUcc, eval_key);
        GPUcc.AddEvalKey(std::move(eval_key_gpu));

        for (int i : rotation_indices) {
            auto clave_rotacion = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, i, cc);
            FIDESlib::CKKS::KeySwitchingKey clave_rotacion_gpu(GPUcc);
            clave_rotacion_gpu.Initialize(GPUcc, clave_rotacion);
            GPUcc.AddRotationKey(i, std::move(clave_rotacion_gpu));
        }

        // // Bootstrapping Precomputation
        // cc->EvalBootstrapSetup({3, 3}, {16, 16}, numSlots);
        // cc->EvalBootstrapKeyGen(keys.secretKey, numSlots);

        // FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, numSlots, GPUcc);

        // --------- Encrypt Inputs -------
        const std::string model_name = "bert-tiny-sst2";
        const std::string model_path = std::string(root_dir + "weights-" + model_name);

        std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> QKT_cpu = encryptMatrixtoCPU(
            std::string(model_path + "/intermediate_results/QKT.txt"), keys.publicKey, numSlots, blockSize, 128, 128, false);
        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> QKT;
        encryptMatrixtoGPU(std::string(model_path + "/intermediate_results/QKT.txt"), QKT, keys.publicKey, GPUcc,
                        numSlots, blockSize, 128, 128);

        std::vector<std::vector<FIDESlib::CKKS::Plaintext>> Wln, bln;
        encodeMatrixtoGPU(std::string(model_path + "/layer1_Wln2.txt"), Wln, keys.publicKey, GPUcc, numSlots, blockSize, 128, 128, 0, true);
        encodeMatrixtoGPU(std::string(model_path + "/layer1_bln2.txt"), bln, keys.publicKey, GPUcc, numSlots, blockSize, 128, 128, 0, true);

        auto ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/tokens.txt"), keys.publicKey, numSlots, blockSize, 128, 128);
        // ------- Function Evaluation on GPU ------
        std::cout << "Layer Norm on GPU: " << std::endl;

        // auto start_gpu = std::chrono::high_resolution_clock::now();
        // MatrixMultScalar(QKT, 0.125);
        // EvalSoftmax_Matrix(cc, QKT, GPUcc.GetEvalKey(), numSlots, blockSize, 4);
        // auto end_gpu = std::chrono::high_resolution_clock::now();

        printMatrix(decryptGPUMatrix(QKT, keys.secretKey, ct_tokens, numSlots, blockSize), 3, 3, "Input: ", false);

        EvalLayerNorm_Matrix(cc, QKT, GPUcc.GetEvalKey(), Wln, bln, numSlots, blockSize, 4);

        printMatrix(decryptGPUMatrix(QKT, keys.secretKey, ct_tokens, numSlots, blockSize), 3, 3, "Output: ", false);

        // std::cout << "took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
        //         << " ms." << std::endl
        //         << std::endl;
    }
    INSTANTIATE_TEST_SUITE_P(LLMTests, PolyApproxTests, testing::Values(tparams64_15_LLM_flex));
}  // namespace FIDESlib::Testing