//
// Created by seyda on 6/10/25.
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

#include "MatMul.cuh"
#include "PolyApprox.cuh"
#include "Transformer.cuh"
#include "Transpose.cuh"

#include "CKKS/AccumulateBroadcast.cuh"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"

using namespace FIDESlib::CKKS;

namespace FIDESlib::Testing {

class TransformerTests : public GeneralParametrizedTest {};

auto dropMatrixLevel = [](std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level) {
    for (auto& i : in)
        for (auto& j : i) {
            if (j.NoiseLevel == 2)
                j.rescale();
            if (j.getLevel() > level) {
                j.dropToLevel(level);
                assert(j.getLevel() == level);
            }
        }
};

struct EncoderConfiguration {
    bool verbose = true;
    int bStep = 16;
    uint32_t bStepBoot = 4;
    int bStepAcc = 4;
    uint32_t levelsStC = 3;
    uint32_t levelsCtS = 3;
    int level_matmul = 6;
    bool prescale = true;
    size_t rows = 128;
    size_t cols1 = 128;
    size_t cols2 = 128;
    int numSlots;
    int blockSize;
};

std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(
    PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
    TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
    EncoderConfiguration& conf) {
    constexpr bool PRINT = false;

    Context& cc = tokens[0][0].cc;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_K, GPUResult_Q, GPUResult_V;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_QKT, GPUResult_Sm_V, GPUResult_Output, GPUResult_Up,
        GPUResult_Down;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens",
                    false);
    dropMatrixLevel(tokens, conf.level_matmul);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens",
                    false);

    // K, Q, V PCMM
    PCMM_GPU(tokens, weights_layer.Wk, conf.blockSize, GPUResult_K, precomp_gpu, weights_layer.bk);
    cudaDeviceSynchronize();
    if constexpr (PRINT)
        std::cout << "# limbs K: " << GPUResult_K[0][0].getLevel() << " " << GPUResult_K[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_K, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "GPUResult_K", false);
    // Transpose K
    auto GPUResult_K_T = MatrixTranspose_GPU(std::move(GPUResult_K), conf.blockSize, Tprecomp_gpu);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_K_T, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "GPUResult_K_T", false);
    if constexpr (PRINT)
        std::cout << "# limbs KT: " << GPUResult_K_T[0][0].getLevel() << std::endl;

    PCMM_GPU(tokens, weights_layer.Wq, conf.blockSize, GPUResult_Q, precomp_gpu, weights_layer.bq);
    if constexpr (PRINT)
        std::cout << "# limbs Q: " << GPUResult_Q[0][0].getLevel() << " " << GPUResult_Q[0][0].NoiseLevel << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Q, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "GPUResult_Q", false);
    // Q.K CCMM
    dropMatrixLevel(GPUResult_Q, conf.level_matmul - 3);
    dropMatrixLevel(GPUResult_K_T, conf.level_matmul - 4);
    CCMM_GPU(GPUResult_Q, GPUResult_K_T, conf.blockSize, GPUResult_QKT, precomp_gpu);
    GPUResult_Q.clear();
    GPUResult_K_T.clear();
    if constexpr (PRINT)
        std::cout << "# limbs QKT: " << GPUResult_QKT[0][0].getLevel() << " " << GPUResult_QKT[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "GPUResult_QKT", false);
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_QKT, conf.numSlots, conf.prescale);
    if constexpr (PRINT)
        std::cout << "# limbs QKT: " << GPUResult_QKT[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "GPUResult_QKT", true);
    // Softmax
    auto CPUcontext = keys_.secretKey->GetCryptoContext();
    EvalSoftmax_Matrix(CPUcontext, GPUResult_QKT, cc.GetEvalKey(), conf.numSlots, conf.blockSize, conf.bStepAcc, true);
    if constexpr (PRINT)
        std::cout << "# limbs Sm: " << GPUResult_QKT[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_QKT, conf.numSlots, true);
    if constexpr (PRINT)
        std::cout << "# limbs Sm: " << GPUResult_QKT[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_QKT, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Softmax: ", false);

    PCMM_GPU(tokens, weights_layer.Wv, conf.blockSize, GPUResult_V, precomp_gpu, weights_layer.bv);
    tokens.clear();
    if constexpr (PRINT)
        std::cout << "# limbs V: " << GPUResult_V[0][0].getLevel() << " " << GPUResult_V[0][0].NoiseLevel << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_V: ", false);

    if constexpr (PRINT)
        std::cout << "# limbs QKT: " << GPUResult_QKT[0][0].getLevel() << " " << GPUResult_QKT[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        std::cout << "# limbs V: " << GPUResult_V[0][0].getLevel() << " " << GPUResult_V[0][0].NoiseLevel << std::endl;

    dropMatrixLevel(GPUResult_QKT, conf.level_matmul - 3);
    dropMatrixLevel(GPUResult_V, conf.level_matmul - 4);
    // Attention CCMM
    if constexpr (PRINT)
        std::cout << "# limbs QKT: " << GPUResult_QKT[0][0].getLevel() << " " << GPUResult_QKT[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        std::cout << "# limbs V: " << GPUResult_V[0][0].getLevel() << " " << GPUResult_V[0][0].NoiseLevel << std::endl;

    CCMM_GPU(GPUResult_QKT, GPUResult_V, conf.blockSize, GPUResult_Sm_V, precomp_gpu);
    GPUResult_V.clear();
    GPUResult_QKT.clear();
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Sm_V: ", false);
    if constexpr (PRINT)
        std::cout << "# limbs Attn: " << GPUResult_Sm_V[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Sm_V, conf.numSlots, conf.prescale);
    if constexpr (PRINT)
        std::cout << "# limbs Attn: " << GPUResult_Sm_V[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Sm_V: ", false);
    // Output CCMM
    dropMatrixLevel(GPUResult_Sm_V, conf.level_matmul);
    PCMM_GPU(GPUResult_Sm_V, weights_layer.Wo, conf.blockSize, GPUResult_Output, precomp_gpu, weights_layer.bo);
    if constexpr (PRINT)
        std::cout << "# limbs O: " << GPUResult_Output[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_output: ", false);
    // TODO unused lower levels here
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Output, conf.numSlots, false);
    if constexpr (PRINT)
        std::cout << "# limbs O: " << GPUResult_Output[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_output: ", true);
    // Layer Norm
    EvalLayerNorm_Matrix(CPUcontext, GPUResult_Output, cc.GetEvalKey(), weights_layer.Wln1, weights_layer.bln1,
                         conf.numSlots, conf.blockSize, conf.bStepAcc, true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_LN: ", false);
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Output, conf.numSlots, true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Output[0][0].getLevel() << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_LN: ", false);
    // Up PCMM
    dropMatrixLevel(GPUResult_Output, conf.level_matmul - 2);
    PCMM_GPU(GPUResult_Output, weights_layer.Wu, conf.blockSize, GPUResult_Up, precomp_gpu, weights_layer.bu);
    GPUResult_Output.clear();
    if constexpr (PRINT)
        std::cout << "# limbs U: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_up: ", false);

    // MatrixSquare();
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Up, conf.numSlots, false);
    if constexpr (PRINT)
        std::cout << "# limbs U: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel
                  << std::endl;

    // dropMatrixLevel(GPUResult_Up, conf.level_matmul);

    if constexpr (PRINT)
        std::cout << "# limbs U: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel
                  << std::endl;

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_up: ", true);
    // ReLU
    EvalRelu_Matrix(GPUResult_Up, cc.GetEvalKey(), conf.numSlots);
    if constexpr (PRINT)
        std::cout << "# limbs r: " << GPUResult_Up[0][0].getLevel() << " " << GPUResult_Up[0][0].NoiseLevel
                  << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_ReLu: ", false);
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Up, conf.numSlots, true);
    if constexpr (PRINT)
        std::cout << "# limbs r: " << GPUResult_Up[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_ReLu: ", true);
    // Down PCMM
    dropMatrixLevel(GPUResult_Up, conf.level_matmul);
    PCMM_GPU(GPUResult_Up, weights_layer.Wd, conf.blockSize, GPUResult_Down, precomp_gpu, weights_layer.bd);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", false);
    if constexpr (PRINT)
        std::cout << "# limbs D: " << GPUResult_Down[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Down, conf.numSlots);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << std::endl;

    // Layer Norm
    EvalLayerNorm_Matrix(CPUcontext, GPUResult_Down, cc.GetEvalKey(), weights_layer.Wln2, weights_layer.bln2,
                         conf.numSlots, conf.blockSize, conf.bStepAcc, true);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down_LN: ", false);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Down, conf.numSlots, true);
    if constexpr (PRINT)
        std::cout << "# limbs LN: " << GPUResult_Down[0][0].getLevel() << std::endl;
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down_LN: ", false);
    return GPUResult_Down;
}

TEST_P(TransformerTests, EmbeddingGeneration) {
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);

    EncoderConfiguration conf{.numSlots = cc->GetEncodingParams()->GetBatchSize(),
                              .blockSize = int(sqrt(cc->GetEncodingParams()->GetBatchSize()))};

    {
        char* res = getenv("FIDESLIB_USE_NUM_GPUS");
        if (res && !(0 == std::strcmp(res, ""))) {
            int num_dev = atoi(res);
            if (num_dev > 0) {
                std::vector<int> dev;
                for (int i = 0; i < num_dev; ++i) {
                    dev.push_back(i);
                }
                devices = dev;
            }
            std::cout << "Devices: " << num_dev << std::endl;
        }
    }

    {
        char* res = getenv("FIDESLIB_USE_BSTEPBOOT");
        if (res && !(0 == std::strcmp(res, ""))) {
            int num_dev = atoi(res);
            conf.bStepBoot = num_dev;
            std::cout << "bStepBoot: " << num_dev << std::endl;
        }
    }

    {
        char* res = getenv("FIDESLIB_USE_BSTEP");
        if (res && !(0 == std::strcmp(res, ""))) {
            int num_dev = atoi(res);
            conf.bStep = num_dev;
            std::cout << "bStep: " << num_dev << std::endl;
        }
    }

    {
        char* res = getenv("FIDESLIB_USE_STC_CTS_LEVELS");
        if (res && !(0 == std::strcmp(res, ""))) {
            int num_dev = atoi(res);
            conf.levelsCtS = num_dev;
            conf.levelsStC = num_dev;
            std::cout << "StC and CtS levels: " << num_dev << std::endl;
        }
    }

    {
        char* res = getenv("FIDESLIB_USE_BSTEPACC");
        if (res && !(0 == std::strcmp(res, ""))) {
            int num_dev = atoi(res);
            conf.bStepAcc = num_dev;
            std::cout << "bStepAcc: " << num_dev << std::endl;
        }
    }

    if (conf.verbose) {
        std::cout << "Block size: " << conf.blockSize << std::endl;
        std::cout << "numSlots: " << conf.numSlots << std::endl;
    }

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), devices};

    GPUcc.batch = 100;

    // ------- Generate Keys and Move to GPU--------
    keys = cc->KeyGen();
    keys_ = keys;
    cc->EvalMultKeyGen(keys.secretKey);
    // Move keys to GPU
    auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc);
    eval_key_gpu.Initialize(GPUcc, eval_key);
    GPUcc.AddEvalKey(std::move(eval_key_gpu));

    std::cout << "//////////////////////////// Precomputations /////////////////////////////" << std::endl;

    std::vector<int32_t> rotation_indices = GenerateRotationIndices_GPU(conf.blockSize, conf.bStep, conf.bStepAcc);
    // Bad optional access??
    GenAndAddRotationKeys(cc, keys, GPUcc, rotation_indices);

    // Bootstrapping Precomputation
    cc->EvalBootstrapSetup({conf.levelsCtS, conf.levelsStC}, {conf.bStepBoot, conf.bStepBoot}, conf.numSlots);
    cc->EvalBootstrapKeyGen(keys.secretKey, conf.numSlots);

    FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, conf.numSlots, GPUcc);

    // Tokenizer
    std::string sentence = "a rewarding work of art for only the most patient and challenge-hungry moviegoers.";

    const std::string model_name = "bert-tiny-sst2";
    const std::string model_path = std::string(root_dir + "weights-" + model_name);

    std::cout << "Tokenizing the following sentence: '" << sentence << "'\n";
    tokenizer(sentence, model_name, model_path);

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
    encryptMatrixtoGPU(std::string(model_path + "/tokens.txt"), tokens_gpu, keys.publicKey, GPUcc, conf.numSlots,
                       conf.blockSize, conf.rows, conf.cols1, conf.level_matmul);
    ct_tokens = encryptMatrixtoCPU(std::string(model_path + "/tokens.txt"), keys.publicKey, conf.numSlots,
                                   conf.blockSize, conf.rows, conf.cols1);
    // Loading weights and biases

    struct PtWeights_GPU weights_layer0 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 0, conf.numSlots, conf.blockSize, conf.rows, conf.cols1,
                        conf.cols2, conf.level_matmul);

    struct PtWeights_GPU weights_layer1 =
        GetPtWeightsGPU(GPUcc, keys.publicKey, model_path, 1, conf.numSlots, conf.blockSize, conf.rows, conf.cols1,
                        conf.cols2, conf.level_matmul);

    struct MatrixMatrixProductPrecomputations_GPU precomp_gpu = getMatrixMatrixProductPrecomputations_GPU(
        GPUcc, cc, conf.blockSize, conf.bStep, conf.level_matmul, conf.level_matmul - 3, conf.prescale, conf.numSlots);

    TransposePrecomputations_GPU Tprecomp_gpu =
        getMatrixTransposePrecomputations_GPU(GPUcc, cc, conf.blockSize, conf.bStep, conf.level_matmul - 3);

    dropMatrixLevel(tokens_gpu, conf.level_matmul);

    FIDESlib::CKKS::RawCipherText raw_res;

    for (int i = 0; i < 1; ++i) {
        std::cout << "/////////////////////////////// ENCODER 1 //////////////////////////////" << std::endl;
        cudaDeviceSynchronize();
        auto start_bert_tiny = std::chrono::high_resolution_clock::now();
        auto start_gpu = std::chrono::high_resolution_clock::now();
        auto tokens_gpu2 = encoder(weights_layer0, precomp_gpu, Tprecomp_gpu, tokens_gpu, conf);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();
        std::cout << "took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;

        cudaDeviceSynchronize();
        std::cout << "/////////////////////////////// ENCODER 2 //////////////////////////////" << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
        tokens_gpu2 = encoder(weights_layer1, precomp_gpu, Tprecomp_gpu, tokens_gpu2, conf);
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        std::cout << "////////////////////////// POOLER & CLASSIFIER /////////////////////////" << std::endl;
        cudaDeviceSynchronize();
        start_gpu = std::chrono::high_resolution_clock::now();
        auto result =
            pooler(tokens_gpu2, weights_layer1.Wp, weights_layer1.bp, conf.numSlots, conf.blockSize, conf.bStepAcc);
        auto result_f =
            classifier(cc, result, weights_layer1.Wc, weights_layer1.bc, conf.numSlots, conf.blockSize, conf.bStepAcc);
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "took " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;

        auto end_bert_tiny = std::chrono::high_resolution_clock::now();
        std::cout << "BERT-tiny took "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_bert_tiny).count())
                  << " ms in total." << std::endl
                  << std::endl;

        result_f.store(GPUcc, raw_res);
    }

    auto result_gpu(ct_tokens[0][0]);
    GetOpenFHECipherText(result_gpu, raw_res);

    lbcrypto::Plaintext pt_result_gpu;
    cc->Decrypt(keys.secretKey, result_gpu, &pt_result_gpu);
    pt_result_gpu->SetLength(2);
    // std::cout << "result: " << pt_result_gpu;    // incorrect
}
INSTANTIATE_TEST_SUITE_P(LLMTests, TransformerTests,
                         testing::Values(tparams64_15_LLM_flex, tparams64_16_LLM_flex, tparams64_17_LLM_flex));
}  // namespace FIDESlib::Testing
