//
// Created by seyda on 5/28/25.
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
#include "Transpose.cuh"

#include "Transformer.cuh"

using namespace std;
using namespace FIDESlib::CKKS;
using namespace lbcrypto;
using namespace std::chrono;

namespace FIDESlib::Testing {

class CCMMTests : public GeneralParametrizedTest {};

TEST_P(CCMMTests, BSGS) {

    bool verbose = true;

    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Context GPUcc{fideslibParams.adaptTo(raw_param), generalTestParams.GPUs};

    GPUcc.batch = 100;

    int numSlots = cc->GetEncodingParams()->GetBatchSize();
    int blockSize = int(sqrt(numSlots));
    int bStep = 16;
    int level = 3;
    int horiz_print_size = 2;
    int verti_print_size = 2;
    if (verbose)
        std::cout << "Block size: " << blockSize << std::endl;

    auto GT_K = readGroundTruth("../weights-bert-tiny/intermediate_outputs/layer0_K.txt");

    // ------- Generate Keys and Move to GPU--------
    auto keys = cc->KeyGen();

    size_t rows = 128;
    size_t cols1 = 128;
    size_t cols2 = 128;

    // ------- CCMM on GPU ------

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Wk_gpu, Wq_gpu, Wv_gpu;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
    encryptMatrixtoGPU("../weights-bert-tiny/tokens.txt", tokens_gpu, keys.publicKey, GPUcc, numSlots, blockSize, rows,
                       cols1, level);

    encryptMatrixtoGPU("../weights-bert-tiny/layer0_Wk.txt", Wk_gpu, keys.publicKey, GPUcc, numSlots, blockSize, cols1,
                       cols2, level - 1);
    encryptMatrixtoGPU("../weights-bert-tiny/layer0_Wq.txt", Wq_gpu, keys.publicKey, GPUcc, numSlots, blockSize, cols1,
                       cols2, level - 1);
    encryptMatrixtoGPU("../weights-bert-tiny/layer0_Wv.txt", Wv_gpu, keys.publicKey, GPUcc, numSlots, blockSize, cols1,
                       cols2, level - 1);

    auto ct_tokens =
        encryptMatrixtoCPU("../weights-bert-tiny/tokens.txt", keys.publicKey, numSlots, blockSize, rows, cols1, false);

    cc->EvalMultKeyGen(keys.secretKey);

    // JKLS MatMul rotation indices
    std::vector<int32_t> rotation_indices_MM = GenerateMatMulRotationIndices_GPU(blockSize, bStep);

    GenAndAddRotationKeys(cc, keys, GPUcc, rotation_indices_MM);

    // Move keys to GPU
    auto eval_key = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    FIDESlib::CKKS::KeySwitchingKey eval_key_gpu(GPUcc);
    eval_key_gpu.Initialize(GPUcc, eval_key);
    GPUcc.AddEvalKey(std::move(eval_key_gpu));

    if (verbose)
        std::cout << "MatrixMatrixProduct on GPU: " << std::endl;

    if (verbose)
        std::cout << "Precomp: ";
    auto start_gpu = std::chrono::high_resolution_clock::now();
    struct MatrixMatrixProductPrecomputations_GPU precomp_gpu =
        getMatrixMatrixProductPrecomputations_GPU(GPUcc, cc, blockSize, bStep, level, level, false, 0);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
              << " ms." << std::endl;

    if (verbose)
        std::cout << "3 CCMMs: " << std::endl;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> result1, result2, result3;
    size_t N = 1;
    cudaDeviceSynchronize();
    start_gpu = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; i++) {
        CCMM_GPU(tokens_gpu, Wk_gpu, blockSize, result1, precomp_gpu);
        CCMM_GPU(tokens_gpu, Wq_gpu, blockSize, result2, precomp_gpu);
        CCMM_GPU(tokens_gpu, Wv_gpu, blockSize, result3, precomp_gpu);
    }
    cudaDeviceSynchronize();
    end_gpu = std::chrono::high_resolution_clock::now();

    std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) / 3
              << " ms." << std::endl;

    auto final_CCMMResult_K = decryptGPUMatrix(result1, keys.secretKey, ct_tokens, numSlots, blockSize);
    printMatrix(final_CCMMResult_K, horiz_print_size, verti_print_size, "CCMM", false);

    auto final_CCMMResult_Q = decryptGPUMatrix(result2, keys.secretKey, ct_tokens, numSlots, blockSize);
    printMatrix(final_CCMMResult_Q, horiz_print_size, verti_print_size, "CCMM", false);

    auto final_CCMMResult_V = decryptGPUMatrix(result3, keys.secretKey, ct_tokens, numSlots, blockSize);
    printMatrix(final_CCMMResult_V, horiz_print_size, verti_print_size, "CCMM", false);

    // // --------- CCMM on CPU -------
    // auto ctxt1 = encryptMatrixtoCPU("../weights-bert-tiny/layer0_Wk.txt", keys.publicKey, blockSize, numSlots, cols1, cols2, false);
    // auto ctxt2 = encryptMatrixtoCPU("../weights-bert-tiny/layer0_Wq.txt", keys.publicKey, blockSize, numSlots, cols1, cols2, false);

    // if (verbose)    std::cout << "MatrixMatrixProduct on CPU: " << std::endl;

    // if (verbose)    std::cout << "Precomp: " ;
    // auto start_cpu = std::chrono::high_resolution_clock::now();
    // struct MatrixMatrixProductPrecomputations precomp = getMatrixMatrixProductPrecomputations(cc, blockSize);
    // auto end_cpu = std::chrono::high_resolution_clock::now();
    // std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count())
    // << " ms." << std::endl;

    // std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> cResult, cResult2, cResult3;

    // if (verbose)    std::cout << "3 CCMMs: " << std::endl ;

    // start_cpu = std::chrono::high_resolution_clock::now();
    // int N = 1;
    // for (size_t i = 0; i < N; i++){
    //     std::cout << "Step 1..." << std::endl;
    //     MatrixMatrixProduct(ctxt1, ctxt2, blockSize, cResult, precomp);
    //     std::cout << "Step 2..." << std::endl;
    //     MatrixMatrixProduct(cResult, ctxt2, blockSize, cResult2, precomp);
    //     std::cout << "Step 3..." << std::endl;
    //     MatrixMatrixProduct(cResult2, ctxt2, blockSize, cResult3, precomp);
    // }
    // end_cpu = std::chrono::high_resolution_clock::now();

    // std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count())/3
    // << " ms." << std::endl;
}

INSTANTIATE_TEST_SUITE_P(LLMTests, CCMMTests, testing::Values(tparams64_15_LLM_flex));
}  // namespace FIDESlib::Testing