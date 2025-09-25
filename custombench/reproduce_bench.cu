#include <string>
#include <iomanip>
#include <omp.h>
#include <vector>

#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Parameters.cuh"
#include "CKKS/Context.cuh"
#include <CKKS/Plaintext.cuh>
#include <CKKS/Ciphertext.cuh>

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> openFHE_params;
    openFHE_params.SetMultiplicativeDepth(23);
    openFHE_params.SetSecurityLevel(lbcrypto::HEStd_NotSet);
    openFHE_params.SetFirstModSize(60);
    openFHE_params.SetScalingModSize(30);
    openFHE_params.SetRingDim(1 << 16);
    openFHE_params.SetNumLargeDigits(4);
    openFHE_params.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);

    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = GenCryptoContext(openFHE_params);
    cc->Enable(lbcrypto::PKE);
    cc->Enable(lbcrypto::KEYSWITCH);
    cc->Enable(lbcrypto::LEVELEDSHE);
    cc->Enable(lbcrypto::ADVANCEDSHE);
    cc->Enable(lbcrypto::FHE);

    FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Parameters params{};
    params = params.adaptTo(raw_param);
    std::vector<int> GPUs {0};
    FIDESlib::CKKS::Context GPUcc{params, GPUs};
    GPUcc.batch = 12;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys = cc->KeyGen();    
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1});

    std::cout << "primes size: " << params.primes.size() << std::endl;
    std::cout << "s_primes size: " << params.Sprimes.size() << std::endl;

    const int row_size = 1 << 15;
    std::vector<double> message(row_size, 1);

    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(message);
    lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(message);

    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
    FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

    FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
    FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);

    FIDESlib::CKKS::KeySwitchingKey kskEval(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
    kskEval.Initialize(GPUcc, rawKskEval);

    FIDESlib::CKKS::KeySwitchingKey kskRot(GPUcc);
    FIDESlib::CKKS::RawKeySwitchKey rawKskEvalr = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
    kskRot.Initialize(GPUcc, rawKskEvalr);

    float time = 0;
    float time_addition = 0;
    float time_multiplication = 0;
    float time_rotaterow = 0;
    float time_rescale = 0;

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    int repeat_count = 100;
    for (int j = 0; j < repeat_count; j++)
    {
        {
            FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
            FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
            cudaEventRecord(start_time);
            GPUct1.add(GPUct2);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_addition += time;
        }

        {   
            FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
            FIDESlib::CKKS::Ciphertext GPUct2(GPUcc, raw2);
            cudaEventRecord(start_time);
            GPUct1.mult(GPUct2, kskEval, false);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_multiplication += time;

            cudaEventRecord(start_time);
            GPUct1.rescale();
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_rescale += time;
        }

        {   
            FIDESlib::CKKS::Ciphertext GPUct1(GPUcc, raw1);
            cudaEventRecord(start_time);
            GPUct1.rotate(1, kskRot);
            cudaEventRecord(stop_time);
            cudaEventSynchronize(stop_time);
            cudaEventElapsedTime(&time, start_time, stop_time);
            time_rotaterow += time;
        }


        cudaDeviceSynchronize();
    }

    std::cout
        << "=================== Benchmark CKKS with poly_modulus_degree: "
        << (1 << 16) << " ===================" << std::endl;
    std::cout << "Average addition timing: "
                << (time_addition / repeat_count) << " ms" << std::endl;
    std::cout << "Average multiplication + relinearization timing: "
                << (time_multiplication / repeat_count) << " ms" << std::endl;
    std::cout << "Average rotate row timing: "
                << (time_rotaterow / repeat_count) << " ms" << std::endl;
    std::cout << "Average rescale timing: " << (time_rescale / repeat_count)
                << " ms" << std::endl;

    return EXIT_SUCCESS;
}