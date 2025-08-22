#include "Transformer.cuh"

#include "CKKS/AccumulateBroadcast.cuh"

namespace FIDESlib::CKKS {

struct PtWeights_GPU GetPtWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                     const std::string& model_path, int layerNo, int numSlots, int blockSize, int rows,
                                     int cols1, int cols2, const int level) {

    PtWeights_GPU pt_weights_gpu;

    std::string path = std::string(model_path + "/layer") + std::to_string(layerNo);

    //int level = 18;
    // if (layerNo == 1){
    //     level = 15;
    // }

    encodeMatrixtoGPU(path + "_Wk.txt", pt_weights_gpu.Wk, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_Wq.txt", pt_weights_gpu.Wq, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_Wv.txt", pt_weights_gpu.Wv, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_bk.txt", pt_weights_gpu.bk, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 3, true);
    encodeMatrixtoGPU(path + "_bq.txt", pt_weights_gpu.bq, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 3, true);
    encodeMatrixtoGPU(path + "_bv.txt", pt_weights_gpu.bv, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 3, true);

    encodeMatrixtoGPU(path + "_Wo.txt", pt_weights_gpu.Wo, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_Wu.txt", pt_weights_gpu.Wu, publicKey, GPUcc, numSlots, blockSize, cols1, cols2 * 4,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_Wd.txt", pt_weights_gpu.Wd, publicKey, GPUcc, numSlots, blockSize, cols1 * 4, cols2,
                      level - 2, false);
    encodeMatrixtoGPU(path + "_bo.txt", pt_weights_gpu.bo, publicKey, GPUcc, numSlots, blockSize, cols1, cols2,
                      level - 3, true);
    encodeMatrixtoGPU(path + "_bu.txt", pt_weights_gpu.bu, publicKey, GPUcc, numSlots, blockSize, cols1, cols2 * 4,
                      level - 3, true);
    encodeMatrixtoGPU(path + "_bd.txt", pt_weights_gpu.bd, publicKey, GPUcc, numSlots, blockSize, cols1 * 4, cols2,
                      level - 3, true);

    encodeMatrixtoGPU(path + "_Wln1.txt", pt_weights_gpu.Wln1, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, 2,
                      true, true);
    encodeMatrixtoGPU(path + "_bln1.txt", pt_weights_gpu.bln1, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, 0,
                      true, true);
    encodeMatrixtoGPU(path + "_Wln2.txt", pt_weights_gpu.Wln2, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, 2,
                      true, true);
    encodeMatrixtoGPU(path + "_bln2.txt", pt_weights_gpu.bln2, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, 0,
                      true, true);

    if (layerNo == 1) {
        encodeMatrixtoGPU(model_path + "/Wp.txt", pt_weights_gpu.Wp, publicKey, GPUcc, numSlots, blockSize, cols1,
                          cols2, level - 2, true);
        encodeMatrixtoGPU(model_path + "/bp.txt", pt_weights_gpu.bp, publicKey, GPUcc, numSlots, blockSize, cols1,
                          cols2, level - 3, true);
        encodeMatrixtoGPU(model_path + "/Wc.txt", pt_weights_gpu.Wc, publicKey, GPUcc, numSlots, blockSize, cols1,
                          cols2, level, true);
        encodeMatrixtoGPU(model_path + "/bc.txt", pt_weights_gpu.bc, publicKey, GPUcc, numSlots, blockSize, cols1,
                          cols2, level - 1, true);
    }

    return pt_weights_gpu;
}

struct Weights_GPU GetWeightsGPU(FIDESlib::CKKS::Context& GPUcc, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                 const std::string& model_path, int layerNo, int numSlots, int blockSize, int rows,
                                 int cols1, int cols2, const int level) {

    Weights_GPU weights_gpu;

    std::string path = std::string(model_path + "/layer") + std::to_string(layerNo);

    encryptMatrixtoGPU(path + "_Wk.txt", weights_gpu.Wk, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_Wq.txt", weights_gpu.Wq, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_Wv.txt", weights_gpu.Wv, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_bk.txt", weights_gpu.bk, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bq.txt", weights_gpu.bq, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bv.txt", weights_gpu.bv, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);

    encryptMatrixtoGPU(path + "_Wo.txt", weights_gpu.Wo, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_Wu.txt", weights_gpu.Wu, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_Wd.txt", weights_gpu.Wd, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       false);
    encryptMatrixtoGPU(path + "_bo.txt", weights_gpu.bo, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bu.txt", weights_gpu.bu, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bd.txt", weights_gpu.bd, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);

    encryptMatrixtoGPU(path + "_Wln1.txt", weights_gpu.Wln1, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bln1.txt", weights_gpu.bln1, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_Wln2.txt", weights_gpu.Wln2, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bln2.txt", weights_gpu.bln2, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);

    encryptMatrixtoGPU(path + "_Wp.txt", weights_gpu.Wp, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);
    encryptMatrixtoGPU(path + "_bp.txt", weights_gpu.bp, publicKey, GPUcc, numSlots, blockSize, cols1, cols2, level,
                       true);

    return weights_gpu;
}

std::vector<std::vector<lbcrypto::Plaintext>> EncodeMatrix(const std::vector<std::vector<std::vector<double>>>& matrix,
                                                           lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
                                                           int level) {

    std::vector<std::vector<lbcrypto::Plaintext>> ptMatrix(matrix.size());
    auto cc = publicKey->GetCryptoContext();
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(matrix[i][j], 1, level);
            ptMatrix[i].emplace_back(ptxt1);
        }
    }
    return ptMatrix;
}

void encodeMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& pt_inputs_gpu,
                       lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc, int numSlots,
                       int blockSize, size_t rows, size_t cols, int level, bool if_repeat, bool prescale) {

    auto cc = publicKey->GetCryptoContext();

    std::vector<std::vector<double>> inputs;
    if (if_repeat) {
        load_bias(filename, inputs, rows, cols);
    } else {
        load_weights(filename, inputs, rows, cols);
    }

    auto inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);

    if (prescale) {
        double scale = GetPreScaleFactor(GPUcc, numSlots);
        for (auto& i : inputs_temp)
            for (auto& j : i)
                for (auto& k : j)
                    k *= scale;
    }
    if (!if_repeat) {
        for (auto& i : inputs_temp)
            for (auto& j : i)
                j = getPCMM_bMatrix(j, blockSize);
    }
    auto pt_inputs = EncodeMatrix(inputs_temp, publicKey, GPUcc.L - level);

    pt_inputs_gpu.resize(pt_inputs.size());
    for (size_t i = 0; i < pt_inputs.size(); ++i) {
        pt_inputs_gpu[i].reserve(pt_inputs[0].size());
        for (size_t j = 0; j < pt_inputs[0].size(); ++j) {
            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(cc, pt_inputs[i][j]);
            FIDESlib::CKKS::Plaintext GPUpt1(GPUcc, raw_pt);
            pt_inputs_gpu[i].emplace_back(std::move(GPUpt1));
        }
    }
}

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> encryptMatrixtoCPU(
    const std::string& filename, lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, int numSlots, int blockSize,
    size_t rows, size_t cols, bool if_repeat) {

    auto cc = publicKey->GetCryptoContext();

    std::vector<std::vector<double>> inputs;
    if (if_repeat) {
        load_bias(filename, inputs, rows, cols);
    } else {
        load_weights(filename, inputs, rows, cols);
    }

    auto inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);
    auto inputs_cpu = EncryptMatrix(inputs_temp, publicKey);

    return inputs_cpu;
}

void encryptMatrixtoGPU(const std::string& filename, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& inputs_gpu,
                        lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey, FIDESlib::CKKS::Context& GPUcc,
                        int numSlots, int blockSize, size_t rows, size_t cols, int level, bool if_repeat) {

    auto cc = publicKey->GetCryptoContext();

    std::vector<std::vector<double>> inputs;
    if (if_repeat) {
        load_bias(filename, inputs, rows, cols);
    } else {
        load_weights(filename, inputs, rows, cols);
    }

    auto inputs_temp = extractAndLinearizeMatrix(inputs, numSlots, blockSize);
    auto ct_inputs = EncryptMatrix(inputs_temp, publicKey, GPUcc.L - level);
    //auto ct_inputs = EncryptMatrix(inputs_temp, publicKey, level);

    inputs_gpu.resize(ct_inputs.size());
    for (size_t i = 0; i < ct_inputs.size(); ++i) {
        inputs_gpu[i].reserve(ct_inputs[0].size());
        for (size_t j = 0; j < ct_inputs[0].size(); ++j) {
            auto raw = FIDESlib::CKKS::GetRawCipherText(cc, ct_inputs[i][j]);
            inputs_gpu[i].emplace_back(GPUcc, raw);
        }
    }
}

std::vector<std::vector<double>> decryptGPUMatrix(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& result_gpu,
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& dummy, int numSlots, int blockSize) {

    FIDESlib::CKKS::Context& GPUcc = result_gpu[0][0].cc;

    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> result_cpu(result_gpu.size());
    for (size_t i = 0; i < result_gpu.size(); ++i) {
        result_cpu[i].reserve(result_gpu[0].size());
        for (size_t j = 0; j < result_gpu[0].size(); ++j) {
            auto ctxt = dummy[0][0]->Clone();

            FIDESlib::CKKS::RawCipherText raw_res;
            result_gpu[i][j].store(GPUcc, raw_res);
            auto result(ctxt);
            GetOpenFHECipherText(result, raw_res);
            result_cpu[i].emplace_back(result);
        }
    }
    auto result_decrypted = DecryptMatrix(result_cpu, privateKey, numSlots);
    auto final_result_cpu = convertToLargeMatrix(result_decrypted, blockSize);

    return final_result_cpu;
}

FIDESlib::CKKS::Ciphertext classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                                      std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
                                      std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                                      std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots,
                                      int blockSize, int bStepAcc) {

    FIDESlib::CKKS::Context& cc = input[0][0].cc;

    FIDESlib::CKKS::Ciphertext sum(cc);

    for (size_t i = 0; i < input.size(); i++) {
        for (size_t j = 0; j < input[0].size(); j++) {
            input[i][j].multPt(weight[i][j], false);

            // auto output = rotsum_GPU(input[i][j], blockSize, 1);
            FIDESlib::CKKS::Accumulate(input[i][j], bStepAcc, 1, blockSize);

            auto& output = input[i][j];
            output.addPt(bias[i][j]);

            std::vector<double> mask(numSlots, 0.0);
            mask[0] = 1;
            mask[blockSize] = 1;
            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask));
            FIDESlib::CKKS::Plaintext GPUpt(cc, raw_pt);

            output.multPt(GPUpt, false);

            FIDESlib::CKKS::Ciphertext rotatedCt(cc);
            rotatedCt.copy(output);
            rotatedCt.rotate(blockSize - 1, cc.GetRotationKey(blockSize - 1));

            output.add(rotatedCt);

            if (i == 0 & j == 0) {
                sum.copy(output);
            } else {
                sum.add(output);
            }
        }
    }
    return sum;
}

FIDESlib::CKKS::Ciphertext classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                                      std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
                                      std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& weight,
                                      std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias, int numSlots,
                                      int blockSize, int bStepAcc) {

    FIDESlib::CKKS::Context& cc = input[0][0].cc;

    FIDESlib::CKKS::Ciphertext sum(cc);

    for (size_t i = 0; i < input.size(); i++) {
        for (size_t j = 0; j < input[0].size(); j++) {
            input[i][j].mult(input[i][j], weight[i][j], cc.GetEvalKey());
            auto output = rotsum_GPU(input[i][j], blockSize, 1);
            output.add(bias[i][j]);

            std::vector<double> mask(numSlots, 0.0);

            mask[0] = 1;
            mask[blockSize] = 1;

            auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask));
            FIDESlib::CKKS::Plaintext GPUpt(cc, raw_pt);

            output.multPt(GPUpt, false);

            FIDESlib::CKKS::Ciphertext rotatedCt(cc);
            rotatedCt.copy(output);
            rotatedCt.rotate(blockSize - 1, cc.GetRotationKey(blockSize - 1));

            output.add(rotatedCt);

            if (i == 0 & j == 0) {
                sum.copy(output);
            } else {
                sum.add(output);
            }
        }
    }
    return sum;
}
std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> pooler(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& weight,
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias, int numSlots, int blockSize, int bStepAcc) {

    FIDESlib::CKKS::Context& cc = input[0][0].cc;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> output_array;
    output_array.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(input[0].size());
        for (size_t j = 0; j < input[0].size(); j++) {

            input[i][j].mult(input[i][j], weight[i][j], cc.GetEvalKey());

            auto output = rotsum_GPU(input[i][j], blockSize, 1);

            output.add(bias[i][j]);

            evalTanh(output, cc.GetEvalKey(), numSlots, -1, 1, true);

            row.emplace_back(std::move(output));
        }
        output_array.emplace_back(std::move(row));
    }

    return output_array;
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> pooler(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
                                                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                                                            std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias,
                                                            int numSlots, int blockSize, int bStepAcc) {
    FIDESlib::CKKS::Context& cc = input[0][0].cc;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> output_array;
    output_array.reserve(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(input[0].size());
        for (size_t j = 0; j < input[0].size(); j++) {
            if (input[i][j].NoiseLevel == 2)
                input[i][j].rescale();
            input[i][j].dropToLevel(weight[i][j].c0.getLevel());

            input[i][j].multPt(weight[i][j], false);

            //auto output = rotsum_GPU(input[i][j], blockSize, 1);
            // FIDESlib::CKKS::Accumulate(input[i][j], bStepAcc, blockSize, blockSize);

            auto& output = input[i][j];
            output.addPt(bias[i][j]);

            output.multScalar(0.01);

            evalTanh(output, cc.GetEvalKey(), numSlots, -1, 1, true);

            row.emplace_back(std::move(output));
        }
        output_array.emplace_back(std::move(row));
    }

    return output_array;
}

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bStep, int bStepAcc) {
    // JKLS MatMul rotation indices
    std::vector<int32_t> rotation_indices_MM = GenerateMatMulRotationIndices_GPU(blockSize, bStep);

    // Transpose rotation indices
    std::vector<int> rotation_indices_T = GenerateTransposeRotationIndices_GPU(blockSize, bStep);

    std::vector<int> rotsum_indices = {1,  2,  4,  8,   16,  32,  64, -1,
                                       -2, -4, -8, -16, -32, -64, 127};  // 127 is for pooling

    std::vector<int> accum_indices = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, 1, blockSize);
    std::vector<int> broad_indices =
        FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, blockSize, blockSize * blockSize);

    std::vector<int> broad_indices2 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize);
    // if (blockSize == 128) rotsum_indices = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // Merge the rotation indices and remove duplicates

    std::set<int32_t> merged_set;  //(rotsum_indices.begin(), rotsum_indices.end());
    merged_set.insert(rotation_indices_MM.begin(), rotation_indices_MM.end());
    merged_set.insert(rotation_indices_T.begin(), rotation_indices_T.end());
    merged_set.insert(accum_indices.begin(), accum_indices.end());
    // merged_set.insert(broad_indices.begin(), broad_indices.end());
    merged_set.insert(broad_indices2.begin(), broad_indices2.end());
    std::vector<int32_t> rotation_indices(merged_set.begin(), merged_set.end());

    return rotation_indices;
}

void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots, bool input_prescaled) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            Bootstrap(matrix[i][j], numSlots, input_prescaled);
        }
    }
}

void MatrixSquare(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
                  const KeySwitchingKey& keySwitchingKey) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].mult(matrix[i][j], matrix[i][j], keySwitchingKey);
        }
    }
}

void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double scale) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].multScalar(scale);
        }
    }
}

void tokenizer(const std::string& sentence, const std::string& model_name, const std::string& model_path) {
    std::string cmd =
        "python3 ../src/python/ExtractEmbeddings.py \"" + sentence + "\" \"" + model_name + "\" \"" + model_path + "\"";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("Tokenizer script failed with exit code: " + std::to_string(ret));
    }
}

// Function to load .txt inputs into a 2-D matrix
void load_weights(const std::string& filename, std::vector<std::vector<double>>& matrix_weights, int rows, int cols) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: (load random instead) " << filename << std::endl;
        //exit(EXIT_FAILURE);
        matrix_weights = generateRandomMatrix(rows, cols, 42);
        return;
    }

    matrix_weights.assign(rows, std::vector<double>(cols, 0.0));

    std::string line;
    size_t i = 0;

    while (std::getline(file, line) && i < static_cast<size_t>(rows)) {
        std::istringstream ss(line);
        double value;
        size_t j = 0;

        while (ss >> value && j < static_cast<size_t>(cols)) {
            matrix_weights[i][j] = value;
            j++;
        }
        i++;
    }
}

// Function to load .txt bias into a 2D matrix [rows][cols], zero-padded and row-repeated
void load_bias(const std::string& filename, std::vector<std::vector<double>>& bias_matrix, int rows, int cols) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file (load random instead) " << filename << std::endl;
        //exit(EXIT_FAILURE);
        bias_matrix = generateRandomMatrix(rows, cols, 42);
        return;
    }

    std::vector<double> bias_row;
    bias_row.reserve(cols);

    std::string line;
    double value;
    while (std::getline(file, line) && static_cast<int>(bias_row.size()) < cols) {
        std::istringstream ss(line);
        if (ss >> value) {
            bias_row.emplace_back(value);
        }
    }

    // Zero-pad if needed
    while (static_cast<int>(bias_row.size()) < cols) {
        bias_row.emplace_back(0.0);
    }

    // Repeat the row `rows` times
    bias_matrix.assign(rows, bias_row);
}

std::vector<std::vector<double>> readGroundTruth(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid number: " << val << std::endl;
                continue;
            }
        }
        if (!row.empty())
            matrix.push_back(row);
    }

    return matrix;
}

}  // namespace FIDESlib::CKKS
