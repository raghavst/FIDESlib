#include "CKKS/LinearTransform.cuh"
#include "MatMul.cuh"

namespace FIDESlib::CKKS {

#if LOW_MEM

void CCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    // Carlos A.D. TODO: Equalize levels (+ noise degree) of inputs to reduce unnecessary complexity in computations

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);
    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cc);
    linearTransform1.copy(cMat1);
    linearTransform1.dropToLevel(
        std::min(cMat1.getLevel() + cMat1.NoiseLevel - 1, cMat2.getLevel() + cMat1.NoiseLevel - 1 + 1));
    linearTransform2.copy(cMat2);
    linearTransform2.dropToLevel(
        std::min(cMat1.getLevel() + cMat1.NoiseLevel - 1 - 1, cMat2.getLevel() + cMat1.NoiseLevel - 1));

    assert(cMat1.getLevel() - cMat1.NoiseLevel + 1 >= 3);
    assert(cMat2.getLevel() - cMat2.NoiseLevel + 1 >= 2);
    assert(precomp.pts_1[0]->c0.getLevel() >= cMat1.c0.getLevel() - cMat1.NoiseLevel + 1);
    assert(precomp.pts_2[0]->c0.getLevel() >= cMat2.c0.getLevel() - cMat2.NoiseLevel + 1);
    assert(precomp.pts_3_1[0]->c0.getLevel() >= cMat1.c0.getLevel() - cMat1.NoiseLevel);

#if BSGS
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);
#else
    linearTransform1.multPt(cMat1, precomp.sigmaPlaintexts[0][0]);

    FIDESlib::CKKS::Ciphertext prev_rotation(cc);
    FIDESlib::CKKS::Ciphertext productCt(cc);

    prev_rotation.copy(cMat1);
    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(1, cc.GetRotationKey(1));

        productCt.multPt(prev_rotation, precomp.sigmaPlaintexts[0][i]);
        linearTransform1.add(productCt);
    }

    prev_rotation.copy(cMat1);
    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(-1, cc.GetRotationKey(-1));

        productCt.multPt(prev_rotation, precomp.sigmaPlaintexts[1][i - 1]);
        linearTransform1.add(productCt);
    }
#endif

#if BSGS

    LinearTransform(linearTransform2, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);
#else
    linearTransform2.multPt(cMat2, precomp.tauPlaintexts[0]);

    prev_rotation.copy(cMat2);

    for (size_t i = 1; i < rowSize; i++) {
        prev_rotation.rotate(rowSize, cc.GetRotationKey(rowSize));

        productCt.multPt(prev_rotation, precomp.tauPlaintexts[i /* * rowSize */]);
        linearTransform2.add(productCt);
    }
#endif

    // cProduct.copy(linearTransform1);
    //
    // Steps 2 and 3: Initial computation

#if BSGS
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    LinearTransformSpecial(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                           precomp.pts_3_2, 1, rowSize);
    cProduct.copy(linearTransform1);
#else
    cProduct.mult(linearTransform1, linearTransform2, cc.GetEvalKey());
    prev_rotation.copy(linearTransform1);
    FIDESlib::CKKS::Ciphertext prev_rotation_minus_row(cc);
    prev_rotation_minus_row.rotate(prev_rotation, -(int)rowSize, cc.GetRotationKey(-(int)rowSize));
    FIDESlib::CKKS::Ciphertext productCt2(cc);

    FIDESlib::CKKS::Ciphertext prev_rotation2(cc);
    prev_rotation2.copy(linearTransform2);

    for (size_t i = 1; i < rowSize; i++) {
        // Step 2
        prev_rotation.rotate(1, linearTransform1.cc.GetRotationKey(1));

        productCt.multPt(prev_rotation, precomp.phiPlaintexts[i][0]);

        prev_rotation_minus_row.rotate(1, linearTransform1.cc.GetRotationKey(1));

        productCt2.multPt(prev_rotation_minus_row, precomp.phiPlaintexts[i][1]);

        productCt.add(productCt2);

        prev_rotation2.rotate(rowSize, cc.GetRotationKey(rowSize));

        // Step 3
        productCt.mult(prev_rotation2, cc.GetEvalKey());
        cProduct.add(productCt);
    }

#endif
}

#else

void CCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, FIDESlib::CKKS::Ciphertext& cMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cMat1.cc);
    linearTransform1.multPt(cMat1, precomp.sigmaPlaintexts[0][0]);

    for (size_t i = 1; i < rowSize; i++) {
        FIDESlib::CKKS::Ciphertext rotatedCt(cMat1.cc);
        rotatedCt.copy(cMat1);
        rotatedCt.rotate(i, cMat1.cc.GetRotationKey(i));

        FIDESlib::CKKS::Ciphertext productCt(rotatedCt.cc);
        productCt.multPt(rotatedCt, precomp.sigmaPlaintexts[0][i]);
        linearTransform1.add(productCt);

        rotatedCt.copy(cMat1);
        rotatedCt.rotate(-i, cMat1.cc.GetRotationKey(-i));

        productCt.multPt(rotatedCt, precomp.sigmaPlaintexts[0][rowSize * rowSize - i]);
        linearTransform1.add(productCt);
    }

    // Step 1-2: Linear transform for second matrix
    FIDESlib::CKKS::Ciphertext linearTransform2(cMat2.cc);
    linearTransform2.multPt(cMat2, precomp.tauPlaintexts[0]);

    for (size_t i = 1; i < rowSize; i++) {
        FIDESlib::CKKS::Ciphertext rotatedCt(cMat2.cc);
        rotatedCt.copy(cMat2);
        rotatedCt.rotate(i * rowSize, cMat2.cc.GetRotationKey(i * rowSize));

        FIDESlib::CKKS::Ciphertext productCt(rotatedCt.cc);
        productCt.multPt(rotatedCt, precomp.tauPlaintexts[i * rowSize]);
        linearTransform2.add(productCt);
    }

    // Steps 2 and 3: Initial computation
    cProduct.mult(linearTransform1, linearTransform2, linearTransform2.cc.GetEvalKey());

    for (size_t i = 1; i < rowSize; i++) {
        // Step 2
        FIDESlib::CKKS::Ciphertext rotatedCt(linearTransform1.cc);
        rotatedCt.copy(linearTransform1);
        rotatedCt.rotate(i, linearTransform1.cc.GetRotationKey(i));

        FIDESlib::CKKS::Ciphertext productCt1(rotatedCt.cc);
        productCt1.multPt(rotatedCt, precomp.phiPlaintexts[i][0]);

        rotatedCt.copy(linearTransform1);
        rotatedCt.rotate(i - rowSize, linearTransform1.cc.GetRotationKey(i - rowSize));

        FIDESlib::CKKS::Ciphertext productCt2(rotatedCt.cc);
        productCt2.multPt(rotatedCt, precomp.phiPlaintexts[i][1]);

        FIDESlib::CKKS::Ciphertext linearTransformPhi(productCt1.cc);
        linearTransformPhi.add(productCt1, productCt2);

        FIDESlib::CKKS::Ciphertext linearTransformPsi(linearTransform2.cc);
        linearTransformPsi.copy(linearTransform2);
        linearTransformPsi.rotate(i * rowSize, linearTransform2.cc.GetRotationKey(i * rowSize));

        // Step 3
        FIDESlib::CKKS::Ciphertext tempProduct(linearTransformPhi.cc);
        tempProduct.mult(linearTransformPhi, linearTransformPsi, linearTransformPsi.cc.GetEvalKey());
        cProduct.add(tempProduct);
    }
}

#endif

// matrix multiplication
void CCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp) {

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());

        for (size_t j = 0; j < matrix2[0].size(); j++) {

            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);

            // auto start_gpu = std::chrono::high_resolution_clock::now();
            CCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            // auto end_gpu = std::chrono::high_resolution_clock::now();
            // std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
            // << " ms." << std::endl;

            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);

                // auto start_gpu = std::chrono::high_resolution_clock::now();
                CCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                // auto end_gpu = std::chrono::high_resolution_clock::now();
                // std::cout << "took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                // << " ms." << std::endl;

                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }
}

// matrix multiplication with bias
void CCMMwBias_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2, uint32_t rowSize,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
                   const MatrixMatrixProductPrecomputations_GPU& precomp,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& bias) {

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            CCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            dotProd.add(bias[0][j]);

            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                CCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProdNew.add(bias[i][j]);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }
}

void PCMMSquare_GPU(FIDESlib::CKKS::Ciphertext& cMat1, const FIDESlib::CKKS::Plaintext& pMat2, uint32_t rowSize,
                    FIDESlib::CKKS::Ciphertext& cProduct, const MatrixMatrixProductPrecomputations_GPU& precomp) {

    FIDESlib::CKKS::Context& cc = cMat1.cc;
    // Step 1-1: Linear transform for first matrix
    FIDESlib::CKKS::Ciphertext linearTransform1(cc);

    FIDESlib::CKKS::Plaintext linearTransform2(cc);
    linearTransform1.copy(cMat1);
    linearTransform1.dropToLevel(
        std::min(cMat1.getLevel() + cMat1.NoiseLevel - 1, pMat2.c0.getLevel() + cMat1.NoiseLevel - 1 + 1));
    linearTransform2.copy(pMat2);

    assert(cMat1.getLevel() - cMat1.NoiseLevel + 1 >= 3);
    assert(pMat2.c0.getLevel() - pMat2.NoiseLevel + 1 >= 2);
    assert(precomp.pts_1[0]->c0.getLevel() >= cMat1.c0.getLevel() - cMat1.NoiseLevel + 1);
    assert(precomp.pts_3_1[0]->c0.getLevel() >= cMat1.c0.getLevel() - cMat1.NoiseLevel);
    linearTransform1.copy(cMat1);
    LinearTransform(linearTransform1, 2 * rowSize - 1, precomp.bStep, precomp.pts_1, 1, -(int)rowSize + 1);

    // Step 1-2: Linear transform for second matrix

    //LinearTransformPt(linearTransform2, cc, rowSize, precomp.bStep, precomp.pts_2, rowSize, 0);

    // Steps 2 and 3: Initial computation
    Ciphertext aux(cc);
    aux.rotate(linearTransform1, -(int)rowSize /*+ 1*/, cc.GetRotationKey(-(int)rowSize /*+ 1*/));
    LinearTransformSpecialPt(linearTransform1, aux, linearTransform2, rowSize, precomp.bStep, precomp.pts_3_1,
                             precomp.pts_3_2, 1,
                             rowSize);  // LT2 is now plaintext
    if (linearTransform1.NoiseLevel == 2)
        linearTransform1.rescale();
    linearTransform1.dropToLevel(cMat1.getLevel() - 3);
    cProduct.copy(linearTransform1);
}

// PCMM will always have bias
void PCMM_GPU(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix1,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& matrix2, uint32_t rowSize,
              std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& product,
              const MatrixMatrixProductPrecomputations_GPU& precomp,
              std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias) {

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix2[0].size());
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext dotProd(matrix1[i][0].cc);
            PCMMSquare_GPU(matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);
            dotProd.addPt(bias[0][j]);
            for (size_t k = 1; k < matrix2.size(); k++) {
                FIDESlib::CKKS::Ciphertext dotProdNew(matrix1[i][k].cc);
                PCMMSquare_GPU(matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                dotProd.add(dotProdNew);
            }
            row.emplace_back(std::move(dotProd));
        }
        product.emplace_back(std::move(row));
    }
}

// Function to convert lbcrypto precomputations to GPU precomputations
struct MatrixMatrixProductPrecomputations_GPU convertToGPUPrecomputations(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
    const MatrixMatrixProductPrecomputations& cpuPrecomp, int bStep, const int levelCP, const int levelCC,
    bool fuse_boot_prescale_CCMM, const int slots) {

    struct MatrixMatrixProductPrecomputations_GPU gpuPrecomp;
    gpuPrecomp.rowSize = cpuPrecomp.rowSize;

#if BSGS
    gpuPrecomp.bStep = bStep;
#endif

    gpuPrecomp.sigmaPlaintexts.resize(2);
    // Convert sigma plaintexts
    {
        auto pt_rots =
            GetLinearTransformPlaintextRotationIndices(2 * gpuPrecomp.rowSize - 1, bStep, 1, -gpuPrecomp.rowSize + 1);

        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {

            const auto& sigmaPt_ = cpuPrecomp.sigmaPlaintexts[i];
#if BSGS
#if !EXT
            auto sigmaPt = context->MakeCKKSPackedPlaintext(
                Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[i + gpuPrecomp.rowSize - 1]), 1, GPUcc.L - levelCP);
#else
            auto sigmaPt =
                encodeExt(context, Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[i + gpuPrecomp.rowSize - 1]), 1,
                          GPUcc.L - level);
#endif
#endif
            auto raw_sigma = FIDESlib::CKKS::GetRawPlainText(context, sigmaPt);
            FIDESlib::CKKS::Plaintext sigma_gpu(GPUcc, raw_sigma);
            gpuPrecomp.sigmaPlaintexts[0].emplace_back(std::move(sigma_gpu));
        }

        for (int i = 0; i < gpuPrecomp.rowSize - 1; ++i) {
            const auto& sigmaPt_ = cpuPrecomp.sigmaPlaintexts[gpuPrecomp.rowSize * gpuPrecomp.rowSize - 1 - i];
#if BSGS
#if !EXT
            auto sigmaPt = context->MakeCKKSPackedPlaintext(
                Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[gpuPrecomp.rowSize - 2 - i]), 1, GPUcc.L - levelCP);
#else

            auto sigmaPt =
                encodeExt(context, Rotate(sigmaPt_->GetCKKSPackedValue(), pt_rots[gpuPrecomp.rowSize - 2 - i]), 1,
                          GPUcc.L - level + 1);
#endif
#endif
            auto raw_sigma = FIDESlib::CKKS::GetRawPlainText(context, sigmaPt);
            FIDESlib::CKKS::Plaintext sigma_gpu(GPUcc, raw_sigma);
            gpuPrecomp.sigmaPlaintexts[1].emplace_back(std::move(sigma_gpu));
        }
#if BSGS
        gpuPrecomp.pts_1.resize(gpuPrecomp.rowSize * 2 - 1);
        for (int i = 0; i < gpuPrecomp.rowSize - 1; ++i) {
            gpuPrecomp.pts_1[gpuPrecomp.rowSize - 2 - i] = &gpuPrecomp.sigmaPlaintexts[1][i];
        }
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_1[i + gpuPrecomp.rowSize - 1] = &gpuPrecomp.sigmaPlaintexts[0][i];
        }
#endif
    }
    {
        auto pt_rots = GetLinearTransformPlaintextRotationIndices(gpuPrecomp.rowSize, bStep, gpuPrecomp.rowSize);

        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            const auto& tauPt_ = cpuPrecomp.tauPlaintexts[i * gpuPrecomp.rowSize];
            auto values = tauPt_->GetCKKSPackedValue();
            if (fuse_boot_prescale_CCMM) {
                double scale = FIDESlib::CKKS::GetPreScaleFactor(GPUcc, slots);
                //std::cout << scale << std::endl;
                for (auto& k : values)
                    k *= scale;
            } else {
                // for (auto& k : values)
                //     k *= 0.001;
            }

#if BSGS
#if !EXT
            auto tauPt = context->MakeCKKSPackedPlaintext(Rotate(values, pt_rots[i]), 1, GPUcc.L - levelCC + 1);
#else

            auto tauPt = encodeExt(context, Rotate(values, pt_rots[i]), 1, GPUcc.L - level);
#endif
#endif
            {
                auto raw_tau = FIDESlib::CKKS::GetRawPlainText(context, tauPt);
                FIDESlib::CKKS::Plaintext tau_gpu(GPUcc, raw_tau);
                gpuPrecomp.tauPlaintexts.emplace_back(std::move(tau_gpu));
            }
        }

#if BSGS
        gpuPrecomp.pts_2.resize(gpuPrecomp.rowSize);
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_2[i] = &gpuPrecomp.tauPlaintexts[i];
        }
#endif
    }

    {
        auto pt_rots = GetLinearTransformPlaintextRotationIndices(gpuPrecomp.rowSize, bStep, 1, 0);
        // Convert phi plaintexts

        int i = 0;
        for (const auto& phiVec : cpuPrecomp.phiPlaintexts) {
            std::vector<FIDESlib::CKKS::Plaintext> phiGpuVec;
            int j = 0;
            for (const auto& phiPt_ : phiVec) {
                const auto& tauPt_ = cpuPrecomp.tauPlaintexts[i * gpuPrecomp.rowSize];
                auto values = phiPt_->GetCKKSPackedValue();
#if BSGS
#if !EXT
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(values, pt_rots[i]), 1, GPUcc.L - levelCP + 1);
#else
                auto phiPt = context->MakeCKKSPackedPlaintext(Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1,
                                                              GPUcc.L - level);
                //        auto phiPt = encodeExt(context, Rotate(phiPt_->GetCKKSPackedValue(), pt_rots[i]), 1, GPUcc.L - level);
#endif
#endif
                auto raw_phi = FIDESlib::CKKS::GetRawPlainText(context, phiPt);
                FIDESlib::CKKS::Plaintext phi_gpu(GPUcc, raw_phi);
                phiGpuVec.emplace_back(std::move(phi_gpu));
                ++j;
            }
            gpuPrecomp.phiPlaintexts.emplace_back(std::move(phiGpuVec));
            ++i;
        }

#if BSGS
        gpuPrecomp.pts_3_2.resize(gpuPrecomp.rowSize);
        for (int i = 1; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_2[i] = &gpuPrecomp.phiPlaintexts[i][1];
        }
        gpuPrecomp.pts_3_1.resize(gpuPrecomp.rowSize);
        for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
            gpuPrecomp.pts_3_1[i] = &gpuPrecomp.phiPlaintexts[i][0];
        }
#endif
    }
    //gpuPrecomp.tauVectors.resize(gpuPrecomp.rowSize);
    //for (int i = 0; i < gpuPrecomp.rowSize; ++i) {
    //        gpuPrecomp.tauVectors[i] = cpuPrecomp.tauVectors[i];
    //    }

    return gpuPrecomp;
}

// Direct GPU version of getMatrixMatrixProductPrecomputations
struct MatrixMatrixProductPrecomputations_GPU getMatrixMatrixProductPrecomputations_GPU(
    FIDESlib::CKKS::Context& GPUcc, lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize, int bStep,
    const int levelCP, const int levelCC, const bool fuse_boot_prescale_CCMM, const int slots) {

    // First get CPU precomputations
    MatrixMatrixProductPrecomputations cpuPrecomp = getMatrixMatrixProductPrecomputations(context, rowSize);

    // Then convert to GPU precomputations
    return convertToGPUPrecomputations(GPUcc, context, cpuPrecomp, bStep, levelCP, levelCC, fuse_boot_prescale_CCMM,
                                       slots);
}

std::vector<double> getPCMM_bMatrix(std::vector<double> weights, int rowSize) {
    int slots = weights.size();
    std::vector<double> data(slots, 0.0);
    for (int j = 0; j < rowSize; ++j) {
        for (int i = 0; i < slots / rowSize; ++i) {
            data[rowSize * i + j] = weights[(rowSize * (slots / rowSize + i + j) + j) % slots];
        }
    }
    return data;
}

std::vector<int> GenerateMatMulRotationIndices_GPU(int rowSize, int bStep) {
    std::set<int> indices;
#if LOW_MEM

#if BSGS
    std::vector<int> aux = GetLinearTransformRotationIndices(bStep, 1, -rowSize + 1);
    std::vector<int> aux2 = GetLinearTransformRotationIndices(bStep, rowSize, 0);
    for (auto& i : aux)
        indices.insert(i);
    for (auto& i : aux2)
        indices.insert(i);
    indices.insert(-rowSize);

    // For the special transform for steps 2 and 3
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);
    if (gStep - 1 != 0 && rowSize - 1 != 0)
        indices.insert((gStep - 1) * bStep * (rowSize - 1));
    if (rowSize - 1 != 0)
        indices.insert(-bStep * (rowSize - 1));
#else
    indices.insert(1);
    indices.insert(-1);
    indices.insert(rowSize);
    indices.insert(-rowSize);
#endif

#else
    for (size_t i = 1; i < rowSize; i++) {
        indices.insert(i);
        indices.insert(-i);
        indices.insert(i * rowSize);
        indices.insert(i - rowSize);
    }
#endif

    std::vector<int> indicesList(indices.begin(), indices.end());
    return indicesList;
}

FIDESlib::CKKS::Ciphertext rotsum_GPU(FIDESlib::CKKS::Ciphertext& in, int blockSize, int padding) {
    { std::cout << "Replace RotSum with CKKS::Accumulate " << std::endl; }
    Context& cc = in.cc;

    FIDESlib::CKKS::Ciphertext prev_rotation(cc);
    prev_rotation.copy(in);

    for (int i = 0; i < std::log2(blockSize); ++i) {
        int rot_index = padding * (1 << i);
        FIDESlib::CKKS::Ciphertext rotated(cc);
        rotated.copy(prev_rotation);
        rotated.rotate(rot_index, cc.GetRotationKey(rot_index), true);
        prev_rotation.add(rotated);
    }

    return prev_rotation;
}

}  // namespace FIDESlib::CKKS