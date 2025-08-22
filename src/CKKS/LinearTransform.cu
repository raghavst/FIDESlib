//
// Created by carlosad on 7/05/25.
//
#include <source_location>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include "CudaUtils.cuh"

void FIDESlib::CKKS::LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, int stride,
                                     int offset) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts.size() >= rowSize);
    for (auto i : pts) {
        assert(i != nullptr);
    }
    Context& cc = ctxt.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();

    std::vector<Ciphertext>& fastRotation = cc.getBootstrapAuxilarCiphertexts();

    for (int i = fastRotation.size(); i < bStep; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);
        keys.push_back(i == 0 ? nullptr : &cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    bool ext = true;
    if (bStep == 1)
        ext = false;
    for (auto& i : pts) {
        if (!i->c0.isModUp()) {
            ext = false;
        }
    }

    ctxt.rotate_hoisted(keys, indexes, fastRotationPtr, ext);

    Ciphertext inner(cc);

    for (uint32_t j = gStep - 1; j < gStep; --j) {
        int n = 1;
        //inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                n++;
                //inner.addMultPt(fastRotation[i - 1], A[bStep * j + i], false);
            }
        }

        if (fastRotation[0].getLevel() > inner.getLevel()) {
            inner.c0.grow(fastRotation[0].getLevel(), true);
            inner.c1.grow(fastRotation[0].getLevel(), true);
        } else {
            inner.dropToLevel(fastRotation[0].getLevel());
        }

        inner.dotProductPt(fastRotation.data(), (const Plaintext**)pts.data() + j * bStep, n, ext);

        if (j == gStep - 1) {
            ctxt.copy(inner);
        } else {
            if (!ext)
                inner.extend();
            ctxt.add(inner);
            ctxt.modDown(false);
        }

        if (j > 0) {
            ctxt.rotate((int)stride * bStep, cc.GetRotationKey((int)stride * bStep), false);
        }
    }

    if (offset != 0) {
        ctxt.rotate(offset, cc.GetRotationKey(offset), true);
    }
}

void FIDESlib::CKKS::LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep,
                                            std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                            int stride3) {
    constexpr bool PRINT = false;
    if constexpr (PRINT)
        std::cout << std::endl << "LinearTransformSpecial ";

    if constexpr (PRINT) {
        std::cout << "ctxt1 ";
        for (auto& j : ctxt1.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt1.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    if constexpr (PRINT) {
        std::cout << "ctxt2 ";
        for (auto& j : ctxt2.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt2.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    if constexpr (PRINT) {
        std::cout << "ctxt3 ";
        for (auto& j : ctxt3.c0.GPU)
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
        for (auto& j : ctxt3.c0.GPU)
            for (auto& k : j.SPECIALlimb) {
                SWITCH(k, printThisLimb(1));
            }
        std::cout << std::endl;
    }

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();
    if (ctxt3.NoiseLevel == 2)
        ctxt3.rescale();

    std::vector<Ciphertext>& fastRotation = cc.getBootstrapAuxilarCiphertexts();

    for (int i = fastRotation.size(); i < (bStep - 1) * 3; ++i)
        fastRotation.emplace_back(cc);

    /*
    std::vector<Ciphertext> fastRotation2;
    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i) {
        fastRotation2.emplace_back(cc);
        fastRotation1.emplace_back(cc);
    }
    */

    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation[i - 1]);
        fastRotationPtr2.push_back(&fastRotation[(bStep - 1) + i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    /*
    std::vector<Ciphertext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);
*/
    std::vector<Ciphertext*> fastRotationPtr3;
    std::vector<int> indexes3;
    std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation[2 * (bStep - 1) + i - 1]);
        keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        ctxt3.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)), true);

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        inner.multPt(ctxt1, *pts1[bStep * j], false);
        if (bStep * j != 0)
            inner.addMultPt(ctxt2, *pts2[bStep * j], true);
        inner.mult(ctxt3, cc.GetEvalKey(), false, false);

        if constexpr (PRINT) {
            std::cout << "inner " << bStep * j << " ";
            for (auto& j : inner.c0.GPU)
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            std::cout << std::endl;
            for (auto& j : inner.c0.GPU)
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            std::cout << std::endl;
        }

        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        auto keys3_ = keys3;
                        auto indexes3_ = indexes3;
                        auto fastRotationPtr3_ = fastRotationPtr3;

                        keys3_.resize(size);
                        indexes3_.resize(size);
                        fastRotationPtr3_.resize(size);

                        ctxt3.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3_, false);
                    } else {
                        ctxt3.rotate_hoisted(keys3, indexes3, fastRotationPtr3, false);
                    }
                }
                aux.multPt(fastRotation[i - 1], *pts1[bStep * j + i], false);
                aux.addMultPt(fastRotation[(bStep - 1) + i - 1], *pts2[bStep * j + i], true);
                aux.mult(fastRotation[2 * (bStep - 1) + i - 1], cc.GetEvalKey(), false, false);
                if constexpr (PRINT) {
                    std::cout << "inner " << bStep * j + i << " ";
                    for (auto& j : aux.c0.GPU)
                        for (auto& k : j.limb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    std::cout << std::endl;
                    for (auto& j : aux.c0.GPU)
                        for (auto& k : j.SPECIALlimb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    std::cout << std::endl;
                }
                inner.add(aux);
            }
        }

        if (j == gStep - 1) {
            result.copy(inner);
        } else {
            result.add(inner);
        }
        result.modDown(false);
        if (j > 0) {
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep), false);
            ctxt3.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)), true);
        }
    }

    ctxt1.copy(result);
    CudaCheckErrorModNoSync;
}

std::vector<int> FIDESlib::CKKS::GetLinearTransformRotationIndices(int bStep, int stride, int offset) {
    std::vector<int> res(bStep + (offset != 0));
    for (int i = 1; i <= bStep; ++i)
        res[i - 1] = i * stride;
    if (offset != 0)
        res[bStep] = offset;
    return res;
}

std::vector<int> FIDESlib::CKKS::GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride,
                                                                            int offset) {
    std::vector<int> res(rowSize);
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    for (int j = 0; j < gStep; j++) {
        for (int i = 0; i < bStep; ++i) {
            if (i + j * bStep < rowSize)
                res[i + j * bStep] = -bStep * j * stride - offset;
        }
    }
    return res;
}
/*
void FIDESlib::CKKS::LinearTransformPt(FIDESlib::CKKS::Plaintext& ptxt, FIDESlib::CKKS::Context& cc, int rowSize,
                                       int bStep, std::vector<Plaintext*> pts, int stride, int offset) {

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    assert(pts.size() >= rowSize);
    for (auto i : pts) {
        assert(i != nullptr);
    }

    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    std::vector<Plaintext> fastRotation;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Plaintext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;

    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }
    ptxt.rotate_hoisted(indexes, fastRotationPtr);

    Plaintext result(cc);
    Plaintext inner(cc);

    for (uint32_t j = gStep - 1; j < gStep; --j) {
        Plaintext temp(cc);
        temp.copy(ptxt);
        temp.multPt(*pts[bStep * j], false);
        inner.copy(temp);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < rowSize) {
                Plaintext tmp(cc);
                tmp.copy(fastRotation[i - 1]);
                tmp.multPt(*pts[(bStep * j + i)], false);
                inner.addPt(tmp);
            }
        }
        if (j > 0) {
            if (j == gStep - 1) {
                result.copy(inner);
            } else {
                Plaintext tmp(cc);
                tmp.copy(result);
                inner.addPt(tmp);
                result.copy(inner);
                // result.addPt(inner); // the d-tour here is due to the level adjustment logic in the RNSPoly structure
            }
            result.automorph(stride * bStep);
        } else {
            if (gStep == 1) {
                result.copy(inner);
            } else {
                Plaintext tmp(cc);
                tmp.copy(result);
                inner.addPt(tmp);
                result.copy(inner);
                // result.addPt(inner); // the d-tour here is due to the level adjustment logic in the RNSPoly structure
            }
        }
    }
    if (offset != 0) {
        result.automorph(offset);
    }
    result.rescale();
    ptxt.copy(result);
    cudaDeviceSynchronize();
}
*/

void FIDESlib::CKKS::LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep,
                                              std::vector<Plaintext*> pts1, std::vector<Plaintext*> pts2, int stride,
                                              int stride3) {

    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    assert(pts1.size() >= rowSize);
    for (auto i : pts1) {
        assert(i != nullptr);
    }
    assert(pts2.size() >= rowSize);

    Context& cc = ctxt1.cc;
    uint32_t gStep = ceil(static_cast<double>(rowSize) / bStep);

    if (ctxt1.NoiseLevel == 2)
        ctxt1.rescale();
    if (ctxt2.NoiseLevel == 2)
        ctxt2.rescale();

    std::vector<Ciphertext>& fastRotation = cc.getBootstrapAuxilarCiphertexts();

    for (int i = fastRotation.size(); i < (bStep - 1) * 2; ++i)
        fastRotation.emplace_back(cc);

    /*
    std::vector<Ciphertext> fastRotation1;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation1.emplace_back(cc);
    std::vector<Ciphertext> fastRotation2;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation2.emplace_back(cc);
*/
    std::vector<Ciphertext*> fastRotationPtr1;
    std::vector<Ciphertext*> fastRotationPtr2;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr1.push_back(&fastRotation[i - 1]);
        fastRotationPtr2.push_back(&fastRotation[(bStep - 1) + i - 1]);
        keys.push_back(&cc.GetRotationKey(i * stride));
        indexes.push_back(i * stride);
    }

    ctxt1.rotate_hoisted(keys, indexes, fastRotationPtr1, false);
    ctxt2.rotate_hoisted(keys, indexes, fastRotationPtr2, false);

    std::vector<Plaintext> fastRotation3;
    for (int i = 0; i < bStep - 1; ++i)
        fastRotation3.emplace_back(cc);

    std::vector<Plaintext*> fastRotationPtr3;
    std::vector<int> indexes3;
    //std::vector<KeySwitchingKey*> keys3;
    for (int i = 1; i < bStep; ++i) {
        fastRotationPtr3.push_back(&fastRotation3[i - 1]);
        //keys3.push_back(&cc.GetRotationKey(i * stride3));
        indexes3.push_back(i * stride3);
    }

    Ciphertext result(cc);
    Ciphertext inner(cc);
    Ciphertext aux(cc);

    if ((gStep - 1) * bStep * (rowSize - 1) != 0)
        // ctxt2.rotate((gStep - 1) * bStep * (rowSize - 1), cc.GetRotationKey((gStep - 1) * bStep * (rowSize - 1)));
        ptxt.automorph((gStep - 1) * bStep * (rowSize - 1));

    for (uint32_t j = gStep - 1; j < gStep; --j) {

        inner.multPt(ctxt1, *pts1[bStep * j], false);
        if (bStep * j != 0)
            inner.addMultPt(ctxt2, *pts2[bStep * j], false);

        inner.rescale();
        inner.multPt(ptxt, false);
        for (uint32_t i = 1; i < bStep; i++) {

            if (bStep * j + i < rowSize) {

                if (i == 1) {
                    int size = std::min((int)bStep - 1, (int)(rowSize - (bStep * j + i)));
                    if (size < bStep - 1) {
                        //auto keys3_ = keys3;
                        auto indexes3_ = indexes3;

                        //keys3_.resize(size);
                        indexes3_.resize(size);
                        // ctxt2.rotate_hoisted(keys3_, indexes3_, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3_, fastRotationPtr3);
                    } else {
                        // ctxt2.rotate_hoisted(keys3, indexes3, fastRotationPtr3);
                        ptxt.rotate_hoisted(indexes3, fastRotationPtr3);
                    }
                }
                aux.multPt(fastRotation[i - 1], *pts1[bStep * j + i], false);
                aux.addMultPt(fastRotation[(bStep - 1) + i - 1], *pts2[bStep * j + i], false);
                aux.rescale();
                aux.multPt(fastRotation3[i - 1], false);

                inner.add(aux);
            }
        }

        if (j == gStep - 1) {
            result.copy(inner);
        } else {
            result.add(inner);
        }
        if (j > 0) {
            result.rotate(stride * bStep, cc.GetRotationKey(stride * bStep), true);
            // ctxt2.rotate(-bStep * (rowSize - 1), cc.GetRotationKey(-bStep * (rowSize - 1)));
            ptxt.automorph(-bStep * (rowSize - 1));
        }
    }

    ctxt1.copy(result);
    cudaDeviceSynchronize();
}
