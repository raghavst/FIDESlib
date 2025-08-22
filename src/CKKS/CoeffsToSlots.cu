//
// Created by carlosad on 27/11/24.
//

#include <ranges>
#include <vector>
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"

using namespace FIDESlib::CKKS;

#if AFFINE_LT

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    constexpr bool PRINT = false;

    Context& cc = ctxt.cc;
    // Computing the baby-step bStep and the giant-step gStep.
    uint32_t bStep = cc.GetBootPrecomputation(slots).LT.bStep;
    uint32_t gStep = ceil(static_cast<double>(slots) / bStep);

    uint32_t M = cc.N * 2;
    uint32_t N = cc.N;

    // computes the NTTs for each CRT limb (for the hoisted automorphisms used
    // later on)
    //auto digits = cc->EvalFastRotationPrecompute(ct);
    std::vector<Ciphertext>& fastRotation = cc.getBootstrapAuxilarCiphertexts();

    for (int i = fastRotation.size(); i < bStep; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);
        keys.push_back(i == 0 ? nullptr : &cc.GetRotationKey(i));
        indexes.push_back(i);
    }

    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Input LT ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }

    bool ext = true;
    if (bStep == 1)
        ext = false;
    for (auto& i : decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A) {
        if (!i.c0.isModUp()) {
            ext = false;
        }
    }

    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        for (int i = 0; i < bStep; ++i) {
            std::cout << "In hoistRotation ";
            for (auto& j : fastRotation[i].c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : fastRotation[i].c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }
        cudaDeviceSynchronize();
    }

    ctxt.rotate_hoisted(keys, indexes, fastRotationPtr, ext);
    /*
    fastRotationPtr[0]->modDown(false);
    ctxt.copy(*fastRotationPtr[0]);
    return;
*/
    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        for (int i = 0; i < bStep; ++i) {
            std::cout << "Out hoistRotation ";
            for (auto& j : fastRotation[i].c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : fastRotation[i].c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
        }
        cudaDeviceSynchronize();
    }
    Ciphertext inner(cc);
    std::vector<Plaintext>& A = decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A;

    for (uint32_t j = gStep - 1; j < gStep; --j) {
        int n = 1;
        //inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < slots) {
                if constexpr (PRINT) {
                    cudaDeviceSynchronize();
                    std::cout << "input pt ";
                    for (auto& j : A[bStep * j + i].c0.GPU) {
                        cudaSetDevice(j.device);
                        for (auto& k : j.limb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    }
                    std::cout << std::endl;
                    for (auto& j : A[bStep * j + i].c0.GPU) {
                        cudaSetDevice(j.device);

                        for (auto& k : j.SPECIALlimb) {
                            SWITCH(k, printThisLimb(1));
                        }
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
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

        inner.dotProductPt(fastRotation.data(), A.data() + j * bStep, n, ext);

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "inner ";
            for (auto& j : inner.c0.GPU) {
                cudaSetDevice(j.device);

                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : inner.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        //if (ext)
        //    inner.modDown(false);
        if (j == gStep - 1) {
            ctxt.copy(inner);
        } else {
            ctxt.add(inner);
        }

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "ctxt add";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        if (ext)
            ctxt.modDown(false);

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "ctxt moddown";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
        if (j > 0) {
            ctxt.rotate((int)bStep, cc.GetRotationKey((int)bStep), !ext);
            //ctxt.rotate((int)bStep, cc.GetRotationKey((int)bStep), true);
        }

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "ctxt rotate";
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.limb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& k : j.SPECIALlimb) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }

    if constexpr (PRINT) {
        std::cout << "ctxt result";
        for (auto& j : inner.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& k : j.limb) {
                SWITCH(k, printThisLimb(1));
            }
        }
        std::cout << std::endl;
    }
}

#else

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});

    Context& cc = ctxt.cc;
    // Computing the baby-step bStep and the giant-step gStep.
    uint32_t bStep = cc.GetBootPrecomputation(slots).LT.bStep;
    uint32_t gStep = ceil(static_cast<double>(slots) / bStep);

    uint32_t M = cc.N * 2;
    uint32_t N = cc.N;

    // computes the NTTs for each CRT limb (for the hoisted automorphisms used
    // later on)
    //auto digits = cc->EvalFastRotationPrecompute(ct);

    std::vector<Ciphertext> fastRotation;

    for (int i = 0; i < bStep; ++i)
        fastRotation.emplace_back(cc);

    std::vector<Ciphertext*> fastRotationPtr;
    std::vector<int> indexes;
    std::vector<KeySwitchingKey*> keys;
    for (int i = 0; i < bStep; ++i) {
        fastRotationPtr.push_back(&fastRotation[i]);
        keys.push_back(i == 0 ? nullptr : &cc.GetRotationKey(i));
        indexes.push_back(i);
    }

    if (1) {
        ctxt.rotate_hoisted(keys, indexes, fastRotationPtr);
    } else {
        for (int i = 0; i < bStep - 1; ++i) {
            fastRotation[i].rotate(ctxt, i + 1, cc.GetRotationKey(i + 1));
            //cudaDeviceSynchronize();
        }
        //cudaDeviceSynchronize();
    }
    Ciphertext result(cc);
    Ciphertext inner(cc);
    std::vector<Plaintext>& A = decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A;

    for (uint32_t j = 0; j < gStep; j++) {

        int n = 1;
        //inner.multPt(ctxt, A[bStep * j], false);
        for (uint32_t i = 1; i < bStep; i++) {
            if (bStep * j + i < slots) {
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

        inner.dotProductPt(fastRotation.data(), A.data() + j * bStep, n);

        if (j == 0) {
            ctxt.copy(inner);
        } else {
            inner.rotate(bStep * j, cc.GetRotationKey(bStep * j));
            ctxt.add(inner);
        }
    }
}

#endif

void FIDESlib::CKKS::EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()});
    constexpr bool PRINT = false;
    Context& cc = ctxt.cc;

    if constexpr (PRINT) {
        cudaDeviceSynchronize();
        std::cout << "Input stc ";
        for (auto& j : ctxt.c0.GPU) {
            cudaSetDevice(j.device);
            for (auto& i : j.limb) {
                SWITCH(i, printThisLimb(1));
            }
        }
        std::cout << std::endl;
        cudaDeviceSynchronize();
    }
    //  No need for Encrypted Bit Reverse
    //Ciphertext& result = ctxt;
    // hoisted automorphisms
    if (ctxt.NoiseLevel == 2)
        ctxt.rescale();
    std::vector<Ciphertext>& auxiliar = cc.getBootstrapAuxilarCiphertexts();

    //Ciphertext outer(cc);
    Ciphertext inner(cc);

    int steps = 0;
    for (BootstrapPrecomputation::LTstep& step :
         (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS)) {
        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)

        std::vector<Ciphertext*> fastRotationPtr;
        std::vector<int> indexes;
        std::vector<KeySwitchingKey*> keys;

        for (int i = auxiliar.size(); i < step.bStep; ++i) {
            auxiliar.emplace_back(cc);
        }
        for (int i = 0; i < step.bStep; ++i) {
            fastRotationPtr.push_back(&auxiliar[i]);
            keys.push_back(step.rotIn[i] ? &cc.GetRotationKey(step.rotIn[i]) : nullptr);
            indexes.push_back(step.rotIn[i]);
        }

        bool ext = true;
        if (step.bStep == 1)
            ext = false;
        for (auto& i : step.A) {
            if (!i.c0.isModUp()) {
                ext = false;
            }
        }
        if (PRINT)
            std::cout << "Ext: " << ext << std::endl;

        ctxt.rotate_hoisted(keys, indexes, fastRotationPtr, ext);
        for (int i = step.gStep - 1; (int)i >= 0; --i) {

            // for the first iteration with j=0:
            int32_t G = step.bStep * i;
            //inner.multPt(auxiliar[0], step.A[G], false);
            // continue the loop
            int n = 1;
            for (int32_t j = 1; j < step.bStep; j++) {
                if ((G + j) != step.slots) {
                    n++;
                    // inner.addMultPt(auxiliar[j], step.A[G + j], false);
                }
            }
            if (auxiliar[0].getLevel() > inner.getLevel()) {
                inner.c0.grow(auxiliar[0].getLevel(), true);
                inner.c1.grow(auxiliar[0].getLevel(), true);
            } else {
                inner.dropToLevel(auxiliar[0].getLevel());
            }

            inner.dotProductPt(auxiliar.data(), step.A.data() + G, n, ext);

            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "inner Step: " << steps << " bStep: " << i << std::endl;
                for (auto& j : inner.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(1));
                    }
                }
                std::cout << std::endl;
                for (auto& j : inner.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.SPECIALlimb) {
                        SWITCH(i, printThisLimb(1));
                    }
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }

            constexpr bool baseline = false;
            if constexpr (baseline) {
                if (ext)
                    inner.modDown(false);
            }

            if (i == step.gStep - 1) {
                ctxt.copy(inner);
            } else {
                ctxt.add(inner);
            }

            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "ctxt add Step: " << steps << " bStep: " << i << std::endl;
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(1));
                    }
                }
                std::cout << std::endl;
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.SPECIALlimb) {
                        SWITCH(i, printThisLimb(1));
                    }
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }

            if (i > 0) {
                if (step.rotOut[1] - step.rotOut[0] != 0) {
                    if constexpr (!baseline) {
                        if (ext)
                            ctxt.modDown(false);
                    }
                    ctxt.rotate((int)(step.rotOut[1] - step.rotOut[0]),
                                cc.GetRotationKey((int)(step.rotOut[1] - step.rotOut[0])), baseline || !ext);
                }
            }
            if constexpr (PRINT) {
                cudaDeviceSynchronize();
                std::cout << "Step: " << steps << " bStep: " << i << std::endl;
                for (auto& j : ctxt.c0.GPU) {
                    cudaSetDevice(j.device);
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(1));
                    }
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
        }

        if (ctxt.c0.isModUp()) {
            ctxt.modDown();
        }
        if (step.rotOut[0] != 0) {
            ctxt.rotate(step.rotOut[0], cc.GetRotationKey(step.rotOut[0]), true);
        }
        steps++;
        if (steps != (decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS).size())
            ctxt.rescale();

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "Step: " << steps << std::endl;
            for (auto& j : ctxt.c0.GPU) {
                cudaSetDevice(j.device);
                for (auto& i : j.limb) {
                    SWITCH(i, printThisLimb(1));
                }
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }

    //CudaCheckErrorMod;
}
