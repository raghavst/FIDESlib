//
// Created by carlosad on 24/04/24.
//

#include "CKKS/Ciphertext.cuh"
#include <omp.h>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Plaintext.cuh"

namespace FIDESlib::CKKS {

constexpr bool RESCALE_DOUBLE = true;

enum OPS {
    NOP,
    ADD,
    ADDPT,
    MULT,
    MULTPT,
    RESCALE,
    ROTATE,
    COPY,
    SQUARE,
    ADDSCALAR,
    MULTSCALAR,
    ADDMULTPT,
    ADDMULTPTINS,
    WSUM,
    WSUMINPUTS,
    CONJUGATE,
    HOISTEDROTATE,
    HOISTEDROTATEOUTS,
};

constexpr std::array<const char*, 18> opstr{
    "                   Noop: ", "                   HAdd: ", "                  AddPt: ", "                   Mult: ",
    "                 MultPt: ",  // 5
    "                Rescale: ", "                 Rotate: ", "                   Copy: ", "                 Square: ",
    "              ScalarAdd: ",  // 10
    "             ScalarMult: ", "              AddMultPt: ", "     AddMultPt (inputs): ", "                   WSum: ",
    "          WSum (inputs): ",  // 15
    "              Conjugate: ", "          HoistedRotate: ", "HoistedRotate (outputs): "};

std::map<OPS, int> op_count;

Ciphertext::Ciphertext(Ciphertext&& ct_moved) noexcept
    : my_range(std::move(ct_moved.my_range)),
      cc(ct_moved.cc),
      c0(std::move(ct_moved.c0)),
      c1(std::move(ct_moved.c1)),
      NoiseFactor(ct_moved.NoiseFactor),
      NoiseLevel(ct_moved.NoiseLevel) {}

Ciphertext::Ciphertext(Context& cc)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc),
      c1(cc) {
    CudaNvtxStop();
}

Ciphertext::Ciphertext(Context& cc, const RawCipherText& rawct)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc, rawct.sub_0),
      c1(cc, rawct.sub_1) {
    NoiseLevel = rawct.NoiseLevel;
    NoiseFactor = rawct.Noise;
    CudaNvtxStop();
}

void Ciphertext::add(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForAddOrSub(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForAddOrSub(*this))
                add(b_);
            else
                assert(false);
            return;
        }
    }

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        assert(this->getLevel() == b.getLevel());
    } else if (getLevel() > b.getLevel()) {
        c0.dropToLevel(b.getLevel());
        c1.dropToLevel(b.getLevel());
    }
    op_count[OPS::ADD]++;

    c0.add(b.c0);
    c1.add(b.c1);
}

void Ciphertext::sub(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForAddOrSub(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForAddOrSub(*this))
                sub(b_);
            else
                assert(false);
            return;
        }
    }

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        assert(this->getLevel() == b.getLevel());
    } else if (getLevel() > b.getLevel()) {
        c0.dropToLevel(b.getLevel());
        c1.dropToLevel(b.getLevel());
    }
    op_count[OPS::ADD]++;

    c0.sub(b.c0);
    c1.sub(b.c1);
}

void Ciphertext::addPt(const Plaintext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (b.NoiseLevel == 1 && NoiseLevel == 2 && b.c0.getLevel() == getLevel() - 1) {
            this->rescale();
        }

        if (b.c0.getLevel() != this->getLevel()) {
            Plaintext b_(cc);
            if (!b_.adjustPlaintextToCiphertext(b, *this)) {
                assert(false);
            } else {
                addPt(b_);
            }
            return;
        }
    }
    assert(NoiseLevel == b.NoiseLevel);
    op_count[OPS::ADDPT]++;

    c0.add(b.c0);
    //NoiseFactor += b.NoiseFactor;
}

void Ciphertext::load(const RawCipherText& rawct) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.load(rawct.sub_0, rawct.moduli);
    c1.load(rawct.sub_1, rawct.moduli);

    NoiseLevel = rawct.NoiseLevel;
    NoiseFactor = rawct.Noise;
}

void Ciphertext::store(const Context& cc, RawCipherText& rawct) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    cudaDeviceSynchronize();
    rawct.numRes = c0.getLevel() + 1;
    rawct.sub_0.resize(rawct.numRes);
    rawct.sub_1.resize(rawct.numRes);
    c0.store(rawct.sub_0);
    c1.store(rawct.sub_1);
    rawct.N = cc.N;
    c0.sync();
    c1.sync();

    rawct.NoiseLevel = NoiseLevel;
    rawct.Noise = NoiseFactor;
    // TODO store other interesting metadata
    cudaDeviceSynchronize();
    CudaCheckErrorMod;
}

void Ciphertext::modDown(bool free) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.moddown(true, free);
    c1.moddown(true, free);
    if (free) {
        c0.freeSpecialLimbs();
        c1.freeSpecialLimbs();
    }
}

void Ciphertext::modUp() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.modup();
    //c1.modup();
}

void Ciphertext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    constexpr bool PRINT = false;
    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if constexpr (PRINT)
            std::cout << "multPt: Rescale input ciphertext" << std::endl;
        if (NoiseLevel == 2)
            this->rescale();
    }

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (b.c0.getLevel() != this->getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
            Plaintext b_(cc);
            if constexpr (PRINT)
                std::cout << "multPt: adjust input plaintext" << std::endl;

            //if (!this->adju)
            if (!b_.adjustPlaintextToCiphertext(b, *this)) {
                if constexpr (PRINT)
                    std::cout << "multPt: FAILED!" << std::endl;
                assert(false);
            } else {
                if (NoiseLevel == 2)
                    this->rescale();
                if (b_.NoiseLevel == 2) {
                    if constexpr (PRINT)
                        std::cout << "multPt: Rescale input plaintext" << std::endl;
                    b_.rescale();
                }
                multPt(b_, rescale);
            }
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    op_count[OPS::MULTPT]++;

    c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);
    c1.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}

void Ciphertext::rescale() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    //assert(this->NoiseLevel == 2);
    if (cc.rescaleTechnique != Context::FIXEDMANUAL) {
        // this wouldn't do anything in OpenFHE
    }
    op_count[OPS::RESCALE]++;

    if constexpr (RESCALE_DOUBLE) {
        c0.rescaleDouble(c1);
    } else {
        c0.rescale();
        c1.rescale();
    }

    // Manage metadata
    NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
    NoiseLevel -= 1;
}

/** "in" needs to have Digit and Gather limbs pre-generated */
RNSPoly& MGPUkeySwitchCore(RNSPoly& in, const KeySwitchingKey& kskEval, const bool moddown) {

    {
        RNSPoly& aux = in.modup_ksk_moddown_mgpu(kskEval, moddown);

        return aux;
    }
}

void Ciphertext::mult(const Ciphertext& b, const KeySwitchingKey& kskEval, bool rescale, const bool moddown) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (!adjustForMult(b)) {
            Ciphertext b_(cc);
            b_.copy(b);
            if (b_.adjustForMult(*this))
                mult(b_, kskEval, rescale, moddown);
            else
                assert(false);
            return;
        }
    }
    assert(NoiseLevel == 1);
    assert(NoiseLevel == b.NoiseLevel);
    op_count[OPS::MULT]++;
    /*
    if (getLevel() > b.getLevel()) {
        this->c0.dropToLevel(b.getLevel());
        this->c1.dropToLevel(b.getLevel());
    }
    */
    //assert(c0.getLevel() <= b.c0.getLevel());
    //assert(c1.getLevel() <= b.c1.getLevel());
    Out(KEYSWITCH, " start ");
    assert(this->NoiseLevel == 1);
    assert(b.NoiseLevel == 1);

    if (cc.GPUid.size() == 1) {
        if constexpr (0) {
            constexpr bool PRINT = true;
            bool SELECT = true;

            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cc.getKeySwitchAux().multElement(c1, b.c1);

            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU: " << 0 << "Input data ";
                    for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].limb.size(); ++j) {
                        std::cout << cc.getKeySwitchAux().GPU[0].meta[j].id;
                        SWITCH(cc.getKeySwitchAux().GPU[0].limb[j], printThisLimb(2));
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
            }

            cc.getKeySwitchAux().modup();

            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU: " << 0 << "Out ModUp after NTT ";
                    for (size_t j = 0; j < cc.getKeySwitchAux().GPU[0].DIGITlimb.size(); ++j) {
                        for (size_t i = 0; i < cc.getKeySwitchAux().GPU[0].DIGITlimb[j].size(); ++i) {
                            std::cout << cc.getKeySwitchAux().GPU[0].DIGITmeta[j][i].id;
                            SWITCH(cc.getKeySwitchAux().GPU[0].DIGITlimb[j][i], printThisLimb(2));
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
            }

            auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);

            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU out KSK specials: ";
                    for (const auto& j : {&aux0, &cc.getKeySwitchAux()}) {
                        for (const auto& k : j->GPU) {
                            for (auto& i : k.SPECIALlimb) {
                                SWITCH(i, printThisLimb(2));
                            }
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
            }
            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU out KSK limbs: ";
                    for (const auto& j : {&aux0, &cc.getKeySwitchAux()}) {
                        for (const auto& k : j->GPU) {
                            for (auto& i : k.limb) {
                                SWITCH(i, printThisLimb(2));
                            }
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
            }

            cc.getKeySwitchAux().moddown(true, false);
            aux0.moddown(true, false);
            c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());
            c0.mult1Add2(b.c0, aux0);
            //c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
            if (rescale) {
                this->rescale();
            }
            /*
            cudaDeviceSynchronize();
            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cudaDeviceSynchronize();
            cc.getKeySwitchAux().multElement(c1, b.c1);
            cudaDeviceSynchronize();
            cc.getKeySwitchAux().modup();
            cudaDeviceSynchronize();
            auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, c0.getLevel(), nullptr);
            cudaDeviceSynchronize();
            cc.getKeySwitchAux().moddown(true, false);
            cudaDeviceSynchronize();
            c1.mult1AddMult23Add4(b.c0, c0, b.c1, cc.getKeySwitchAux());  // Read 4 first for better cache locality.
            cudaDeviceSynchronize();
            aux0.moddown(true, false);
            cudaDeviceSynchronize();
            c0.mult1Add2(b.c0, aux0);
            cudaDeviceSynchronize();

            if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
                this->rescale();
            }
             */
        } else {
            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cc.getKeySwitchAux().multModupDotKSK(c1, b.c1, c0, b.c0, kskEval);
            {  // TODO MAD Figure 4: add before fused ModDown+Rescale
            }
            if (moddown)
                c1.moddown(true, false);
            if (moddown)
                c0.moddown(true, false);
            if (moddown && rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
                this->rescale();
        }
    } else {
        constexpr bool PRINT = false;

        if constexpr (PRINT)
            std::cout << "Init mult" << std::endl;
        RNSPoly& in = cc.getKeySwitchAux();
        in.setLevel(c1.getLevel());
        in.multElement(c1, b.c1);

        RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, moddown);

        if (moddown) {
            c1.mult1AddMult23Add4(b.c0, c0, b.c1, in);  // Read 4 first for better cache locality.
            c0.mult1Add2(b.c0, aux);
        } else {
            c1.multNoModdownEnd(c0, b.c0, b.c1, in, aux);
        }
        if (moddown && rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
            if constexpr (PRINT)
                std::cout << "Rescale c1" << std::endl;
            this->rescale();
        }
        if constexpr (PRINT)
            std::cout << "End mult" << std::endl;
        if constexpr (PRINT)
            CudaCheckErrorMod;
    }

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    Out(KEYSWITCH, " finish ");
}

void Ciphertext::square(const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    Out(KEYSWITCH, " start ");

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);
    op_count[OPS::SQUARE]++;

    if (cc.GPUid.size() == 1) {
        if constexpr (0) {
            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cc.getKeySwitchAux().squareElement(c1);
            cc.getKeySwitchAux().modup();
            auto& aux0 = cc.getKeySwitchAux().dotKSKInPlace(kskEval, nullptr);
            cc.getKeySwitchAux().moddown(true, false);
            aux0.moddown(true, false);
            //c1.mult1AddMult23Add4(c0, c0, c1, cc.getKeySwitchAux());
            c1.binomialSquareFold(c0, aux0, cc.getKeySwitchAux());
            if (rescale) {
                this->rescale();
            }
        } else if constexpr (1) {
            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cc.getKeySwitchAux().squareModupDotKSK(c0, c1, kskEval);

            c1.moddown(true, false);

            c0.moddown(true, false);
            if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL)
                this->rescale();
        } else {
            this->mult(*this, kskEval, rescale);
        }
    } else {
        RNSPoly& in = cc.getKeySwitchAux();
        in.setLevel(c1.getLevel());
        in.squareElement(c1);

        RNSPoly& aux = MGPUkeySwitchCore(in, kskEval, true);

        c1.binomialSquareFold(c0, aux, in);
        if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
            this->rescale();
        }
    }
    // Manage metadata
    NoiseLevel += NoiseLevel;
    NoiseFactor *= NoiseFactor;
    Out(KEYSWITCH, " finish ");
}

void Ciphertext::multScalarNoPrecheck(const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::MULTSCALAR]++;

    auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
    c0.multScalar(elem);
    c1.multScalar(elem);

    // Manage metadata
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());

    if (rescale && cc.rescaleTechnique == Context::FIXEDAUTO) {
        this->rescale();
    }
}

void Ciphertext::multScalar(const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);
    multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == Context::FIXEDMANUAL);
}

void Ciphertext::addScalar(const double c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::ADDSCALAR]++;

    auto elem = cc.ElemForEvalAddOrSub(c0.getLevel(), std::abs(c), this->NoiseLevel);

    if (c >= 0.0) {
        c0.addScalar(elem);
    } else {
        c0.subScalar(elem);
    }
}

void Ciphertext::automorph(const int index, const int br) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.automorph(index, br, nullptr);
    c1.automorph(index, br, nullptr);
}

void Ciphertext::automorph_multi(const int index, const int br) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.automorph_multi(index, br);
    c1.automorph_multi(index, br);
}

void Ciphertext::extend() {
    c0.generateSpecialLimbs(true);
    c1.generateSpecialLimbs(true);

    c0.scaleByP();
    c1.scaleByP();
}

void Ciphertext::rotate(const int index, const KeySwitchingKey& kskRot, const bool moddown) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::ROTATE]++;
    assert(index != 0);
    constexpr bool PRINT = false;

    if (cc.GPUid.size() == 1) {
        if constexpr (0) {
            if constexpr (PRINT) {
                std::cout << "Output Automorph 1.";
                for (auto& j : c1.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }
            c1.modupInto(cc.getKeySwitchAux());
            RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot);
            c1.moddown();
            if constexpr (PRINT) {
                std::cout << "Output Automorph 1.";
                for (auto& j : c1.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }
            c1.automorph(index, 1, nullptr);

            aux0.moddown(true, false);
            if constexpr (PRINT) {
                std::cout << "c0\n";
                for (auto& j : c0.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }
            c0.add(aux0);
            if constexpr (PRINT) {
                std::cout << "Output KeySwitch 0.";
                for (auto& j : aux0.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }

            if constexpr (PRINT) {
                std::cout << "Output Add 0.";
                for (auto& j : c0.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }
            c0.automorph(index, 1, nullptr);
            if constexpr (PRINT) {
                std::cout << "Output Rot 0.";
                for (auto& j : c0.GPU)
                    for (auto& i : j.limb) {
                        SWITCH(i, printThisLimb(2));
                    }
            }
        } else if constexpr (1) {
            cc.getKeySwitchAux().setLevel(c1.getLevel());
            cc.getKeySwitchAux().rotateModupDotKSK(c0, c1, kskRot);

            if (moddown)
                c1.moddown(true, false);
            c1.automorph(index, 1, nullptr);
            if (moddown)
                c0.moddown(true, false);
            c0.automorph(index, 1, nullptr);
        } else {

            c1.modupInto(cc.getKeySwitchAux());
            RNSPoly& aux0 = c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), kskRot);
            c1.moddown();
            c1.automorph(index, 1, nullptr);

            aux0.moddown(true, false);
            c0.add(aux0);
            c0.automorph(index, 1, nullptr);
        }
    } else {
        RNSPoly& in = cc.getKeySwitchAux();
        in.setLevel(c1.getLevel());
        in.copy(c1);

        RNSPoly& aux = MGPUkeySwitchCore(in, kskRot, moddown);
        if (!moddown) {
            //in.moddown(true, false);
            c1.SetModUp(false);
            c1.generateSpecialLimbs(false);
        }
        c1.automorph(index, 1, &in);
        //c1.moddown(true, false);

        if (!moddown) {
            //aux.moddown(true, false);
            c0.SetModUp(false);
            c0.generateSpecialLimbs(true);
        }
        aux.add(aux, c0);
        //c0.add(aux);
        c0.automorph(index, 1, &aux);
        //c0.moddown();
    }
}

void Ciphertext::rotate(const Ciphertext& c, const int index, const KeySwitchingKey& kskRot) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(c);
    this->rotate(index, kskRot, true);
}

void Ciphertext::conjugate(const Ciphertext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::CONJUGATE]++;

    if (cc.GPUid.size() == 1) {
        this->copy(c);
        //this->rotate(2 * cc.N - 1, cc.GetRotationKey(2 * cc.N - 1));

        int index = 2 * cc.N - 1;
        cc.getKeySwitchAux().setLevel(c1.getLevel());
        cc.getKeySwitchAux().rotateModupDotKSK(c0, c1, cc.GetRotationKey(index));

        c1.moddown(true, false);
        for (int i = 0; i < (int)c1.GPU.size(); ++i) {
            c1.GPU.at(i).automorph(index, 1, nullptr, c1.isModUp());
        }
        c0.moddown(true, false);
        for (int i = 0; i < (int)c0.GPU.size(); ++i) {
            c0.GPU.at(i).automorph(index, 1, nullptr, c0.isModUp());
        }
    } else {
        this->copy(c);

        RNSPoly& in = cc.getKeySwitchAux();
        in.setLevel(c1.getLevel());
        in.copyShallow(c1);

        int index = 2 * cc.N - 1;

        RNSPoly& aux = MGPUkeySwitchCore(in, cc.GetRotationKey(index), true);
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (int i = 0; i < (int)cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            c1.GPU.at(i).automorph(index, 1, &in.GPU.at(i), c1.isModUp());
        }
        c0.add(aux);
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (int i = 0; i < (int)cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            c0.GPU.at(i).automorph(index, 1, nullptr, c0.isModUp());
        }
    }
}

void Ciphertext::rotate_hoisted(const std::vector<KeySwitchingKey*>& ksk, const std::vector<int>& indexes,
                                std::vector<Ciphertext*> results, const bool ext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::HOISTEDROTATE]++;
    op_count[OPS::HOISTEDROTATEOUTS] += ksk.size();

    assert(ksk.size() == results.size());

    bool grow_full = true;
    for (auto& i : results) {
        if (i->c0.getLevel() == -1) {
            i->c0.grow(grow_full ? cc.L : this->c0.getLevel(), true);
        }
        if (i->c1.getLevel() == -1) {
            i->c1.grow(grow_full ? cc.L : this->c1.getLevel(), true);
        }
        if (ext)
            i->c0.generateSpecialLimbs(false);
        if (ext)
            i->c1.generateSpecialLimbs(false);

        i->c0.setLevel(c0.getLevel());
        i->c1.setLevel(c1.getLevel());
    }

    if (cc.GPUid.size() == 1) {
        cc.getKeySwitchAux().setLevel(c1.getLevel());
        c1.modupInto(cc.getKeySwitchAux());

        for (size_t i = 0; i < ksk.size(); ++i) {
            if (indexes[i] == 0) {
                results[i]->copy(*this);
                if (ext) {
                    results[i]->extend();
                }
            } else {
                RNSPoly& aux0 = results[i]->c1.dotKSKInPlaceFrom(cc.getKeySwitchAux(), *ksk[i], &c1);
                //results[i]->c0.dropToLevel(getLevel());
                //results[i]->c1.dropToLevel(getLevel());
                if (!ext)
                    results[i]->c1.moddown(true, false);
                results[i]->c1.automorph(indexes[i], 1);
                if (!ext)
                    aux0.moddown(true, false);

                results[i]->c0.add(c0, aux0);
                results[i]->c0.automorph(indexes[i], 1);

                results[i]->NoiseLevel = NoiseLevel;
                results[i]->NoiseFactor = NoiseFactor;
            }
        }
    } else {
        if constexpr (0) {
            for (size_t i = 0; i < ksk.size(); ++i) {
                if (indexes[i] != 0) {
                    results[i]->rotate(*this, indexes[i], *ksk[i]);
                } else {
                    results[i]->copy(*this);
                }
            }
        } else {
            RNSPoly& in = cc.getKeySwitchAux();
            in.setLevel(c1.getLevel());
            in.copy(c1);
            in.modup();

            for (size_t i = 0; i < ksk.size(); ++i) {
                if (indexes[i] == 0) {
                    results[i]->copy(*this);
                    if (ext) {
                        results[i]->extend();
                    }
                } else {
                    RNSPoly& aux0 = in.dotKSKInPlace(*ksk[i], &c1);
                    //results[i]->c0.dropToLevel(getLevel());
                    //results[i]->c1.dropToLevel(getLevel());
                    if (!ext) {
                        in.moddown(true, false);
                    }
                    //std::cout << "in ismodup: " << in.isModUp() << std::endl;
                    results[i]->c1.automorph(indexes[i], 1, &in);
                    //std::cout << "results[i] c1 ismodup: " << results[i]->c1.isModUp() << std::endl;
                    if (!ext) {
                        aux0.moddown(true, false);
                    }
                    //std::cout << "aux0 ismodup: " << aux0.isModUp() << std::endl;
                    //std::cout << "c0 ismodup: " << c0.isModUp() << std::endl;
                    results[i]->c0.add(c0, aux0);
                    //std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;
                    results[i]->c0.automorph(indexes[i], 1, nullptr);
                    //std::cout << "results[i] c0 ismodup: " << results[i]->c0.isModUp() << std::endl;
                    results[i]->NoiseLevel = NoiseLevel;
                    results[i]->NoiseFactor = NoiseFactor;
                }
            }
        }
    }
}
void Ciphertext::mult(const Ciphertext& b, const Ciphertext& c, const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (this == &b && this == &c) {
        this->square(kskEval, rescale);
    } else if (this == &b) {
        this->mult(c, kskEval, rescale);
    } else if (this == &c) {
        this->mult(b, kskEval, rescale);
    } else {
        if (b.getLevel() <= c.getLevel()) {
            this->copy(b);
            this->mult(c, kskEval, rescale);
        } else {
            this->copy(c);
            this->mult(b, kskEval, rescale);
        }
    }
}

void Ciphertext::square(const Ciphertext& src, const KeySwitchingKey& kskEval, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    if (this == &src) {
        this->square(kskEval, rescale);
    } else {
        this->copy(src);
        this->square(kskEval, rescale);
    }
}
void Ciphertext::dropToLevel(int level) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    c0.dropToLevel(level);
    c1.dropToLevel(level);
}
int Ciphertext::getLevel() const {
    assert(c0.getLevel() == c1.getLevel());
    return c0.getLevel();
}
void Ciphertext::multScalar(const Ciphertext& b, const double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(b);
    this->multScalar(c, rescale);
}
void Ciphertext::evalLinearWSumMutable(uint32_t n, const std::vector<Ciphertext*>& ctxs, std::vector<double> weights) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::WSUM]++;
    op_count[OPS::WSUMINPUTS] += n;

    // TODO adjujst level of inputs for correct precission

    if constexpr (1) {
        this->c0.grow(ctxs[0]->getLevel(), true);
        this->c1.grow(ctxs[0]->getLevel(), true);
        this->NoiseLevel = 1;

        for (size_t i = 0; i < n; ++i) {
            if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
                assert(ctxs[i]->NoiseLevel == 1);
                assert(getLevel() <= ctxs[i]->getLevel());
            } else {
                assert(ctxs[i]->NoiseLevel == 1);

                // assert(getLevel() == ctxs[i]->getLevel()); TODO uncomment and fix FIXEDAUTO bug
            }
        }

        std::vector<uint64_t> elem(MAXP * n);

        //#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            auto aux = cc.ElemForEvalMult(c0.getLevel(), weights[i]);
            for (size_t j = 0; j < aux.size(); ++j)
                elem[i * MAXP + j] = aux[j];
        }

        std::vector<const RNSPoly*> c0s(n), c1s(n);

        for (size_t i = 0; i < n; ++i) {
            c0s[i] = &ctxs[i]->c0;
            c1s[i] = &ctxs[i]->c1;
        }
        c0.evalLinearWSum(n, c0s, elem);
        c1.evalLinearWSum(n, c1s, elem);

        this->NoiseLevel = 2;
        this->NoiseFactor = ctxs[0]->NoiseFactor;
        NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel());
    } else {
        this->multScalar(*ctxs[0], weights[0], false);
        for (int i = 1; i < n; ++i) {
            assert(getLevel() <= ctxs[i]->getLevel());
        }
        for (int i = 1; i < n; ++i) {
            this->addMultScalar(*ctxs[i], weights[i]);
        }
    }
}
void Ciphertext::addMultScalar(const Ciphertext& b, double d) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::MULTSCALAR]++;
    op_count[OPS::COPY]++;
    op_count[OPS::ADD]++;

    assert(NoiseLevel == 2);
    assert(b.NoiseLevel == 1);
    assert(b.getLevel() >= getLevel());
    auto elem = cc.ElemForEvalMult(c0.getLevel(), d);

    RNSPoly aux0(cc);
    RNSPoly aux1(cc);
    aux0.copy(b.c0);
    aux0.multScalar(elem);
    c0.add(aux0);
    aux1.copy(b.c1);
    aux1.multScalar(elem);
    c1.add(aux1);
}

void Ciphertext::addScalar(const Ciphertext& b, double c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(b);
    this->addScalar(c);
}
void Ciphertext::add(const Ciphertext& b, const Ciphertext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(NoiseLevel <= 2);
    if (this == &b && this == &c) {
        this->add(c);
    } else if (this == &b) {
        this->add(c);
    } else if (this == &c) {
        this->add(b);
    } else {
        if (b.getLevel() <= c.getLevel()) {
            this->copy(b);
            this->add(c);
        } else {
            this->copy(c);
            this->add(b);
        }
    }
}
void Ciphertext::copy(const Ciphertext& ciphertext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::COPY]++;
    CudaCheckErrorModNoSync;
    c0.copy(ciphertext.c0);
    c1.copy(ciphertext.c1);
    //cudaDeviceSynchronize();
    this->NoiseLevel = ciphertext.NoiseLevel;
    this->NoiseFactor = ciphertext.NoiseFactor;
}
void Ciphertext::multPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(c);
    multPt(b, rescale);
}
void Ciphertext::addMultPt(const Ciphertext& c, const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    op_count[OPS::ADDMULTPT]++;

    assert(NoiseLevel == 2);
    assert(c.NoiseLevel == 1);
    assert(b.NoiseLevel == 1);

    c0.addMult(c.c0, b.c0);
    c1.addMult(c.c1, b.c0);

    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        this->rescale();
    }
}
void Ciphertext::addPt(const Ciphertext& ciphertext, const Plaintext& plaintext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    this->copy(ciphertext);
    this->addPt(plaintext);
}

void Ciphertext::sub(const Ciphertext& ciphertext, const Ciphertext& ciphertext1) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    assert(ciphertext.getLevel() <= ciphertext1.getLevel());
    this->copy(ciphertext);
    this->sub(ciphertext1);
}
bool Ciphertext::adjustForAddOrSub(const Ciphertext& b) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDMANUAL) {
        if (b.NoiseLevel > NoiseLevel || (b.getLevel() < getLevel()))
            return false;
        else
            return true;
    } else if (cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (getLevel() - NoiseLevel > b.getLevel() - b.NoiseLevel) {
            if (b.NoiseLevel == 1 && NoiseLevel == 2) {
                this->dropToLevel(b.getLevel() + 1);
                rescale();
            } else {
                this->dropToLevel(b.getLevel());
            }
            return true;
        } else if (b.NoiseLevel == 1 && NoiseLevel == 2) {
            rescale();
            return true;
        } else if (NoiseLevel == 1 && b.NoiseLevel == 2) {
            return false;
        } else {
            return true;
        }
    } else if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        usint c1lvl = getLevel();
        usint c2lvl = b.getLevel();
        usint c1depth = this->NoiseLevel;
        usint c2depth = b.NoiseLevel;
        auto sizeQl1 = c1lvl + 1;
        //auto sizeQl2 = c2lvl + 1;

        if (c1lvl > c2lvl) {
            if (c1depth == 2) {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = b.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    double q1 =
                        cc.param.ModReduceFactor[sizeQl1 - 1];  // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                    multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
                    rescale();
                    if (getLevel() > b.getLevel()) {
                        this->dropToLevel(b.getLevel());
                    }

                    assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
                    NoiseFactor = b.NoiseFactor;
                    /*
                    rescale();
                    double scf1 = NoiseFactor;
                    double scf2 = b.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  // cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalarNoPrecheck(scf2 / scf1 / scf);
                    this->dropToLevel(c2lvl);
                    //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
                    NoiseFactor = scf2;
*/
                } else {
                    if (c1lvl - 1 == c2lvl) {
                        rescale();
                    } else {
                        double scf1 = NoiseFactor;
                        double scf2 =
                            cc.param
                                .ScalingFactorRealBig[c2lvl + 1];  //cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                        double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                        double q1 =
                            cc.param.ModReduceFactor[sizeQl1 - 1];  //cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                        multScalarNoPrecheck(scf2 / scf1 * q1 / scf);
                        rescale();
                        if (getLevel() - 1 > b.getLevel()) {
                            this->dropToLevel(b.getLevel() + 1);
                            //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
                        }
                        rescale();
                        assert(std::abs((NoiseFactor * scf2 / scf1 * q1 / scf - b.NoiseFactor) / b.NoiseFactor) <
                               0.001);

                        NoiseFactor = b.NoiseFactor;
                    }
                }
            } else {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = b.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  // cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalarNoPrecheck(scf2 / scf1 / scf);
                    this->dropToLevel(c2lvl);
                    //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
                    assert(std::abs((NoiseFactor * scf2 / scf1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
                    NoiseFactor = scf2;
                } else {
                    double scf1 = NoiseFactor;
                    double scf2 =
                        cc.param.ScalingFactorRealBig[c2lvl + 1];    //cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalarNoPrecheck(scf2 / scf1 / scf);
                    if (c1lvl - 1 > c2lvl) {
                        this->dropToLevel(c2lvl + 1);
                        //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                    }
                    rescale();
                    assert(std::abs((NoiseFactor * scf2 / scf1 / scf - b.NoiseFactor) / b.NoiseFactor) < 0.001);
                    NoiseFactor = b.NoiseFactor;
                }
            }
            return true;
        } else if (c1lvl < c2lvl) {
            return false;
        } else {
            if (c1depth < c2depth) {
                multScalar(1.0, false);
            } else if (c2depth < c1depth) {
                return false;
            }
            return true;
        }
    }
    assert("This never happens" == nullptr);
    return false;
}

bool Ciphertext::adjustForMult(const Ciphertext& ciphertext) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (adjustForAddOrSub(ciphertext)) {
        if (NoiseLevel == 2)
            rescale();
        if (ciphertext.NoiseLevel == 2)
            return false;
        else
            return true;
    } else {
        if (NoiseLevel == 2)
            rescale();
        return false;
    }
}
bool Ciphertext::hasSameScalingFactor(const Plaintext& b) const {
    return NoiseFactor > b.NoiseFactor * (1 - 1e-9) && NoiseFactor < b.NoiseFactor * (1 + 1e-9);
}
void Ciphertext::clearOpRecord() {
    op_count.clear();
}

void Ciphertext::dotProductPt(const Ciphertext* ciphertexts, const Plaintext* plaintexts, const int n, const bool ext) {

    std::vector<const RNSPoly*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);

    for (int i = 0; i < n; ++i) {
        c0s[i] = &(ciphertexts[i].c0);
        c1s[i] = &(ciphertexts[i].c1);
        pts[i] = &(plaintexts[i].c0);
        assert(getLevel() <= ciphertexts[i].getLevel());
        assert(getLevel() <= plaintexts[i].c0.getLevel());
        if (ext) {
            assert(ciphertexts[i].c0.isModUp());
            assert(plaintexts[i].c0.isModUp());
        }
    }

    c0.dotProductPt(c1, c0s, c1s, pts, ext);

    // Manage metadata
    NoiseLevel = ciphertexts[0].NoiseLevel + plaintexts[0].NoiseLevel;
    NoiseFactor = ciphertexts[0].NoiseFactor * plaintexts[0].NoiseFactor;
}

void Ciphertext::dotProductPt(const Ciphertext* ciphertexts, const Plaintext** plaintexts, const int n,
                              const bool ext) {

    std::vector<const RNSPoly*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);

    for (int i = 0; i < n; ++i) {
        c0s[i] = &(ciphertexts[i].c0);
        c1s[i] = &(ciphertexts[i].c1);
        pts[i] = &(plaintexts[i]->c0);
        assert(getLevel() <= ciphertexts[i].getLevel());
        assert(getLevel() <= plaintexts[i]->c0.getLevel());
        if (ext) {
            assert(ciphertexts[i].c0.isModUp());
            assert(plaintexts[i]->c0.isModUp());
        }
    }
    c0.dotProductPt(c1, c0s, c1s, pts, ext);

    // Manage metadata
    NoiseLevel = ciphertexts[0].NoiseLevel + plaintexts[0]->NoiseLevel;
    NoiseFactor = ciphertexts[0].NoiseFactor * plaintexts[0]->NoiseFactor;
}

void Ciphertext::printOpRecord() {
    std::cout << "|-------------- OP COUNT --------------|\n";
    for (const auto& [op, c] : op_count) {
        std::cout << opstr[op] << c << "\n";
    }
    std::cout << "|--------------------------------------|" << std::endl;
}

}  // namespace FIDESlib::CKKS
