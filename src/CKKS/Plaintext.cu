//
// Created by carlosad on 25/04/24.
//

#include <cereal/external/rapidjson/internal/itoa.h>
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/Plaintext.cuh"

namespace FIDESlib::CKKS {

Plaintext::Plaintext(Context& cc)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc) {
    CudaNvtxStop();
}

Plaintext::Plaintext(Context& cc, const RawPlainText& raw)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      c0(cc) {
    load(raw);
    CudaNvtxStop();
}

void Plaintext::load(const RawPlainText& raw) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    c0.loadConstant(raw.sub_0, raw.moduli);

    NoiseFactor = raw.Noise;
    NoiseLevel = raw.NoiseLevel;
    /*
    cudaDeviceSynchronize();
    std::cout << "plaintext ";
    for (auto& j : c0.GPU) {
        cudaSetDevice(j.device);

        for (auto& k : j.limb) {
            SWITCH(k, printThisLimb(1));
        }
    }
    std::cout << std::endl;
    for (auto& j : c0.GPU) {
        cudaSetDevice(j.device);
        for (auto& k : j.SPECIALlimb) {
            SWITCH(k, printThisLimb(1));
        }
    }
    std::cout << std::endl;
    cudaDeviceSynchronize();
    */
}

void Plaintext::store(RawPlainText& raw) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    cudaDeviceSynchronize();

    raw.numRes = c0.getLevel() + 1;
    raw.sub_0.resize(raw.numRes);
    c0.store(raw.sub_0);
    raw.N = cc.N;
    c0.sync();

    raw.Noise = NoiseFactor;
    raw.NoiseLevel = NoiseLevel;
    cudaDeviceSynchronize();
}

void Plaintext::moddown() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    c0.moddown(true, true);
}

bool Plaintext::adjustPlaintextToCiphertext(const Plaintext& p, const Ciphertext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    constexpr bool PRINT = false;

    if (cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (p.c0.getLevel() - p.NoiseLevel > c.getLevel() - c.NoiseLevel) {
            this->copy(p);
            if (c.NoiseLevel == 1 && NoiseLevel == 2) {
                this->c0.dropToLevel(c.getLevel() + 1);
                rescale();
            } else {
                this->c0.dropToLevel(c.getLevel());
            }
            return true;
        } else if (c.NoiseLevel == 1 && p.NoiseLevel == 2) {
            this->copy(p);
            rescale();
            return true;
        } else if (p.NoiseLevel == 1 && c.NoiseLevel == 2) {
            return false;
        } else {
            this->copy(p);
            return true;
        }
    }
    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        usint c1lvl = p.c0.getLevel();
        usint c2lvl = c.getLevel();
        usint c1depth = p.NoiseLevel;
        usint c2depth = c.NoiseLevel;
        auto sizeQl1 = c1lvl + 1;
        auto sizeQl2 = c2lvl + 1;

        if (c1lvl > c2lvl) {
            this->copy(p);
            if (c1depth == 2) {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = c.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    double q1 =
                        cc.param.ModReduceFactor[sizeQl1 - 1];  // cryptoParams->GetModReduceFactor(sizeQl1 - 1);
                    multScalar(scf2 / scf1 * q1 / scf, false);
                    rescale();
                    if (c1lvl > c2lvl) {
                        this->c0.dropToLevel(c2lvl);
                        //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                    }
                    NoiseFactor *= scf2 / scf1 * q1 / scf;
                    assert(NoiseFactor == c.NoiseFactor);
                    NoiseFactor = c.NoiseFactor;
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
                        multScalar(scf2 / scf1 * q1 / scf, false);
                        rescale();
                        if (c1lvl - 2 > c2lvl) {
                            this->c0.dropToLevel(c2lvl + 1);
                            //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 2);
                        }
                        rescale();

                        NoiseFactor = c.NoiseFactor;
                    }
                }
            } else {
                if (c2depth == 2) {
                    double scf1 = NoiseFactor;
                    double scf2 = c.NoiseFactor;
                    double scf = cc.param.ScalingFactorReal[c1lvl];  // cryptoParams->GetScalingFactorReal(c1lvl);
                    multScalar(scf2 / scf1 / scf, false);
                    this->c0.dropToLevel(c2lvl);
                    //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl);
                    NoiseFactor = scf2;
                } else {
                    if constexpr (PRINT)
                        std::cout << "Adjusting plaintext with noiseDegree 1" << std::endl;
                    double scf1 = NoiseFactor;
                    double scf2 =
                        cc.param.ScalingFactorRealBig[c2lvl + 1];    //cryptoParams->GetScalingFactorRealBig(c2lvl - 1);
                    double scf = cc.param.ScalingFactorReal[c1lvl];  //cryptoParams->GetScalingFactorReal(c1lvl);
                    if constexpr (PRINT)
                        std::cout << "Scale adjustment: " << scf << std::endl;

                    multScalar(scf2 / scf1 / scf, false);
                    if (c1lvl - 1 > c2lvl) {
                        if constexpr (PRINT)
                            std::cout << "Dropping levels: " << c1lvl - c2lvl - 1 << std::endl;
                        this->c0.dropToLevel(c2lvl + 1);
                        //LevelReduceInternalInPlace(ciphertext1, c2lvl - c1lvl - 1);
                    }
                    rescale();
                    NoiseFactor = c.NoiseFactor;
                }
            }
            return true;
        } else if (c1lvl < c2lvl) {
            return false;
        } else {
            this->copy(p);
            if (c1depth < c2depth) {
                multScalar(1.0, false);
            } else if (c2depth < c1depth) {
                rescale();
            }
            return true;
        }
    }
    return false;
}
void Plaintext::copy(const Plaintext& p) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    this->c0.copy(p.c0);
    this->NoiseFactor = p.NoiseFactor;
    this->NoiseLevel = p.NoiseLevel;
}
void Plaintext::multScalar(double c, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    /*
    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT ||
        cc.rescaleTechnique == Context::FIXEDAUTO) {
        if (NoiseLevel == 2)
            this->rescale();
    }
    assert(this->NoiseLevel == 1);
    */
    auto elem = cc.ElemForEvalMult(c0.getLevel(), c);
    /*
    for (int i = 0; i < elem.size(); i++) {
        std::cout << elem[i] << " ";
    }
    std::cout << std::endl;
*/
    c0.multScalar(elem);

    if (rescale) {
        c0.rescale();
    }
    // Manage metadata
    NoiseLevel += 1;
    NoiseFactor *= cc.param.ScalingFactorReal.at(c0.getLevel() + rescale);
    if (rescale) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + rescale);
        NoiseLevel -= 1;
    }
}

void Plaintext::rotate_hoisted(const std::vector<int>& indexes, std::vector<Plaintext*>& results) {
    assert(indexes.size() == results.size() && "rotate_hoisted: mismatched indexes and results sizes");

    for (size_t i = 0; i < indexes.size(); ++i) {
        int index = indexes[i];
        if (index == 0) {
            results[i]->copy(*this);
        } else {

            // Copy and rotate
            results[i]->copy(*this);
            results[i]->automorph(index);
        }
    }
}

#if false
void Plaintext::multPt(const Plaintext& b, bool rescale) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        if (NoiseLevel == 2)
            this->rescale();
    }

    if (cc.rescaleTechnique == Context::FIXEDAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTO ||
        cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        // if (b.c0.getLevel() != this.getLevel() || b.NoiseLevel == 2 /*!hasSameScalingFactor(b)*/) {
        if (!hasSameScalingFactor(b)) {
            Plaintext b_(cc);
            if (NoiseLevel == 2)
                this->rescale();
            if (b_.NoiseLevel == 2)
                b_.rescale();
            multPt(b_, rescale);
            return;
        }
    }

    assert(NoiseLevel < 2);
    assert(b.NoiseLevel < 2);
    c0.multPt(b.c0, rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL);

    // Manage metadata
    NoiseLevel += b.NoiseLevel;
    NoiseFactor *= b.NoiseFactor;
    if (rescale && cc.rescaleTechnique == CKKS::Context::FIXEDMANUAL) {
        NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
        NoiseLevel -= 1;
    }
}

void Plaintext::addPt(const Plaintext& c) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    // assert(NoiseLevel == b.NoiseLevel);
    c0.add(c.c0);
}

#endif

void Plaintext::rescale() {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));
    assert(this->NoiseLevel >= 2);
    /*
    std::cout << "Rescale plaintext, level" << c0.getLevel() << std::endl;
    for (auto& i : c0.GPU) {
        std::cout << i.limb.size() << " ";
    }
    std::cout << std::endl;
*/
    c0.rescale();

    // Manage metadata
    NoiseFactor /= cc.param.ModReduceFactor.at(c0.getLevel() + 1);
    NoiseLevel -= 1;
}

void Plaintext::automorph(const int index) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    if (c0.isModUp()) {
        std::cout << "isModup plaintext automorph not implemented" << std::endl;
    }
    auto& aux = cc.getModdownAux(0);
    aux.setLevel(c0.getLevel());
    aux.automorph(index, 1, &c0);
    c0.copy(aux);
}

}  // namespace FIDESlib::CKKS