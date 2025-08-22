//
// Created by carlosad on 25/04/24.
//
#include <errno.h>

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

#include <omp.h>
#include <stdexcept>

namespace FIDESlib::CKKS {
void RNSPoly::grow(int new_level, bool single_malloc, bool constant) {
    if (level > new_level)
        return;
    level = new_level;
    //if (level == -1) {
    //std::cout << "from 0" << std::endl;
    if ((!single_malloc || (GPU.at(0).limb.size() > 0)) && GPU.at(0).bufferLIMB == nullptr) {
        // TODO fix bug (check that limb size matches level)
        int init = 0;
        for (auto& g : GPU)
            init += g.limb.size();
        for (int i = init; i <= new_level; ++i) {
            GPU.at(cc.limbGPUid.at(i).x).generateLimb();
        }
    } else {
#pragma omp parallel for num_threads(GPU.size())
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)GPU.size());
            if (!constant) {
                GPU.at(i).generateLimbSingleMalloc();
            } else {
                GPU.at(i).generateLimbConstant();
            }
        }
    }
}

RNSPoly::RNSPoly(Context& context, int level, bool single_malloc) : cc(context), level(-1) {
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        cudaSetDevice(cc.GPUid.at(i));
        GPU.emplace_back(cc, &this->level, i);
    }
    assert(level >= -1 && level <= cc.L);
    grow(level, single_malloc);
}

RNSPoly::RNSPoly(Context& context, const std::vector<std::vector<uint64_t>>& data) : RNSPoly(context, data.size() - 1) {

    assert(data.size() <= cc.prime.size());
    std::vector<uint64_t> moduli(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        moduli[i] = cc.prime[i].p;
    load(data, moduli);
}

RNSPoly::RNSPoly(RNSPoly&& src) noexcept : cc(src.cc), level(src.level), modUp(src.modUp), GPU(std::move(src.GPU)) {
    for (auto& g : GPU) {
        g.level = &(this->level);
    }
}

int RNSPoly::getLevel() const {
    return level;
}

void RNSPoly::store(std::vector<std::vector<uint64_t>>& data) {
    data.resize(level + 1);
    for (size_t i = 0; i < data.size(); ++i) {
        //auto& rec = cc.meta[cc.limbGPUid[i].x][cc.limbGPUid[i].y];
        cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], store_convert(data[i]));
    }
}
bool RNSPoly::isModUp() const {
    return modUp;
}
void RNSPoly::SetModUp(bool newValue) {
    modUp = newValue;
}
void RNSPoly::scaleByP() {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).scaleByP();
    }
    this->SetModUp(true);
}
void RNSPoly::multNoModdownEnd(RNSPoly& c0, const RNSPoly& bc0, const RNSPoly& bc1, const RNSPoly& in,
                               const RNSPoly& aux) {
    assert(in.isModUp());
    assert(aux.isModUp());
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).multNoModdownEnd(c0.GPU.at(i), bc0.GPU.at(i), bc1.GPU.at(i), in.GPU.at(i), aux.GPU.at(i));
    }
    this->SetModUp(true);
    c0.SetModUp(true);
}

void RNSPoly::add(const RNSPoly& p) {

    if (p.isModUp() && !this->isModUp()) {
        //std::cout << "Adapt non modup destination add" << std::endl;
        generateSpecialLimbs(true);
        scaleByP();
    }
    assert(p.isModUp() == this->isModUp());
    assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).add(p.GPU.at(i), this->isModUp() || p.isModUp());
    }
    this->SetModUp(this->isModUp() || p.isModUp());
}

void RNSPoly::sub(const RNSPoly& p) {
    assert(level <= p.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).sub(p.GPU.at(i));
    }
}

void RNSPoly::modup() {
    //  assert(GPU.size() == 1 || 0 == "ModUp Multi-GPU not implemented.");
    RNSPoly& aux = cc.getKeySwitchAux2();
    if (cc.GPUid.size() == 1) {
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            GPU.at(i).modup(aux.GPU.at(i));
        }
    } else {
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            GPU.at(i).modupMGPU(aux.GPU.at(i));
        }
    }
    this->SetModUp(true);
    aux.SetModUp(true);
}

void RNSPoly::sync() {
    for (auto& i : GPU) {
        for (auto& j : i.limb) {
            assert(STREAM(j).ptr != nullptr);
            cudaStreamSynchronize(STREAM(j).ptr);
        }
        cudaStreamSynchronize(i.s.ptr);
    }
}

void RNSPoly::rescale() {
    //    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
    if (GPU.size() == 1) {
        for (auto& i : GPU) {
            i.rescale();
        }
        level -= 1;
    } else {
        int more_than_0 = 0;
        for (size_t i = 0; i < GPU.size(); ++i)
            if (GPU[i].getLimbSize(level) != 0)
                more_than_0++;

        if (more_than_0 == 1) {
            for (auto& i : GPU) {
                if (i.getLimbSize(level) > 0)
                    i.rescale();
            }
        } else {
#pragma omp parallel for num_threads(GPU.size())
            for (size_t i = 0; i < GPU.size(); ++i) {
                if (omp_get_num_threads() != (int)GPU.size())
                    throw std::invalid_argument("OMP didn't create enough threads");
                assert(omp_get_num_threads() == (int)GPU.size());
                GPU[i].rescaleMGPU();
            }
        }
        level -= 1;
    }
}

void RNSPoly::rescaleDouble(RNSPoly& poly) {
    //    assert(GPU.size() == 1 && "Rescale Multi-GPU not implemented.");
    if (GPU.size() == 1) {
        for (auto& i : GPU) {
            i.rescale();
        }
        level -= 1;
        for (auto& i : poly.GPU) {
            i.rescale();
        }
        poly.level -= 1;
    } else {

        int more_than_0 = 0;
        for (size_t i = 0; i < GPU.size(); ++i)
            if (GPU[i].getLimbSize(level) > 0)
                more_than_0++;

        if (more_than_0 == 1) {
            for (size_t i = 0; i < GPU.size(); ++i) {
                if (GPU[i].getLimbSize(level) > 0) {
                    GPU[i].rescale();
                    poly.GPU[i].rescale();
                }
            }
        } else {
#pragma omp parallel for num_threads(GPU.size())
            for (size_t i = 0; i < GPU.size(); ++i) {
                if (omp_get_num_threads() != (int)GPU.size())
                    throw std::invalid_argument("OMP didn't create enough threads");
                assert(omp_get_num_threads() == (int)GPU.size());
                //GPU[i].rescaleMGPU();
                GPU[i].doubleRescaleMGPU(poly.GPU[i]);
            }
        }
        level -= 1;
        poly.level -= 1;
    }
}

void RNSPoly::multPt(const RNSPoly& p, bool rescale) {
    if (rescale) {
        if (GPU.size() == 1) {
            for (size_t i = 0; i < GPU.size(); ++i) {
                GPU.at(i).multPt(p.GPU.at(i));
            }
            --level;
        } else {
#pragma omp parallel for num_threads(GPU.size())
            for (size_t i = 0; i < GPU.size(); ++i) {
                assert(omp_get_num_threads() == (int)GPU.size());
                GPU.at(i).multElement(p.GPU.at(i));
                GPU.at(i).rescaleMGPU();
            }
            --level;
        }
    } else {
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            GPU.at(i).multElement(p.GPU.at(i));
        }
    }
}

void RNSPoly::freeSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).freeSpecialLimbs();
    }
    this->SetModUp(false);
}

template <ALGO algo>
void RNSPoly::NTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).NTT<algo>(batch, sync);
    }
}

#define YY(algo) template void RNSPoly::NTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

template <ALGO algo>
void RNSPoly::INTT(int batch, bool sync) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).INTT<algo>(batch, sync);
    }
}

#define YY(algo) template void RNSPoly::INTT<algo>(int batch, bool sync);

#include "ntt_types.inc"

#undef YY

/*
std::array<RNSPoly, 2> RNSPoly::dotKSK(const KeySwitchingKey& ksk) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    std::array<RNSPoly, 2> result{RNSPoly(cc, level, true), RNSPoly(cc, level, true)};
    result[0].generateSpecialLimbs(false);
    result[1].generateSpecialLimbs(false);

    if constexpr (PRINT)
        for (auto& i : ksk.b.GPU) {
            for (auto& j : i.DECOMPlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }

            for (auto& j : i.DIGITlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
        }
    for (size_t i = 0; i < GPU.size(); ++i) {
        dotKSKinto(result[0], ksk.b, level);
        dotKSKinto(result[1], ksk.a, level);
    }

    Out(KEYSWITCH, "dotKSK out");
    return result;
}
*/

void RNSPoly::generateSpecialLimbs(const bool zero_out) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].generateSpecialLimb(zero_out);
    }
}

void RNSPoly::multElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).multElement(poly.GPU.at(i));
    }
}

void RNSPoly::multElement(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).multElement(poly1.GPU.at(i), poly2.GPU.at(i));
    }
}

RNSPoly RNSPoly::clone(bool single_malloc) const {
    CudaCheckErrorModNoSync;
    auto res = RNSPoly(cc, this->level, single_malloc);

#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        res.GPU.at(i).copyLimb(GPU.at(i));
    }
    return res;
}

void RNSPoly::generateDecompAndDigit(bool iskey) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].generateAllDecompAndDigit(iskey);
    }
}

void RNSPoly::mult1AddMult23Add4(const RNSPoly& poly1, const RNSPoly& poly2, const RNSPoly& poly3,
                                 const RNSPoly& poly4) {

#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).mult1AddMult23Add4(poly1.GPU.at(i), poly2.GPU.at(i), poly3.GPU.at(i), poly4.GPU.at(i));
    }
}

void RNSPoly::mult1Add2(const RNSPoly& poly1, const RNSPoly& poly2) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).mult1Add2(poly1.GPU.at(i), poly2.GPU.at(i));
    }
}

void RNSPoly::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                              const std::vector<std::vector<uint64_t>>& moduli) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).loadDecompDigit(data, moduli);
    }
}

void RNSPoly::dotKSKinto(RNSPoly& acc, const RNSPoly& ksk, const RNSPoly* limbsrc) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        acc.GPU.at(i).dotKSK(GPU.at(i), ksk.GPU.at(i), false, limbsrc ? &limbsrc->GPU.at(i) : nullptr);
    }
}

void RNSPoly::multModupDotKSK(RNSPoly& c1, const RNSPoly& c1tilde, RNSPoly& c0, const RNSPoly& c0tilde,
                              const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "multModupDotKSK Multi-GPU not implemented.");
    assert(c1.level <= c1tilde.level);
    if (cc.rescaleTechnique == Context::FLEXIBLEAUTO || cc.rescaleTechnique == Context::FLEXIBLEAUTOEXT) {
        assert(level == c1.level);
        assert(level == c1tilde.level);
        assert(level == c0.level);
        assert(level == c0tilde.level);
    }
    generateDecompAndDigit(false);
    c0.generateSpecialLimbs(false);
    c1.generateSpecialLimbs(false);
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).multModupDotKSK(c1.GPU.at(i), c1tilde.GPU.at(i), c0.GPU.at(i), c0tilde.GPU.at(i), key.a.GPU.at(i),
                                  key.b.GPU.at(i));
    }
    c0.SetModUp(true);
    c1.SetModUp(true);
}

void RNSPoly::rotateModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "rotateModupDotKSK Multi-GPU not implemented.");
    generateDecompAndDigit(false);
    c0.generateSpecialLimbs(false);
    c1.generateSpecialLimbs(false);
    for (size_t i = 0; i < GPU.size(); ++i) {
        GPU.at(i).rotateModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
    }
    c1.SetModUp(true);
    c0.SetModUp(true);
}

template <ALGO algo>
void RNSPoly::moddown(bool ntt, bool free) {
    if (!this->isModUp()) {
        std::cout << "RNSPoly calling MOdDown on non-modup polynomial." << std::endl;
    }
    assert(this->isModUp());

    if (cc.GPUid.size() == 1) {
        for (int i = 0; i < (int)GPU.size(); ++i) {
            GPU.at(i).moddown<algo>(cc.getModdownAux(0).GPU.at(i), ntt, free);
        }
    } else {
        RNSPoly& aux = cc.getModdownAux(0);
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            if (cc.specialMeta.at(i).size() > cc.splitSpecialMeta.at(i).size()) {
                GPU.at(i).moddownMGPU(aux.GPU.at(i), ntt, free);
            } else {
                GPU.at(i).moddown(aux.GPU.at(i), ntt, free);
            }
        }
        //assert(nullptr == "ModDown Multi-GPU not implemented.");
    }
    this->SetModUp(false);
}

#define YY(algo) template void RNSPoly::moddown<algo>(bool ntt, bool free);

#include "ntt_types.inc"

#undef YY

void RNSPoly::automorph(const int idx, const int br, RNSPoly* src) {
    int k = modpow(5, idx, cc.N * 2);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).automorph(k, br, src ? &src->GPU.at(i) : nullptr, src ? src->isModUp() : this->isModUp());
    }
    if (src)
        this->SetModUp(src->isModUp());
}

void RNSPoly::automorph_multi(const int idx, const int br) {
    int k = modpow(5, idx, cc.N * 2);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).automorph_multi(k, br);
    }
}

RNSPoly& RNSPoly::dotKSKInPlace(const KeySwitchingKey& ksk, RNSPoly* limb_src) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    if (cc.GPUid.size() == 1) {
        if (limb_src) {
            std::cerr << "RNSPoly::dotKSKInPlace: limb_src: parameter ignored, fix this" << std::endl;
        }
        //RNSPoly result{RNSPoly(cc, level, true)};
        cc.getKeySwitchAux2().setLevel(level);
        cc.getKeySwitchAux2().generateSpecialLimbs(false);
        generateSpecialLimbs(false);
        if constexpr (PRINT)
            for (auto& i : ksk.b.GPU) {
                for (auto& j : i.DECOMPlimb) {
                    for (auto& k : j) {
                        SWITCH(k, printThisLimb(1));
                    }
                }

                for (auto& j : i.DIGITlimb) {
                    for (auto& k : j) {
                        SWITCH(k, printThisLimb(1));
                    }
                }
            }
        //dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level);
        //dotKSKInPlace(ksk.a, level);

        this->dotKSKfused(cc.getKeySwitchAux2(), *this, ksk.a, ksk.b, limb_src ? limb_src : this);
    } else {

        RNSPoly& aux = cc.getKeySwitchAux2();
        aux.setLevel(level);
        aux.generateSpecialLimbs(false);
        generateSpecialLimbs(false);
        this->dotKSKfused(aux, *this, ksk.a, ksk.b, limb_src ? limb_src : this);
    }

    this->SetModUp(true);
    cc.getKeySwitchAux2().SetModUp(true);
    Out(KEYSWITCH, "dotKSK out");
    return cc.getKeySwitchAux2();
}

/*
void RNSPoly::dotKSKInPlace(const RNSPoly& ksk_b, int level) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).dotKSK(GPU.at(i), ksk_b.GPU.at(i), level, true);
    }
}
*/

void RNSPoly::setLevel(const int level) {
    assert(level >= -1 && level <= cc.L);
    this->level = level;
}
void RNSPoly::modupInto(RNSPoly& poly) {
    assert(level == poly.level);
    auto& aux = cc.getKeySwitchAux2();
    aux.setLevel(level);

#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).modupInto(poly.GPU.at(i), aux.GPU.at(i));
    }
    poly.SetModUp(true);
}
RNSPoly& RNSPoly::dotKSKInPlaceFrom(RNSPoly& poly, const KeySwitchingKey& ksk, const RNSPoly* limbsrc) {
    constexpr bool PRINT = false;
    Out(KEYSWITCH, "dotKSK in");

    assert(level == poly.level);
    cc.getKeySwitchAux2().setLevel(level);
    cc.getKeySwitchAux2().generateSpecialLimbs(false);
    generateSpecialLimbs(false);
    if constexpr (PRINT)
        for (auto& i : ksk.b.GPU) {
            for (auto& j : i.DECOMPlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
            for (auto& j : i.DIGITlimb) {
                for (auto& k : j) {
                    SWITCH(k, printThisLimb(1));
                }
            }
        }
    //poly.dotKSKinto(cc.getKeySwitchAux2(), ksk.b, level, limbsrc ? limbsrc : this);
    //poly.dotKSKinto(*this, ksk.a, level, limbsrc ? limbsrc : this);

    this->dotKSKfused(cc.getKeySwitchAux2(), poly, ksk.a, ksk.b, limbsrc ? limbsrc : this);

    cc.getKeySwitchAux2().SetModUp(true);
    this->SetModUp(true);
    Out(KEYSWITCH, "dotKSK out");
    return cc.getKeySwitchAux2();
}
void RNSPoly::multScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].multScalar(vector1);
    }
}
void RNSPoly::add(const RNSPoly& a, const RNSPoly& b) {
    assert(level <= a.level);
    assert(level <= b.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).add(a.GPU.at(i), b.GPU.at(i), a.isModUp(), b.isModUp());
    }

    this->SetModUp(a.isModUp() || b.isModUp());
}
void RNSPoly::squareElement(const RNSPoly& poly) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).squareElement(poly.GPU.at(i));
    }
}
void RNSPoly::binomialSquareFold(RNSPoly& c0_res, const RNSPoly& c2_key_switched_0, const RNSPoly& c2_key_switched_1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).binomialSquareFold(c0_res.GPU.at(i), c2_key_switched_0.GPU.at(i), c2_key_switched_1.GPU.at(i));
    }
}
void RNSPoly::addScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].addScalar(vector1);
    }
}
void RNSPoly::subScalar(std::vector<uint64_t>& vector1) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].subScalar(vector1);
    }
}
void RNSPoly::copy(const RNSPoly& poly) {
    //std::cout << "Copy level: " << poly.level << std::endl;
    this->dropToLevel(poly.level);
    this->grow(poly.level, true);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).copyLimb(poly.GPU.at(i));
        if (poly.isModUp())
            GPU.at(i).copySpecialLimb(poly.GPU.at(i));
    }
    this->SetModUp(poly.isModUp());
    //cudaDeviceSynchronize();
}

/** Copy contents without extra checks or resizing */
void RNSPoly::copyShallow(const RNSPoly& poly) {
    this->level = poly.level;
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).copyLimb(poly.GPU.at(i));
    }
    //cudaDeviceSynchronize();
}

void RNSPoly::dropToLevel(int level) {

    if (GPU.at(0).bufferLIMB == nullptr) {
        for (auto& g : GPU) {
            cudaSetDevice(g.device);
            int limbSize = g.getLimbSize(level);
            while ((int)g.limb.size() > limbSize) {
                g.dropLimb();
            }
        }
    }
    if (this->level > level)
        this->level = level;
}
void RNSPoly::addMult(const RNSPoly& poly, const RNSPoly& poly1) {
    assert(level <= poly1.level && level <= poly.level);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).addMult(poly.GPU.at(i), poly1.GPU.at(i));
    }
}

void RNSPoly::load(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
    int limbsize = 0;
    int Slimbsize = 0;
    for (int i = 0; i < (int)data.size(); ++i) {
        if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
            limbsize++;
        } else {
            Slimbsize++;
        }
    }
    //std::cout << "Load " << limbsize << " limbs" << std::endl;

    assert(limbsize - 1 <= cc.L);
    if (level < limbsize - 1)
        grow(limbsize - 1, true);
    assert(level == limbsize - 1);
    for (int i = 0; i < limbsize; ++i) {
        //std::cout << "Load limb" << i << " into gpu " << cc.limbGPUid[i].x << std::endl;
        assert(moduli[i] == cc.prime.at(i).p);
        cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
    }

    if ((int)data.size() > limbsize)
        generateSpecialLimbs(false);
    for (int i = limbsize; i < (int)data.size(); ++i) {
        for (auto& j : GPU) {
            cudaSetDevice(j.device);
            assert(moduli[i] == cc.specialPrime.at(i - limbsize).p);
            SWITCH(j.SPECIALlimb[i - limbsize], load_convert(data[i]));
        }
    }
}

void RNSPoly::loadConstant(const std::vector<std::vector<uint64_t>>& data, const std::vector<uint64_t>& moduli) {
    int limbsize = 0;
    int Slimbsize = 0;
    for (int i = 0; i < (int)data.size(); ++i) {
        if (i <= cc.L && moduli[i] == cc.prime.at(i).p) {
            limbsize++;
        } else {
            Slimbsize++;
        }
    }

    assert(limbsize - 1 <= cc.L);
    if (level < limbsize - 1) {
        grow(limbsize - 1, true, true);
    } else {
        dropToLevel(limbsize - 1);
    }
    assert(level == limbsize - 1);
    for (int i = 0; i < limbsize; ++i) {
        assert(moduli[i] == cc.prime.at(i).p);
        cudaSetDevice(GPU[cc.limbGPUid[i].x].device);
        SWITCH(GPU[cc.limbGPUid[i].x].limb[cc.limbGPUid[i].y], load_convert(data[i]));
    }

    if ((int)data.size() > limbsize) {
        generatePartialSpecialLimbs();
        this->SetModUp(true);
    }
    for (size_t i = limbsize; i < data.size(); ++i) {
        for (size_t j = 0; j < GPU.size(); ++j) {
            for (size_t k = 0; k < cc.splitSpecialMeta.at(j).size(); ++k) {
                if (cc.specialPrime.at(cc.splitSpecialMeta.at(j).at(k).id - cc.L - 1).p == moduli[i]) {
                    cudaSetDevice(GPU[j].device);
                    SWITCH(GPU[j].SPECIALlimb[k], load_convert(data[i]));
                }
            }
        }
    }
}

void RNSPoly::broadcastLimb0() {
    if (cc.GPUid.size() == 1) {
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            GPU.at(i).broadcastLimb0();
        }
    } else {
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t i = 0; i < cc.GPUid.size(); ++i) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            GPU.at(i).broadcastLimb0_mgpu();
        }
    }
}
void RNSPoly::evalLinearWSum(uint32_t n, std::vector<const RNSPoly*>& vec, std::vector<uint64_t>& elem) {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        std::vector<const LimbPartition*> ps(n);
        for (int j = 0; j < (int)n; ++j) {
            ps[j] = &vec[j]->GPU.at(i);
        }
        GPU.at(i).evalLinearWSum(n, ps, elem);
    }
}

void RNSPoly::squareModupDotKSK(RNSPoly& c0, RNSPoly& c1, const KeySwitchingKey& key) {
    assert(GPU.size() == 1 && "squareModupDotKSK Multi-GPU not implemented.");
    generateDecompAndDigit(false);
    c0.generateSpecialLimbs(false);
    c1.generateSpecialLimbs(false);
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU.at(i).squareModupDotKSK(c1.GPU.at(i), c0.GPU.at(i), key.a.GPU.at(i), key.b.GPU.at(i));
    }
    c0.SetModUp(true);
    c1.SetModUp(true);
}

void RNSPoly::generatePartialSpecialLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[i].generatePartialSpecialLimb();
    }
}
void RNSPoly::dotKSKfused(RNSPoly& out2, const RNSPoly& digitSrc, const RNSPoly& ksk_a, const RNSPoly& ksk_b,
                          const RNSPoly* source) {
    RNSPoly& out1 = *this;
    const RNSPoly& src = source ? *source : *this;
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t i = 0; i < cc.GPUid.size(); ++i) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        if (cc.GPUid.size() == 1) {
            out1.GPU[i].dotKSKfused(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
        } else {
            out1.GPU[i].dotKSKfusedMGPU(out2.GPU[i], digitSrc.GPU[i], ksk_a.GPU[i], ksk_b.GPU[i], src.GPU[i]);
        }
    }
}
void RNSPoly::dotProductPt(RNSPoly& c1_, const std::vector<const RNSPoly*>& c0s_,
                           const std::vector<const RNSPoly*>& c1s_, const std::vector<const RNSPoly*>& pts_,
                           const bool ext) {

    if (ext) {
        generateSpecialLimbs(false);
        c1_.generateSpecialLimbs(false);
    }
    int n = pts_.size();
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t j = 0; j < cc.GPUid.size(); ++j) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        std::vector<const LimbPartition*> c0s(n, nullptr), c1s(n, nullptr), pts(n, nullptr);
        for (int i = 0; i < n; ++i) {
            c0s[i] = &(c0s_[i]->GPU[j]);
            c1s[i] = &(c1s_[i]->GPU[j]);
            pts[i] = &(pts_[i]->GPU[j]);
        }
        GPU[j].dotProductPt(c1_.GPU[j], c0s, c1s, pts, ext);
    }
    c1_.SetModUp(ext);
    this->SetModUp(ext);
}
void RNSPoly::gatherAllLimbs() {

    if constexpr (0) {
        for (int d = 0; d < cc.dnum; ++d) {

            for (size_t j = 0; j < cc.GPUid.size(); ++j) {
                cudaSetDevice(GPU[j].device);
                for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                    cc.gatherStream[i][j].wait(GPU[j].s);
                }
            }
#ifdef NCCL
            ncclGroupStart();
            //#pragma omp parallel for num_threads(cc.GPUid.size())
            for (size_t j = 0; j < cc.GPUid.size(); ++j) {
                assert(omp_get_num_threads() == (int)cc.GPUid.size());
                cudaSetDevice(GPU[j].device);
                int start = 0;
                for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                    ncclBroadcast(GPU[j].bufferLIMB, GPU[j].bufferGATHER + cc.N * start, GPU[i].limb.size() * cc.N,
                                  ncclUint64, i, GPU[j].rank, cc.gatherStream[i][j].ptr);
                    start += cc.meta[i].size();
                }
            }
            ncclGroupEnd();
#else
            assert(false);
#endif
            for (size_t j = 0; j < cc.GPUid.size(); ++j) {
                assert(omp_get_num_threads() == (int)cc.GPUid.size());
                cudaSetDevice(GPU[j].device);
                for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                    GPU[j].s.wait(cc.gatherStream[i][j]);
                }
            }
        }
    } else {
        for (size_t j = 0; j < cc.GPUid.size(); ++j) {
            cudaSetDevice(GPU[j].device);
            for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                cc.gatherStream[i][j].wait(GPU[j].s);
            }
        }
#ifdef NCCL
        ncclGroupStart();
        //#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t j = 0; j < cc.GPUid.size(); ++j) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            cudaSetDevice(GPU[j].device);
            int start = 0;
            for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                int size = 0;
                while (size < (int)GPU[i].limb.size() && cc.meta[i].at(size).id <= level)
                    ++size;

                ncclBroadcast(GPU[j].bufferLIMB, GPU[j].bufferGATHER + cc.N * start, size * cc.N, ncclUint64, i,
                              GPU[j].rank, cc.gatherStream[i][j].ptr);
                start += cc.meta[i].size();
            }
        }
        ncclGroupEnd();
#else
        assert(false);
#endif
        for (size_t j = 0; j < cc.GPUid.size(); ++j) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            cudaSetDevice(GPU[j].device);
            for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                GPU[j].s.wait(cc.gatherStream[i][j]);
            }
        }
    }
}

/*
void RNSPoly::generateGatherLimbs() {
#pragma omp parallel for num_threads(cc.GPUid.size())
    for (size_t j = 0; j < cc.GPUid.size(); ++j) {
        assert(omp_get_num_threads() == (int)cc.GPUid.size());
        GPU[j].generateGatherLimb(false);
    }
}
*/

RNSPoly& RNSPoly::modup_ksk_moddown_mgpu(const KeySwitchingKey& key, const bool moddown) {
    RNSPoly& aux = cc.getKeySwitchAux2();
    aux.setLevel(level);
    RNSPoly& aux_limbs1 = cc.getModdownAux(0);
    RNSPoly& aux_limbs2 = cc.getModdownAux(1);
    if (1 || moddown) {
#pragma omp parallel for num_threads(cc.GPUid.size())
        for (size_t j = 0; j < cc.GPUid.size(); ++j) {
            assert(omp_get_num_threads() == (int)cc.GPUid.size());
            if (omp_get_num_threads() != (int)cc.GPUid.size()) {
                std::cerr << "Not enough threads!" << std::endl;
                exit(-1);
            }
            GPU[j].modup_ksk_moddown_mgpu(aux.GPU[j], key.a.GPU[j], key.b.GPU[j], aux_limbs1.GPU[j], aux_limbs2.GPU[j],
                                          moddown);
        }
        this->SetModUp(!moddown);
        aux.SetModUp(!moddown);
        return aux;
    } else {
        this->modup();
        RNSPoly& result = this->dotKSKInPlace(key, nullptr);
        if (moddown) {
            result.moddown(true, false);
            this->moddown(true, false);
        }
        return result;
    }
}

}  // namespace FIDESlib::CKKS