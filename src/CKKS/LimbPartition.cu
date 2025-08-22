//
// Created by carlosad on 27/04/24.
//
#include <algorithm>
#include <array>
#include <variant>
#include <vector>

#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"
#include "LimbUtils.cuh"
#include "NTT.cuh"
#include "Rotation.cuh"
#include "VectorGPU.cuh"

namespace FIDESlib::CKKS {

LimbPartition::LimbPartition(LimbPartition&& l) noexcept
    : cc(l.cc),
      level(l.level),
      id(l.id),
      device((cudaSetDevice(l.device), l.device)),
      rank(l.rank),
      s(std::move(l.s)),
      meta(l.meta),
      SPECIALmeta(l.SPECIALmeta),
      digitid(l.digitid),
      DECOMPmeta(l.DECOMPmeta),
      DIGITmeta(l.DIGITmeta),
      GATHERmeta(l.GATHERmeta),
      limb(std::move(l.limb)),
      SPECIALlimb(std::move(l.SPECIALlimb)),
      DECOMPlimb(std::move(l.DECOMPlimb)),
      DIGITlimb(std::move(l.DIGITlimb)),
      GATHERlimb(std::move(l.GATHERlimb)),
      bufferAUXptrs(l.bufferAUXptrs),
      limbptr(std::move(l.limbptr)),
      auxptr(std::move(l.auxptr)),

      SPECIALlimbptr(std::move(l.SPECIALlimbptr)),
      SPECIALauxptr(std::move(l.SPECIALauxptr)),
      DECOMPlimbptr(std::move(l.DECOMPlimbptr)),
      //      DECOMPauxptr(std::move(l.DECOMPlimbptr)),
      DIGITlimbptr(std::move(l.DIGITlimbptr)),
      //      DIGITauxptr(std::move(l.DIGITlimbptr)),
      GATHERptr(std::move(l.GATHERptr)),
      bufferDECOMPandDIGIT(l.bufferDECOMPandDIGIT),
      bufferSPECIAL(l.bufferSPECIAL),
      bufferLIMB(l.bufferLIMB),
      bufferGATHER(l.bufferGATHER),
      bufferDECOMPandDIGIT_handle(l.bufferDECOMPandDIGIT_handle),
      bufferGATHER_handle(l.bufferGATHER_handle) {
    l.bufferSPECIAL = nullptr;
    l.bufferLIMB = nullptr;
    l.bufferDECOMPandDIGIT = nullptr;
    l.bufferAUXptrs = nullptr;
    l.bufferGATHER = nullptr;
    l.bufferDECOMPandDIGIT_handle = nullptr;
    l.bufferGATHER_handle = nullptr;
}

std::vector<VectorGPU<void*>> LimbPartition::generateDecompLimbptr(
    void** buffer, const std::vector<std::vector<LimbRecord>>& DECOMPmeta, const int device, int offset) {
    std::vector<VectorGPU<void*>> result;
    for (auto& d : DECOMPmeta) {
        result.emplace_back(buffer, d.size(), device, offset);
        offset += MAXP;
    }
    return result;
}

void** CudaMallocAuxBuffer(const Stream& stream, unsigned long size);

void** CudaMallocAuxBuffer(const Stream& stream, unsigned long size) {
    void** malloc;
    CudaCheckErrorModNoSync;
    cudaMallocAsync(&malloc, MAXP * sizeof(void*) * (4 + 4 * size), stream.ptr);
    CudaCheckErrorModNoSync;
    return malloc;
}

Stream initStream() {
    Stream s;
    s.init(50);
    return s;
}

LimbPartition::LimbPartition(Context& cc, int* level, const int id)
    : cc(cc),
      level(level),
      id(id),
      device((cudaSetDevice(cc.GPUid.at(id)), cc.GPUid.at(id))),
      rank(cc.GPUrank.at(id)),
      s(initStream()),
      meta(cc.meta.at(id)),
      SPECIALmeta(cc.specialMeta.at(id)),
      digitid(cc.GPUdigits.at(id)),
      DECOMPmeta(cc.decompMeta.at(id)),
      DIGITmeta(cc.digitMeta.at(id)),
      GATHERmeta(cc.gatherMeta),
      DECOMPlimb(DECOMPmeta.size()),
      DIGITlimb(DIGITmeta.size()),
      bufferAUXptrs(CudaMallocAuxBuffer(s, DECOMPmeta.size())),
      /*
            limbptr(s, meta.size(), device),
            auxptr(s, meta.size(), device),
            SPECIALlimbptr(s, SPECIALmeta.size(), device),
            SPECIALauxptr(s, SPECIALmeta.size(), device),
            */

      limbptr(bufferAUXptrs, meta.size(), device, 0),
      auxptr(bufferAUXptrs, meta.size(), device, MAXP),
      SPECIALlimbptr(bufferAUXptrs, SPECIALmeta.size(), device, 2 * MAXP),
      SPECIALauxptr(bufferAUXptrs, SPECIALmeta.size(), device, 3 * MAXP),

      DECOMPlimbptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, 4 * MAXP)),
      //      DECOMPauxptr(generateDecompLimbptr(bufferAUXptrs, DECOMPmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
      DIGITlimbptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + DECOMPmeta.size()) * MAXP)),
      //      , DIGITauxptr(generateDecompLimbptr(bufferAUXptrs, DIGITmeta, device, (4 + 3 * DECOMPmeta.size()) * MAXP))
      GATHERptr(bufferAUXptrs, GATHERmeta.size(), device, (4 + 2 * DECOMPmeta.size()) * MAXP) {}

LimbPartition::~LimbPartition() {

    cudaSetDevice(device);
    // CudaCheckErrorMod;
    /*
        for (auto &i: limb) s.wait(STREAM(i));
        for (auto &i: SPECIALlimb) s.wait(STREAM(i));
        for (auto &i: DECOMPlimb) for (auto &j: i) s.wait(STREAM(j));
        for (auto &i: DIGITlimb) for (auto &j: i) s.wait(STREAM(j));
*/

    limbptr.free(s);
    auxptr.free(s);
    SPECIALlimbptr.free(s);
    SPECIALauxptr.free(s);
    GATHERptr.free(s);
    for (auto& d : DECOMPlimbptr)
        d.free(s);
    //    for (auto& d : DECOMPauxptr)
    //        d.free(s);
    for (auto& d : DIGITlimbptr)
        d.free(s);
    // for (auto& d : DIGITauxptr)
    //    d.free(s);

    if (bufferDECOMPandDIGIT_handle) {
        cudaStreamSynchronize(s.ptr);
#ifdef NCCL
        NCCLCHECK(ncclCommDeregister(rank, bufferDECOMPandDIGIT_handle));
        NCCLCHECK(ncclMemFree(bufferDECOMPandDIGIT));
#else
        assert(false);
#endif
    } else {
        if (bufferDECOMPandDIGIT)
            cudaFreeAsync(bufferDECOMPandDIGIT, s.ptr);
    }
    if (bufferSPECIAL)
        cudaFreeAsync(bufferSPECIAL, s.ptr);
    if (bufferLIMB)
        cudaFreeAsync(bufferLIMB, s.ptr);
    if (bufferAUXptrs)
        cudaFreeAsync(bufferAUXptrs, s.ptr);
    if (bufferGATHER_handle) {
        cudaStreamSynchronize(s.ptr);
#ifdef NCCL
        NCCLCHECK(ncclCommDeregister(rank, bufferGATHER_handle));
        NCCLCHECK(ncclMemFree(bufferGATHER));
#else
        assert(false);
#endif
    } else {
        if (bufferGATHER)
            cudaFreeAsync(bufferGATHER, s.ptr);
    }
    // CudaCheckErrorMod;
}

void LimbPartition::generate(std::vector<LimbRecord>& records, std::vector<LimbImpl>& limbs, VectorGPU<void*>& ptrs,
                             int pos, VectorGPU<void*>* auxptrs, uint64_t* buffer, size_t offset, uint64_t* buffer_aux,
                             size_t offset_aux) {
    constexpr bool USE_PARTITION_STREAM = true;
    assert(pos < (int)records.size());
    cudaSetDevice(device);

    const int limbs_size = limbs.size();
    int size = std::max((int)(pos - limbs_size + 1), (int)0);
    std::vector<void*> cpu_ptr(size, nullptr);
    std::vector<void*> cpu_auxptr(size, nullptr);
    for (int i = limbs_size; i <= pos; ++i) {
        const LimbRecord& r = records.at(i);
        if (r.type == U32) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, device,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id,
                                                  (uint32_t*)buffer_aux, 2 * offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(Limb<uint32_t>(cc, (uint32_t*)buffer, 2 * offset, device,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else
                limbs.emplace_back(Limb<uint32_t>(cc, device, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id));
            cpu_ptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U32>(limbs.back()).aux.data)[0]};
        }
        if (r.type == U64) {
            if (buffer && buffer_aux) {
                limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, device,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, buffer_aux,
                                                  offset_aux));
                offset += cc.N;
                offset_aux += cc.N;
            } else if (buffer) {
                limbs.emplace_back(Limb<uint64_t>(cc, buffer, offset, device,
                                                  USE_PARTITION_STREAM ? s : records.at(i).stream, r.id, nullptr, 0));
                offset += cc.N;
            } else
                limbs.emplace_back(Limb<uint64_t>(cc, device, USE_PARTITION_STREAM ? s : records.at(i).stream, r.id));

            cpu_ptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).v.data)[0]};
            cpu_auxptr[i - limbs_size] = {&(std::get<U64>(limbs.back()).aux.data)[0]};
            //cudaFreeHost(aux);
        }
        s.wait(USE_PARTITION_STREAM ? s : records.at(i).stream);
    }
    if (size > 0) {
        cudaMemcpyAsync(ptrs.data + limbs_size, cpu_ptr.data(), size * sizeof(void*), cudaMemcpyHostToDevice, s.ptr);
        if (auxptrs) {
            cudaMemcpyAsync((*auxptrs).data + limbs_size, cpu_auxptr.data(), size * sizeof(void*),
                            cudaMemcpyHostToDevice, s.ptr);
        }
    }
    CudaCheckErrorModNoSync;
}

void LimbPartition::generateLimb() {
    cudaSetDevice(device);
    generate(meta, limb, limbptr, (int)limb.size(), &auxptr);
}

/*
void LimbPartition::generateAllDecompLimb(uint64_t* pInt, size_t offset) {
    cudaSetDevice(device);
    DECOMPlimb.resize(DECOMPmeta.size());
    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
        generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)DECOMPmeta[i].size() - 1,
                 nullptr, pInt, offset, nullptr, 0);
        offset += cc.N * DECOMPmeta.at(i).size();
    }
}

*/

void LimbPartition::generateAllDigitLimb(uint64_t* pInt, size_t offset) {
    cudaSetDevice(device);
    DIGITlimb.resize(DIGITmeta.size());
    for (size_t i = 0; i < DIGITmeta.size(); ++i) {
        generate(DIGITmeta[i], DIGITlimb[i], DIGITlimbptr[i], (int)DIGITmeta[i].size() - 1, nullptr /*&DIGITauxptr[i]*/,
                 pInt, offset, nullptr, 0);
        offset += cc.N * DIGITmeta.at(i).size();
    }
}

void LimbPartition::generateSpecialLimb(const bool zero_out) {
    cudaSetDevice(device);
    if (bufferSPECIAL == nullptr) {
        cudaMallocAsync(&bufferSPECIAL, cc.N * SPECIALmeta.size() * 2 * sizeof(uint64_t), s.ptr);
        generate(SPECIALmeta, SPECIALlimb, SPECIALlimbptr, (int)SPECIALmeta.size() - 1, &SPECIALauxptr, bufferSPECIAL,
                 0, bufferSPECIAL, cc.N * SPECIALmeta.size());
    }
    if (zero_out)
        cudaMemsetAsync(bufferSPECIAL, 0, cc.N * SPECIALmeta.size() * sizeof(uint64_t), s.ptr);
}

template <ALGO algo, NTT_MODE mode>
void ApplyNTT(int batch, LimbPartition::NTT_fusion_fields fields, std::vector<LimbImpl>& limb,
              VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, Context& cc, const int primeid_init,
              const int limbsize = -1) {
    constexpr int M = 4;

    const dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
    const dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
    const int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    const int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
    const int size = (limbsize != -1 ? limbsize : limb.size()) - (mode == NTT_RESCALE || mode == NTT_MULTPT);

    for (int i = 0; i < size; i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

        NTT_<false, algo, mode>
            <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr>>>(
                (mode == NTT_RESCALE || mode == NTT_MULTPT) ? limbptr.data + size
                : (mode == NTT_MODDOWN)                     ? fields.op2->limbptr.data + i
                                                            : limbptr.data + i,
                primeid_init + i, auxptr.data + i, nullptr,
                (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0, nullptr, nullptr);

        NTT_<true, algo, mode><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                                 STREAM(limb.at(i)).ptr>>>(
            auxptr.data + i, primeid_init + i, limbptr.data + i,
            mode == NTT_MULTPT ? fields.pt->limbptr.data + i : nullptr,
            (mode == NTT_RESCALE || mode == NTT_MULTPT) ? PRIMEID(limb[size]) : 0, nullptr, nullptr);
    }
}

template <ALGO algo, NTT_MODE mode>
void LimbPartition::NTT(int batch, bool sync, NTT_fusion_fields fields) {
    cudaSetDevice(device);
    int limbsize = getLimbSize(*level);

    if (batch >= 1) {
        if (limbsize > 0) {
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    STREAM(limb[i]).wait(s);
                }
            }
            ApplyNTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    s.wait(STREAM(limb[i]));
                }
            }
        }
    } else {
        assert("Invalid NTT batch configuration!");
    }
}

#define YYY(algo, mode) template void LimbPartition::NTT<algo, mode>(int batch, bool sync, NTT_fusion_fields fields);

#include "ntt_types.inc"
#undef YYY

template <ALGO algo, INTT_MODE mode>
void ApplyINTT(int batch, LimbPartition::INTT_fusion_fields fields, std::vector<LimbImpl>& limb,
               VectorGPU<void*>& limbptr, VectorGPU<void*>& auxptr, Context& cc, const int primeid_init,
               const int limbsize) {
    constexpr int M = 4;

    dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN - (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1 + (cc.logN > 13 ? 0 : 0)) / 2 - 1))};
    int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
    int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

    for (size_t i = 0; i < limbsize; i += batch) {
        uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(limbsize - i));

        INTT_<false, algo, INTT_NONE>
            <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst, STREAM(limb.at(i)).ptr>>>(
                limbptr.data + i, primeid_init + i, auxptr.data + i);

        INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                                       STREAM(limb.at(i)).ptr>>>(auxptr.data + i, primeid_init + i, limbptr.data + i);
    }
}

template <ALGO algo, INTT_MODE mode>
void LimbPartition::INTT(int batch, bool sync, INTT_fusion_fields fields) {
    cudaSetDevice(device);
    // TODO check level
    const int limbsize = getLimbSize(*level);
    if (batch >= 1) {
        if (limbsize > 0) {
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    STREAM(limb[i]).wait(s);
                }
            }
            ApplyINTT<algo, mode>(batch, fields, limb, limbptr, auxptr, cc, PARTITION(id, 0), limbsize);
            if (sync) {
                for (int i = 0; i < limbsize; i += batch) {
                    s.wait(STREAM(limb[i]));
                }
            }
        }
    } else {
        assert("Invalid INTT batch configuration!");
    }
}

#define WWW(algo, mode) template void LimbPartition::INTT<algo, mode>(int batch, bool sync, INTT_fusion_fields fields);

#include "ntt_types.inc"
#undef WWW

void LimbPartition::add(const LimbPartition& p, const bool ext) {
    cudaSetDevice(device);
    const int limbsize = getLimbSize(*level);
    s.wait(p.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);
            {
                add_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                    SPECIALlimbptr.data + i, p.SPECIALlimbptr.data + i,
                    SPECIAL(
                        id,
                        i));  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            }
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            s.wait(STREAM(SPECIALlimb[i]));
        }
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::scaleByP() {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        scaleByP_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(limbptr.data + i,
                                                                                          PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
}

void LimbPartition::sub(const LimbPartition& p) {
    cudaSetDevice(device);
    const int limbsize = getLimbSize(*level);
    s.wait(p.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::multElement(const LimbPartition& p) {
    cudaSetDevice(device);

    int limbsize = getLimbSize(*level);
    assert(limbsize <= (int)p.limb.size());

    s.wait(p.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::multElement(const LimbPartition& partition1, const LimbPartition& partition2) {
    cudaSetDevice(device);
    int limbsize = getLimbSize(*level);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());

    s.wait(partition1.s);
    s.wait(partition2.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        Mult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            (void**)limbptr.data + i, (void**)partition1.limbptr.data + i, (void**)partition2.limbptr.data + i,
            PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    partition1.s.wait(s);
    partition2.s.wait(s);
}

void LimbPartition::rescale() {
    const int limbsize = getLimbSize(*level);
    assert(cc.GPUid.size() > 1 || limbsize > 1);
    if (limbsize == 0)
        return;

    cudaSetDevice(device);

    LimbImpl& top = limb.at(limbsize - 1);
    STREAM(top).wait(s);
    SWITCH(top, INTT<ALGO_SHOUP>());

    {
        for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
            STREAM(limb[i]).wait(STREAM(top));
        }
        NTT<ALGO_SHOUP, NTT_RESCALE>(cc.batch, false, NTT_fusion_fields{});
        for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
            STREAM(top).wait(STREAM(limb[i]));
        }
    }
    s.wait(STREAM(top));
    while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
        STREAM(limb.back()).wait(s);
        limb.pop_back();
    }
}

void LimbPartition::multPt(const LimbPartition& p) {
    const int limbsize = getLimbSize(*level);
    // assert(SPECIALlimb.size() == 0 && p.SPECIALlimb.size() == 0);
    assert(limbsize <= p.limb.size());
    assert(limbsize > 1);
    cudaSetDevice(device);

    constexpr bool capture = false;
    static std::map<int, cudaGraphExec_t> exec_map;

    {
        LimbImpl& top = limb.back();

        cudaGraphExec_t& exec = exec_map[limbsize];

        run_in_graph<capture>(exec, s, [&]() {
            STREAM(top).wait(s);
            SWITCH(top, mult(p.limb.back()));
            SWITCH(top, INTT<ALGO_SHOUP>());

            for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
                STREAM(limb.at(i)).wait(STREAM(top));
            }
            if (limbsize > 1)
                NTT<ALGO_SHOUP, NTT_MULTPT>(cc.batch, false, NTT_fusion_fields{.pt = &p});
            for (size_t i = 0; i < limbsize - 1; i += cc.batch) {
                STREAM(top).wait(STREAM(limb.at(i)));
            }

            s.wait(STREAM(top));
        });

        while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
            STREAM(limb.back()).wait(s);
            limb.pop_back();
        }
    }
}

void LimbPartition::modup(LimbPartition& aux_partition) {

    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    //assert(SPECIALlimb.empty());
    cudaSetDevice(device);

    const int limbsize = *level + 1;
    generateAllDecompAndDigit(false);
    s.wait(aux_partition.s);

    if constexpr (PRINT) {
        std::cout << "Before modup ";
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(2));
        }
        std::cout << std::endl;
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb.at(j).size();
        int size = std::min((int)DECOMPlimb.at(d).size(), limbsize - start);
        if (size <= 0)
            break;
        /*
        for (auto& l : DECOMPlimb[d]) {
            for (auto& p : limb) {
                if (PRIMEID(l) == PRIMEID(p)) {
                    STREAM(l).wait(STREAM(p));
                    SWITCH(l, INTT_from(p));
                    s_d.wait(STREAM(l));
                }
            }
        }
        */
        Stream& s_d = cc.digitStream.at(d).at(id);
        s_d.wait(s);

        {
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s_d);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    limbptr.data + start + i, PARTITION(id, start + i), auxptr.data + start + i);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    auxptr.data + start + i, PARTITION(id, start + i), DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s_d.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After INTT ";
            for (auto& i : DECOMPlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s_d.ptr>>>(DECOMPlimbptr[d].data, *level + 1,
                                                                                     DIGITlimbptr[d].data, digitid[d]);
        }
        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After conv ";
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }

        const int digitsize = host_constants_per_gpu[id].num_primeid_digit_to[digitid.at(d)][*level];
        for (size_t i = 0; i < digitsize; i += cc.batch) {
            STREAM(DIGITlimb.at(d).at(i)).wait(s_d);
        }
        ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, DIGITlimb.at(d), DIGITlimbptr.at(d),
                                 aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            s_d.wait(STREAM(DIGITlimb.at(d).at(i)));
        }

        if constexpr (PRINT) {
            cudaDeviceSynchronize();
            std::cout << "After NTT ";
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }
    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        s.wait(cc.digitStream[d][id]);
    }

    aux_partition.s.wait(s);
}

void LimbPartition::freeSpecialLimbs() {
    cudaSetDevice(device);
    for (size_t i = 0; i < SPECIALlimb.size(); ++i) {
        STREAM(SPECIALlimb.at(i)).wait(s);
    }
    SPECIALlimb.clear();
    if (bufferSPECIAL != nullptr) {
        cudaFreeAsync(bufferSPECIAL, s.ptr);
        bufferSPECIAL = nullptr;
    }
}

void LimbPartition::copyLimb(const LimbPartition& partition) {
    cudaSetDevice(device);
    s.wait(partition.s);
    int limbsize = getLimbSize(*level);
    assert(*level == *partition.level);
    //std::cout << "GPU: " << id << " copy " << limbsize << "limbs" << std::endl;
    if (limbsize > 0)
        copy_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize}, 128, 0, s.ptr>>>(partition.limbptr.data, limbptr.data);
    /*
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        STREAM(limb.at(i)).wait(s);
        SWITCH(limb.at(i), copyV(partition.limb.at(i)));
    }
    for (size_t i = 0; i < partition.limb.size(); ++i) {
        s.wait(STREAM(limb.at(i)));
    }
    */
    partition.s.wait(s);
}
void LimbPartition::copySpecialLimb(const LimbPartition& p) {

    cudaSetDevice(device);
    this->generateSpecialLimb(false);
    s.wait(p.s);
    assert(*level == *p.level);
    int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
    int num_limbs = cc.splitSpecialMeta.at(id).size();
    for (size_t i = start; i < start + num_limbs; i += cc.batch) {
        STREAM(SPECIALlimb[i]).wait(s);
        uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

        copy_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
            p.SPECIALlimbptr.data + i,
            SPECIALlimbptr.data +
                i);  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
    }
    for (size_t i = start; i < start + num_limbs; i += cc.batch) {
        s.wait(STREAM(SPECIALlimb[i]));
    }
    p.s.wait(s);
}

void LimbPartition::generateAllDecompAndDigit(bool iskey) {
    cudaSetDevice(device);
    if (bufferDECOMPandDIGIT == nullptr) {
        int decomp_limbs = 0;
        for (auto& d : DECOMPmeta)
            decomp_limbs += d.size();
        int digit_limbs = 0;
        for (auto& d : DIGITmeta)
            digit_limbs += d.size();

        size_t size = cc.N * (/*decomp_limbs +*/ digit_limbs);
        if (cc.GPUid.size() == 1 || iskey) {
            cudaMallocAsync(&bufferDECOMPandDIGIT, size * sizeof(uint64_t), s.ptr);
        } else {
            // TODO do not do for key switching keys
#ifdef NCCL
            cudaStreamSynchronize(s.ptr);
            NCCLCHECK(ncclMemAlloc((void**)&bufferDECOMPandDIGIT, size * sizeof(uint64_t)));
            NCCLCHECK(
                ncclCommRegister(rank, bufferDECOMPandDIGIT, size * sizeof(uint64_t), &bufferDECOMPandDIGIT_handle));
            cudaDeviceSynchronize();
#else
            assert(false);
#endif
        }
        generateGatherLimb(iskey);
        //generateAllDecompLimb(bufferDECOMPandDIGIT, 0);
        DECOMPlimb.resize(DECOMPmeta.size());
        for (size_t i = 0; i < DECOMPmeta.size(); ++i) {
            for (size_t j = 0; j < DECOMPmeta.at(i).size(); ++j) {
                int pos = 0;
                for (size_t k = 0; k < cc.meta.size(); ++k) {
                    for (size_t l = 0; l < cc.meta[k].size(); ++l) {
                        if (cc.meta[k][l].id == DECOMPmeta[i][j].id) {
                            generate(DECOMPmeta[i], DECOMPlimb[i], DECOMPlimbptr[i], (int)j, nullptr, bufferGATHER,
                                     pos * cc.N, nullptr, 0);
                        }
                        pos++;
                    }
                }
            }
        }
        generateAllDigitLimb(bufferDECOMPandDIGIT, 0 /*cc.N * decomp_limbs*/);
    }
}

void LimbPartition::mult1AddMult23Add4(const LimbPartition& partition1, const LimbPartition& partition2,
                                       const LimbPartition& partition3, const LimbPartition& partition4) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());
    assert(limbsize <= partition3.limb.size());
    assert(limbsize <= partition4.limb.size());

    s.wait(partition1.s);
    s.wait(partition2.s);
    s.wait(partition3.s);
    s.wait(partition4.s);

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        mult1AddMult23Add4_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i,
            partition3.limbptr.data + i, partition4.limbptr.data + i);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.s.wait(s);
    partition2.s.wait(s);
    partition3.s.wait(s);
    partition4.s.wait(s);
}

void LimbPartition::multNoModdownEnd(LimbPartition& c0, const LimbPartition& bc0, const LimbPartition& bc1,
                                     const LimbPartition& in, const LimbPartition& aux) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= c0.limb.size());
    assert(limbsize <= bc0.limb.size());
    assert(limbsize <= bc1.limb.size());
    assert(limbsize <= in.limb.size());
    assert(limbsize <= aux.limb.size());

    s.wait(c0.s);
    s.wait(bc0.s);
    s.wait(bc1.s);
    s.wait(in.s);
    s.wait(aux.s);
    c0.s.wait(aux.s);

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        multnomoddownend_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            PARTITION(id, i), limbptr.data + i, c0.limbptr.data + i, bc0.limbptr.data + i, bc1.limbptr.data + i,
            in.limbptr.data + i, aux.limbptr.data + i);
    }
    this->copySpecialLimb(in);
    c0.copySpecialLimb(aux);
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    aux.s.wait(c0.s);
    aux.s.wait(s);
    in.s.wait(s);
    bc1.s.wait(s);
    bc0.s.wait(s);
    c0.s.wait(s);
}

void LimbPartition::mult1Add2(const LimbPartition& partition1, const LimbPartition& partition2) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize <= partition1.limb.size());
    assert(limbsize <= partition2.limb.size());

    s.wait(partition1.s);
    s.wait(partition2.s);

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        mult1Add2_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            PARTITION(id, i), limbptr.data + i, partition1.limbptr.data + i, partition2.limbptr.data + i);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    partition1.s.wait(s);
    partition2.s.wait(s);
}

void LimbPartition::generateLimbSingleMalloc() {
    cudaSetDevice(device);
    const int limbsize = meta.size();

    assert(limbsize <= meta.size());
    if (bufferLIMB == nullptr) {
        assert(limb.size() == 0);
        cudaMallocAsync(&bufferLIMB, cc.N * limbsize * 2 * sizeof(uint64_t), s.ptr);
    }

    limb.clear();

    generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, bufferLIMB, 0, bufferLIMB, cc.N * (limbsize));
}

void LimbPartition::generateLimbConstant() {
    cudaSetDevice(device);
    const int limbsize = getLimbSize(*level);
    assert(limb.size() == 0);
    assert(limbsize <= meta.size());
    if (bufferLIMB == nullptr) {
        cudaMallocAsync(&bufferLIMB, cc.N * limbsize * sizeof(uint64_t), s.ptr);
    } else {
        cudaFreeAsync(&bufferLIMB, s.ptr);
        cudaMallocAsync(&bufferLIMB, cc.N * limbsize * sizeof(uint64_t), s.ptr);
    }

    limb.clear();
    generate(meta, limb, limbptr, (int)limbsize - 1, &auxptr, bufferLIMB, 0, nullptr, 0);
}

void LimbPartition::loadDecompDigit(const std::vector<std::vector<std::vector<uint64_t>>>& data,
                                    const std::vector<std::vector<uint64_t>>& moduli) {
    cudaSetDevice(device);

    for (size_t i = 0; i < DECOMPmeta.size(); ++i) {

        if (cc.GPUid.size() == 1) {
            for (auto& j : DECOMPlimb.at(i)) {
                for (size_t k = 0; k < data.at(i).size(); ++k) {
                    if (hC_.primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                        STREAM(j).wait(s);
                        SWITCH(j, load(data.at(i).at(k)));
                        k = data.at(i).size();
                    }
                }
            }
        } else {
            for (size_t j = 0; j < limb.size(); ++j) {
                if (meta[j].digit == i) {
                    for (size_t k = 0; k < data.at(i).size(); ++k) {
                        if (hC_.primes[PRIMEID(limb[j])] == moduli.at(i).at(k)) {
                            STREAM(limb[j]).wait(s);
                            SWITCH(limb[j], load(data.at(i).at(k)));
                            k = data.at(i).size();
                        }
                    }
                }
            }
        }

        for (auto& j : DIGITlimb.at(i)) {
            for (size_t k = 0; k < data.at(i).size(); ++k) {
                if (hC_.primes[PRIMEID(j)] == moduli.at(i).at(k)) {
                    STREAM(j).wait(s);
                    SWITCH(j, load(data.at(i).at(k)));
                    k = data.at(i).size();
                }
            }
        }
    }
}

/** TODO: deprecate towards fused version */

void LimbPartition::dotKSK(const LimbPartition& src, const LimbPartition& ksk, const bool inplace,
                           const LimbPartition* limbsrc) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    s.wait(src.s);
    s.wait(ksk.s);
    const int limbsize = *level + 1;
    assert(limbsize <= limb.size());
    assert(limbsize <= src.limb.size());

    if constexpr (0) {
        std::map<int, int> used;
        for (size_t i = 0; i < src.DIGITlimb.size(); ++i) {

            {
                int start = 0;
                for (int j = 0; j < i; ++j)
                    start += src.DECOMPlimb[j].size();
                int size = std::min((int)src.DECOMPlimb[i].size(), (int)limb.size() - start);
                if (size <= 0)
                    break;
            }

            for (size_t j = 0; j < ksk.DECOMPlimb.at(i).size(); ++j) {

                int primeid = PRIMEID(ksk.DECOMPlimb.at(i).at(j));

                for (size_t k = 0; k < src.limb.size(); ++k) {
                    auto& l = src.limb.at(k);
                    if (PRIMEID(l) == primeid) {
                        //STREAM(limb.at(k)).wait(s);
                        //STREAM(limb.at(k)).wait(STREAM(ksk.DECOMPlimb.at(i).at(j)));
                        //STREAM(limb.at(k)).wait(STREAM(l));

                        if (!used[primeid]) {
                            SWITCH(limb.at(k), mult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            used[primeid]++;
                            if constexpr (PRINT)
                                std::cout << "Init " << primeid;  //<< std::endl;
                        } else {
                            SWITCH(limb.at(k), addMult(l, ksk.DECOMPlimb.at(i).at(j), inplace));
                            if constexpr (PRINT)
                                std::cout << "Acc " << primeid;  // << std::endl;
                        }
                        if constexpr (PRINT)
                            SWITCH(limb.at(k), printThisLimb(1));
                        if constexpr (PRINT)
                            SWITCH(l, printThisLimb(1));
                    }
                }

                if constexpr (PRINT)
                    SWITCH(ksk.DECOMPlimb.at(i).at(j), printThisLimb(1));
            }

            CudaCheckErrorModNoSync;
            for (size_t j = 0; j < src.DIGITlimb.at(i).size(); ++j) {
                int primeid = PRIMEID(src.DIGITlimb.at(i).at(j));

                if (primeid < hC_.L) {
                    for (auto& l : limb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));

                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init2 " << primeid;  // << std::endl;
                            } else

                            {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc2 " << primeid;  // << std::endl;
                            }

                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                } else {

                    for (auto& l : SPECIALlimb) {
                        if (PRIMEID(l) == primeid) {
                            //STREAM(l).wait(s);
                            //STREAM(l).wait(STREAM(src.DIGITlimb.at(i).at(j)));
                            //STREAM(l).wait(STREAM(ksk.DIGITlimb.at(i).at(j)));
                            if (!used[primeid]) {
                                SWITCH(l, mult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                used[primeid]++;
                                if constexpr (PRINT)
                                    std::cout << "Init3 " << primeid;  // << std::endl;
                            } else {
                                SWITCH(l, addMult(src.DIGITlimb.at(i).at(j), ksk.DIGITlimb.at(i).at(j), inplace));
                                if constexpr (PRINT)
                                    std::cout << "Acc3 " << primeid;  // << std::endl;
                            }
                            if constexpr (PRINT)
                                SWITCH(l, printThisLimb(1));
                        }
                    }
                }
                if constexpr (PRINT)
                    SWITCH(src.DIGITlimb.at(i).at(j), printThisLimb(1));
                if constexpr (PRINT)
                    SWITCH(ksk.DIGITlimb.at(i).at(j), printThisLimb(1));
            }

            if constexpr (PRINT) {
                for (auto& i : limb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
                for (auto& i : SPECIALlimb) {
                    if constexpr (PRINT)
                        SWITCH(i, printThisLimb(1));
                }
            }
        }

        for (auto& l : limb)
            s.wait(STREAM(l));
        for (auto& l : SPECIALlimb)
            s.wait(STREAM(l));
    } else {

        int start = 0;
        int special = SPECIALmeta.size();

        for (int i = 0; i < DECOMPmeta.size(); ++i) {
            int size = std::min((int)src.DECOMPlimb[i].size(), (int)limbsize - start);
            if (size <= 0) {
                //  std::cout << "Out on " << i << std::endl;
                break;
            }
            Mult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(
                inplace ? auxptr.data + start : limbptr.data + start, ksk.DECOMPlimbptr[i].data,
                limbsrc ? limbsrc->limbptr.data + start : src.limbptr.data + start, start);
            start += DECOMPmeta[i].size();
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize) {
                // std::cout << "Out on " << i << std::endl;
                break;
            }
            if (start > 0) {
                int size = start;
                addMult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(inplace ? auxptr.data : limbptr.data,
                                                                ksk.DIGITlimbptr[i].data + special,
                                                                src.DIGITlimbptr[i].data + special, 0);
            }
            start += DECOMPmeta[i].size();
            if (start < limbsize) {
                int size = limbsize - start;
                addMult_<<<{cc.N / 128, size}, 128, 0, s.ptr>>>(
                    inplace ? auxptr.data + start : limbptr.data + start,
                    ksk.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(),
                    src.DIGITlimbptr[i].data + special + start - DECOMPmeta[i].size(), start);
            }
        }

        start = 0;
        for (int i = 0; i < DIGITmeta.size(); ++i) {
            if (start >= limbsize)
                break;
            start += DECOMPmeta.at(i).size();
            if (i == 0) {
                Mult_<<<{cc.N / 128, special}, 128, 0, s.ptr>>>(inplace ? SPECIALauxptr.data : SPECIALlimbptr.data,
                                                                ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data,
                                                                SPECIAL(id, 0));
            } else {
                addMult_<<<{cc.N / 128, special}, 128, 0, s.ptr>>>(inplace ? SPECIALauxptr.data : SPECIALlimbptr.data,
                                                                   ksk.DIGITlimbptr[i].data, src.DIGITlimbptr[i].data,
                                                                   SPECIAL(id, 0));
            }
        }
    }

    src.s.wait(s);
    ksk.s.wait(s);

    if (inplace) {
        for (auto& l : limb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(limbptr.data, auxptr.data);

        for (auto& l : SPECIALlimb) {
            if (l.index() == U32) {
                std::swap(std::get<U32>(l).v.data, std::get<U32>(l).aux.data);
            } else {
                std::swap(std::get<U64>(l).v.data, std::get<U64>(l).aux.data);
            }
        }
        std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
    }

    if constexpr (PRINT) {
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
        for (auto& i : SPECIALlimb) {
            SWITCH(i, printThisLimb(1));
        }
        std::cout << std::endl;
    }
}

/** TODO: deprecate towards MGPU version */
void LimbPartition::dotKSKfused(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a,
                                const LimbPartition& ksk_b, const LimbPartition& src) {
    cudaSetDevice(device);

    VectorGPU<void**> digits(s, cc.dnum * 6, device);
    std::vector<void**> h_digits(cc.dnum * 6, nullptr);
    LimbPartition& out1 = *this;
    cudaSetDevice(device);
    s.wait(digitSrc.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);
    s.wait(src.s);
    s.wait(out2.s);
    const int limbsize = getLimbSize(*level);

    if constexpr (1) {
        size_t i = 0;
        for (; i < src.DIGITlimb.size(); ++i) {
            {
                int start = 0;
                for (int j = 0; j < i; ++j)
                    start += DECOMPmeta[j].size();
                int size = std::min((int)DECOMPmeta[i].size(), (int)limbsize - start);
                if (size <= 0)
                    break;
            }

            h_digits[i] = digitSrc.DIGITlimbptr.at(i).data;
            h_digits[i + cc.dnum] = ksk_a.DIGITlimbptr.at(i).data;
            h_digits[i + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(i).data;
            h_digits[i + 3 * cc.dnum] = src.limbptr.data + DECOMPmeta.at(i).at(0).id;
            h_digits[i + 4 * cc.dnum] = ksk_a.DECOMPlimbptr.at(i).data;
            h_digits[i + 5 * cc.dnum] = ksk_b.DECOMPlimbptr.at(i).data;
        }

        int num_special = 0;
        while (num_special < (int)DIGITmeta.at(0).size() && DIGITmeta.at(0).at(num_special).id > cc.L)
            num_special++;
        int num_limbs = 0;
        while (num_limbs < (int)meta.size() && meta.at(num_limbs).id <= *level)
            num_limbs++;

        cudaMemcpyAsync(digits.data, h_digits.data(), cc.dnum * 6 * sizeof(void**), cudaMemcpyDefault, s.ptr);
        fusedDotKSK_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)num_special + num_limbs}, 128, 0, s.ptr>>>(
            out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data, digits.data, i,
            id, num_special);
    }
    digits.free(s);
    /*
    CudaCheckErrorMod;
    SWITCH(limb.at(0), printThisLimb());
    SWITCH(SPECIALlimb.at(0), printThisLimb());
    CudaCheckErrorMod;
  */
    src.s.wait(s);
    out2.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
    digitSrc.s.wait(s);
}

void LimbPartition::multModupDotKSK(LimbPartition& c1, const LimbPartition& c1tilde, LimbPartition& c0,
                                    const LimbPartition& c0tilde, const LimbPartition& ksk_a,
                                    const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(c0tilde.s);
    s.wait(c1tilde.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_MULT_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                         blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i,
                    c1tilde.limbptr.data + start + i, c0.limbptr.data + start + i, c1.limbptr.data + start + i,
                    ksk_a.DECOMPlimbptr[d].data + i, ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i,
                    c0tilde.limbptr.data + start + i);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level_plus_1,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    //cudaDeviceSynchronize();

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }
    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    c0tilde.s.wait(s);
    c1tilde.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}
int LimbPartition::getLimbSize(int level) {
    int size = 0;
    while (size < meta.size() && meta[size].id <= level) {
        //assert(limb.size() > size);
        ++size;
    }
    return size;
}

void LimbPartition::rotateModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_ROTATE_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                           blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                    c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                    ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level_plus_1,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}

void LimbPartition::squareModupDotKSK(LimbPartition& c1, LimbPartition& c0, const LimbPartition& ksk_a,
                                      const LimbPartition& ksk_b) {

    const int level_plus_1 = *level + 1;
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    assert(c0.SPECIALlimb.size() == SPECIALmeta.size());
    assert(c1.SPECIALlimb.size() == SPECIALmeta.size());
    cudaSetDevice(device);

    //std::map<int, int> used;
    s.wait(c0.s);
    s.wait(c1.s);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            if (d == 0) {
                std::cout << hC_.primes[PRIMEID(limb[0])] << ": ";
                SWITCH(limb[0], printThisLimb());
            }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            for (int i = 0; i < size; i += cc.batch) {
                STREAM(limb.at(start + i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                INTT_<false, algo, INTT_SQUARE_AND_SAVE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs},
                                                           blockDimFirst, bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                    c1.limbptr.data + start + i, start + i, c1.auxptr.data + start + i, nullptr,
                    c0.limbptr.data + start + i, c1.limbptr.data + start + i, ksk_a.DECOMPlimbptr[d].data + i,
                    ksk_b.DECOMPlimbptr[d].data + i, c0.limbptr.data + start + i, nullptr);

                INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                               bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                    c1.auxptr.data + start + i, start + i, DECOMPlimbptr[d].data + i);
            }
            for (size_t i = 0; i < size; i += cc.batch) {
                s.wait(STREAM(limb.at(start + i)));
            }
        }

        if constexpr (PRINT) {
            std::cout << hC_.primes[PRIMEID(DECOMPlimb[d][0])] << ": ";
            SWITCH(DECOMPlimb[d][0], printThisLimb());
        }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (DECOMPlimb[d].size()) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(DECOMPlimbptr[d].data, level_plus_1,
                                                                                   DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (1) {  // Batched
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            int size = c0.SPECIALlimb.size();
            for (int i = 0; i < size; i += cc.batch) {
                STREAM(c0.SPECIALlimb.at(i)).wait(s);
                uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                              bytesFirst, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                    DIGITlimbptr[d].data + i, SPECIAL(id, i), c1.SPECIALauxptr.data + i, nullptr, 0, nullptr, nullptr);

                if (d == 0) {
                    NTT_<true, algo, NTT_KSK_DOT><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                                    bytesSecond, STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                        c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                        ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i, ksk_b.DIGITlimbptr[d].data + i);
                } else {
                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.SPECIALlimb.at(i)).ptr>>>(
                            c1.SPECIALauxptr.data + i, SPECIAL(id, i), c0.SPECIALlimbptr.data + i,
                            ksk_a.DIGITlimbptr[d].data + i, 0, c1.SPECIALlimbptr.data + i,
                            ksk_b.DIGITlimbptr[d].data + i);
                }
            }
        }
    }

    for (size_t d = 0; d < DECOMPlimb.size(); ++d) {
        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPlimb[j].size();
        int size = std::min((int)DECOMPlimb[d].size(), level_plus_1 - start);
        if (size <= 0)
            break;

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        if constexpr (1)  // batched
        {
            int start = 0;
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                if (j == d)
                    continue;

                int Dstart = start + c0.SPECIALlimb.size();
                int Lstart = start + (j > d ? DECOMPlimb[d].size() : 0);

                int size = std::min((int)DECOMPlimb[j].size(), (int)level_plus_1 - Lstart);
                if (size <= 0)
                    break;

                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                for (int i = 0; i < size; i += cc.batch) {
                    STREAM(c0.limb.at(Lstart + i)).wait(s);
                    uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

                    NTT_<false, algo, NTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                                  bytesFirst, STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                        DIGITlimbptr[d].data + Dstart + i, Lstart + i, c1.auxptr.data + Lstart + i, nullptr, 0, nullptr,
                        nullptr);

                    NTT_<true, algo, NTT_KSK_DOT_ACC>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                           STREAM(c0.limb.at(Lstart + i)).ptr>>>(
                            c1.auxptr.data + Lstart + i, Lstart + i, c0.limbptr.data + Lstart + i,
                            ksk_a.DIGITlimbptr[d].data + Dstart + i, 0, c1.limbptr.data + Lstart + i,
                            ksk_b.DIGITlimbptr[d].data + Dstart + i);
                }

                start += DECOMPlimb[j].size();
            }
        }

        if constexpr (PRINT)
            for (auto& i : DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    for (auto& l : c0.limb)
        s.wait(STREAM(l));
    for (auto& l : c0.SPECIALlimb)
        s.wait(STREAM(l));

    c0.s.wait(s);
    c1.s.wait(s);
    ksk_a.s.wait(s);
    ksk_b.s.wait(s);
}

template <ALGO algo>
void LimbPartition::moddown(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs) {
    assert(SPECIALlimb.size() == SPECIALmeta.size());
    const int limbsize = *level + 1;
    cudaSetDevice(device);
    constexpr bool PRINT = false;

    s.wait(auxLimbs.s);
    {
        if constexpr (PRINT) {
            std::cout << "pre INTT Special GPU ";
            for (auto& i : SPECIALlimb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        if (ntt) {
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                STREAM(SPECIALlimb[i]).wait(s);
            }
            ApplyINTT<algo, INTT_NONE>(cc.batch, INTT_fusion_fields{}, SPECIALlimb, SPECIALlimbptr, SPECIALauxptr, cc,
                                       SPECIAL(id, 0), SPECIALlimb.size());
            for (size_t i = 0; i < SPECIALlimb.size(); i += cc.batch) {
                s.wait(STREAM(SPECIALlimb[i]));
            }
        }

        if constexpr (PRINT) {
            std::cout << "post INTT Special GPU ";
            for (auto& i : SPECIALlimb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        s.wait(auxLimbs.s);

        {
            dim3 blockSize{64, 2};  // blockSize.x * blockSize.y * blockSize.z <= 1024, blockSize.x a multiple of 32

            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;

            ModDown2<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(auxLimbs.limbptr.data, limbsize,
                                                                         SPECIALlimbptr.data, PARTITION(id, 0));
        }

        if constexpr (PRINT) {
            std::cout << "Output ModDown ";
            for (auto& i : auxLimbs.limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            STREAM(limb.at(i)).wait(s);
        }
        if (limbsize > 0)
            NTT<algo, NTT_MODDOWN>(cc.batch, false, NTT_fusion_fields{.op2 = &auxLimbs});

        if constexpr (PRINT) {
            std::cout << "Output ModDown after sub mult.";
            for (auto& i : limb) {
                SWITCH(i, printThisLimb(2));
            }
        }

        for (int i = 0; i < limbsize; i += cc.batch) {
            s.wait(STREAM(limb.at(i)));
        }
    }
    auxLimbs.s.wait(s);

    if (free_special_limbs) {
        freeSpecialLimbs();
    }
}

#define YY(algo) \
    template void LimbPartition::moddown<algo>(LimbPartition & auxLimbs, bool ntt, bool free_special_limbs);
#include "ntt_types.inc"

#undef YY

//////// SEYDA /////////

void LimbPartition::automorph(const int index, const int br, LimbPartition* src, const bool ext) {
    cudaSetDevice(device);

    int limbsize = getLimbSize(*level);

    if (src)
        s.wait(src->s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        automorph_multi_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            src ? src->limbptr.data + i : limbptr.data + i, src ? limbptr.data + i : auxptr.data + i, index, br);
    }
    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);
            automorph_multi_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                src ? src->SPECIALlimbptr.data + i : SPECIALlimbptr.data + i,
                src ? SPECIALlimbptr.data + i : SPECIALauxptr.data + i, index, br);
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            s.wait(STREAM(SPECIALlimb[i]));
        }
        if (!src) {
            for (auto& i : SPECIALlimb) {
                if (i.index() == U32) {
                    std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
                } else {
                    std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
                }
            }
            std::swap(SPECIALlimbptr.data, SPECIALauxptr.data);
        }
    }
    if (!src) {
        for (auto& i : limb) {
            if (i.index() == U32) {
                std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
            } else {
                std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
            }
        }
        std::swap(limbptr.data, auxptr.data);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    if (src)
        src->s.wait(s);
}

void LimbPartition::automorph_multi(const int index, const int br) {
    const int limbsize = getLimbSize(*level);
    dim3 blockDim{128};
    dim3 gridDim{(cc.N) / blockDim.x, (uint32_t)limbsize};

    if (limbsize > 0)
        automorph_multi_<<<gridDim, blockDim, 0, s.ptr>>>(limbptr.data, auxptr.data, index, br);

    for (auto& i : limb) {
        if (i.index() == U32) {
            std::swap(std::get<U32>(i).v.data, std::get<U32>(i).aux.data);
        } else {
            std::swap(std::get<U64>(i).v.data, std::get<U64>(i).aux.data);
        }
    }
    std::swap(limbptr.data, auxptr.data);
}
void LimbPartition::modupInto(LimbPartition& partition, LimbPartition& aux_partition) {
    constexpr ALGO algo = ALGO_SHOUP;
    constexpr bool PRINT = false;
    //assert(SPECIALlimb.empty());
    cudaSetDevice(device);

    const int limbsize = *level + 1;

    s.wait(partition.s);
    s.wait(aux_partition.s);

    if constexpr (PRINT)
        for (auto& i : limb) {
            SWITCH(i, printThisLimb(2));
        }

    for (size_t d = 0; d < DECOMPmeta.size(); ++d) {

        int start = 0;
        for (int j = 0; j < d; ++j)
            start += DECOMPmeta[j].size();
        int size = std::min((int)DECOMPmeta[d].size(), limbsize - start);
        if (size <= 0)
            break;

        constexpr int M = 4;

        dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
        dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
        int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
        int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

        for (int i = 0; i < size; i += cc.batch) {
            STREAM(limb.at(start + i)).wait(s);
            uint32_t num_limbs = std::min((uint32_t)cc.batch, (uint32_t)(size - i));

            INTT_<false, algo, INTT_NONE><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst,
                                            bytesFirst, STREAM(limb.at(start + i)).ptr>>>(
                limbptr.data + start + i, PARTITION(id, start + i), auxptr.data + start + i);

            INTT_<true, algo, INTT_NONE><<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond,
                                           bytesSecond, STREAM(limb.at(start + i)).ptr>>>(
                auxptr.data + start + i, PARTITION(id, start + i), partition.DECOMPlimbptr[d].data + i);
        }
        for (size_t i = 0; i < size; i += cc.batch) {
            s.wait(STREAM(limb.at(start + i)));
        }
        if constexpr (PRINT)
            for (auto& i : partition.DECOMPlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        {
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<algo><<<gridSize, blockSize, shared_bytes, s.ptr>>>(
                partition.DECOMPlimbptr[d].data, *level + 1, partition.DIGITlimbptr[d].data, digitid[d]);
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }

        const int digitsize = hC_.num_primeid_digit_to[digitid.at(d)][*level];

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            STREAM(partition.DIGITlimb.at(d).at(i)).wait(s);
        }

        ApplyNTT<algo, NTT_NONE>(cc.batch, NTT_fusion_fields{}, partition.DIGITlimb.at(d), partition.DIGITlimbptr.at(d),
                                 aux_partition.DIGITlimbptr.at(d), cc, DIGIT(digitid.at(d), 0), digitsize);

        for (size_t i = 0; i < digitsize; i += cc.batch) {
            s.wait(STREAM(partition.DIGITlimb.at(d).at(i)));
        }

        if constexpr (PRINT)
            for (auto& i : partition.DIGITlimb[d]) {
                SWITCH(i, printThisLimb(2));
            }
    }

    aux_partition.s.wait(s);
    partition.s.wait(s);
}

void LimbPartition::multScalar(std::vector<uint64_t>& vector) {
    cudaSetDevice(device);
    /*
    cudaDeviceSynchronize();
    for (auto& l : limb) {
        if (l.index() == U64) {
            scalar_mult_<uint64_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr>>>(std::get<U64>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        } else {
            scalar_mult_<uint32_t, ALGO_BARRETT>
                <<<cc.N / 128, 128, 0, STREAM(l).ptr>>>(std::get<U32>(l).v.data, vector[PRIMEID(l)], PRIMEID(l));
        }
    }
    cudaDeviceSynchronize();
     */
    const int limbsize = getLimbSize(*level);

    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        Scalar_mult_<ALGO_BARRETT><<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, elems, PARTITION(id, i), nullptr);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}

void LimbPartition::addScalar(std::vector<uint64_t>& vector) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        int primeid_init = PARTITION(id, i);
        scalar_add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(limbptr.data + i, elems,
                                                                                            primeid_init);
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}

void LimbPartition::subScalar(std::vector<uint64_t>& vector) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    uint64_t* elems;
    cudaMallocAsync(&elems, vector.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, vector.data(), vector.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        scalar_sub_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(limbptr.data + i, elems,
                                                                                            PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    cudaFreeAsync(elems, s.ptr);
}

void LimbPartition::add(const LimbPartition& a, const LimbPartition& b, const bool ext_a, const bool ext_b) {
    cudaSetDevice(device);
    s.wait(a.s);
    s.wait(b.s);

    const int limbsize = getLimbSize(*level);

    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        if (!ext_a && ext_b) {
            addScaleB_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
                limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
        } else if (!ext_b && ext_a) {
            addScaleB_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
                limbptr.data + i, b.limbptr.data + i, a.limbptr.data + i, PARTITION(id, i));
        } else {
            add_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
                limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
        }
    }

    if (ext_a || ext_b) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            uint32_t size = std::min((int)start + num_limbs - (int)i, cc.batch);

            if (!ext_a && ext_b) {
                copy_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                    b.SPECIALlimbptr.data + i,
                    SPECIALlimbptr.data +
                        i);  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            } else if (!ext_b && ext_a) {
                copy_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                    a.SPECIALlimbptr.data + i,
                    SPECIALlimbptr.data +
                        i);  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            } else {
                add_<<<dim3{(uint32_t)cc.N / 128, size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                    SPECIALlimbptr.data + i, a.SPECIALlimbptr.data + i, b.SPECIALlimbptr.data + i,
                    SPECIAL(
                        id,
                        i));  // TODO: have to check if Limbpartition comes from a plaintext, where extension limbs are mapped differently
            }
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            s.wait(STREAM(SPECIALlimb[i]));
        }
    }

    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }

    a.s.wait(s);
    b.s.wait(s);
}
void LimbPartition::squareElement(const LimbPartition& p) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    s.wait(p.s);
    int size = std::min(limbsize, (int)p.limb.size());
    for (int i = 0; i < size; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)size - i, cc.batch);
        square_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, p.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < size; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    p.s.wait(s);
}
void LimbPartition::binomialSquareFold(LimbPartition& c0_res, const LimbPartition& c2_key_switched_0,
                                       const LimbPartition& c2_key_switched_1) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    s.wait(c0_res.s);
    s.wait(c2_key_switched_0.s);
    s.wait(c2_key_switched_1.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        binomial_square_fold_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            c0_res.limbptr.data + i, c2_key_switched_0.limbptr.data + i, limbptr.data + i,
            c2_key_switched_1.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c0_res.s.wait(s);
    c2_key_switched_0.s.wait(s);
    c2_key_switched_1.s.wait(s);
}
void LimbPartition::dropLimb() {
    cudaSetDevice(device);

    STREAM(limb.back()).wait(s);
    limb.pop_back();
}
void LimbPartition::addMult(const LimbPartition& a, const LimbPartition& b) {
    const int limbsize = getLimbSize(*level);
    assert(a.limb.size() >= limbsize);
    assert(b.limb.size() >= limbsize);
    cudaSetDevice(device);
    s.wait(a.s);
    s.wait(b.s);
    for (int i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        uint32_t num_limbs = std::min((int)limbsize - i, cc.batch);
        addMult_<<<dim3{(uint32_t)cc.N / 128, num_limbs}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data + i, a.limbptr.data + i, b.limbptr.data + i, PARTITION(id, i));
    }
    for (int i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    a.s.wait(s);
    b.s.wait(s);
}
void LimbPartition::broadcastLimb0() {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    assert(limbsize - 1 > 0);
    broadcastLimb0_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize - 1}, 128, 0, s.ptr>>>(limbptr.data);
}
void LimbPartition::evalLinearWSum(uint32_t n, std::vector<const LimbPartition*> ps, std::vector<uint64_t>& weights) {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    for (int i = 0; i < n; ++i) {
        s.wait(ps[i]->s);
    }

    uint64_t* elems;
    cudaMallocAsync(&elems, weights.size() * sizeof(uint64_t), s.ptr);
    cudaMemcpyAsync(elems, weights.data(), weights.size() * sizeof(uint64_t), cudaMemcpyDefault, s.ptr);
    std::vector<void**> psptr(n, nullptr);
    for (int i = 0; i < n; ++i) {
        psptr[i] = ps[i]->limbptr.data;
        assert(ps[i]->limb.size() >= limbsize);
    }
    void*** d_psptr;
    cudaMallocAsync(&d_psptr, psptr.size() * sizeof(void**), s.ptr);
    cudaMemcpyAsync(d_psptr, psptr.data(), psptr.size() * sizeof(void**), cudaMemcpyDefault, s.ptr);

    if (!limb.empty())
        eval_linear_w_sum_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize}, 128, 0, s.ptr>>>(n, limbptr.data, d_psptr,
                                                                                              elems, PARTITION(id, 0));
    cudaFreeAsync(elems, s.ptr);
    cudaFreeAsync(d_psptr, s.ptr);
    for (uint32_t i = 0; i < n; ++i) {
        ps[i]->s.wait(s);
    }
}

/**
  Only for MGPU key generation
 */
void LimbPartition::generatePartialSpecialLimb() {
    cudaSetDevice(device);
    if (bufferSPECIAL == nullptr) {
        cudaMallocAsync(&bufferSPECIAL, cc.N * cc.splitSpecialMeta.at(id).size() * sizeof(uint64_t), s.ptr);
        generate(cc.splitSpecialMeta.at(id), SPECIALlimb, SPECIALlimbptr, (int)cc.splitSpecialMeta.at(id).size() - 1,
                 nullptr, bufferSPECIAL, 0);

        // for (auto& l : SPECIALlimb)
        //     STREAM(l).wait(s);
    }
}
void LimbPartition::dotProductPt(LimbPartition& c1, const std::vector<const LimbPartition*>& c0s,
                                 const std::vector<const LimbPartition*>& c1s,
                                 const std::vector<const LimbPartition*>& pts, const bool ext) {

    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    int n = c0s.size();
    std::vector<void**> h_data(n * 3 * (1 + ext), nullptr);

    for (size_t i = 0; i < n; ++i) {
        h_data[i] = c0s[i]->limbptr.data;
        h_data[i + n] = c1s[i]->limbptr.data;
        h_data[i + 2 * n] = pts[i]->limbptr.data;
        s.wait(c0s[i]->s);
        s.wait(c1s[i]->s);
        s.wait(pts[i]->s);
        assert(c0s[i]->limb.size() >= limbsize);
        assert(c1s[i]->limb.size() >= limbsize);
        assert(pts[i]->limb.size() >= limbsize);
        if (ext) {
            int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
            int num_limbs = cc.splitSpecialMeta.at(id).size();
            assert(c0s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(c1s[i]->SPECIALlimb.size() >= this->SPECIALlimb.size());
            assert(pts[i]->SPECIALlimb.size() >= 0);
            h_data[i + 3 * n] = c0s[i]->SPECIALlimbptr.data + start;
            h_data[i + 4 * n] = c1s[i]->SPECIALlimbptr.data + start;
            h_data[i + 5 * n] = pts[i]->SPECIALlimbptr.data;
        }
    }

    s.wait(c1.s);

    VectorGPU<void**> data(s, n * 3 * (1 + ext), device, h_data.data());

    for (size_t i = 0; i < limbsize; i += cc.batch) {
        STREAM(limb[i]).wait(s);
        int size = std::min((int)limbsize - (int)i, cc.batch);
        dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(limb[i]).ptr>>>(
            limbptr.data, c1.limbptr.data, data.data, i, PARTITION(id, i), n);
    }

    if (ext) {
        int start = cc.splitSpecialMeta.at(id).at(0).id - hC_.L;
        int num_limbs = cc.splitSpecialMeta.at(id).size();
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
            int size = std::min((int)start + num_limbs - (int)i, cc.batch);
            dotProductPt_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)size}, 128, 0, STREAM(SPECIALlimb[i]).ptr>>>(
                SPECIALlimbptr.data + start, c1.SPECIALlimbptr.data + start, data.data + 3 * n, i - start,
                SPECIAL(id, i), n);
        }
        for (size_t i = start; i < start + num_limbs; i += cc.batch) {
            STREAM(SPECIALlimb[i]).wait(s);
        }
    }
    for (size_t i = 0; i < limbsize; i += cc.batch) {
        s.wait(STREAM(limb[i]));
    }
    c1.s.wait(s);
    for (size_t i = 0; i < c0s.size(); ++i) {
        c0s[i]->s.wait(s);
        c1s[i]->s.wait(s);
        pts[i]->s.wait(s);
    }
    data.free(s);
}
void LimbPartition::generateGatherLimb(bool iskey) {
    if (bufferGATHER == nullptr) {
        if (cc.GPUid.size() == 1 || iskey) {
            cudaMallocAsync(&bufferGATHER, GATHERmeta.size() * sizeof(uint64_t) * cc.N, s.ptr);
        } else {
#ifdef NCCL
            cudaStreamSynchronize(s.ptr);
            NCCLCHECK(ncclMemAlloc((void**)&bufferGATHER, GATHERmeta.size() * sizeof(uint64_t) * cc.N));
            NCCLCHECK(ncclCommRegister(rank, bufferGATHER, GATHERmeta.size() * sizeof(uint64_t) * cc.N,
                                       &bufferGATHER_handle));
            cudaDeviceSynchronize();
#else
            assert(false);
#endif
        }
        std::vector<void*> h_gatherptr(GATHERptr.size, nullptr);
        for (size_t i = 0; i < h_gatherptr.size(); ++i)
            h_gatherptr[i] = (void*)(bufferGATHER + cc.N * i);
        cudaMemcpyAsync(GATHERptr.data, h_gatherptr.data(), GATHERptr.size * sizeof(void*), cudaMemcpyHostToDevice,
                        s.ptr);
    }
}

// namespace FIDESlib::CKKS

}  // namespace FIDESlib::CKKS
