//
// Created by carlosad on 8/06/25.
//
#include "CKKS/Context.cuh"
#include "CKKS/Conv.cuh"
#include "CKKS/ElemenwiseBatchKernels.cuh"
#include "CKKS/LimbPartition.cuh"

namespace FIDESlib::CKKS {

void LimbPartition::rescaleMGPU() {
    const int limbsize = getLimbSize(*level);
    cudaSetDevice(device);
    static bool parity = true;

    Stream& stream = parity ? cc.top_limb_stream[id] : cc.top_limb_stream2[id];
    uint64_t* buffer = parity ? cc.top_limb_buffer[id] : cc.top_limb_buffer2[id];
    VectorGPU<void*>& ptr = parity ? cc.top_limbptr[id] : cc.top_limbptr2[id];
    //parity = !parity;

    if (cc.limbGPUid[*level].x == id) {
        if (limb.size() > cc.limbGPUid[*level].y && PRIMEID(limb[cc.limbGPUid[*level].y]) == *level) {

            LimbImpl& top = limb.at(limbsize - 1);
            STREAM(top).wait(s);
            SWITCH(top, INTT<ALGO_SHOUP>());
            STREAM(top).wait(stream);
            cudaMemcpyAsync(buffer, std::get<U64>(top).v.data, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                            STREAM(top).ptr);
            stream.wait(STREAM(top));
            /*
            std::cout << "GPU: " << id << " ";
            SWITCH(top, printThisLimb(2));
            std::cout << std::endl;
*/
            while (bufferLIMB == nullptr && limb.size() > limbsize - 1) {
                STREAM(limb.back()).wait(stream);
                limb.pop_back();
            }
        } else {
            std::cout << "Cant find the top limb!!!" << id << " " << *level << " " << cc.limbGPUid[*level].y << " "
                      << limb.size() << std::endl;
        }
    } else {
        stream.wait(s);
    }

#ifdef NCCL
    if constexpr (0) {
        NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[*level].x, rank, stream.ptr));
    } else {
        NCCLCHECK(ncclGroupStart());
        if (id == cc.limbGPUid[*level].x) {
            for (int i = 0; i < cc.GPUid.size(); ++i) {
                if (i != id)
                    ncclSend(buffer, cc.N, ncclUint64, i, rank, stream.ptr);
            }
        } else {
            ncclRecv(buffer, cc.N, ncclUint64, cc.limbGPUid[*level].x, rank, stream.ptr);
        }
        NCCLCHECK(ncclGroupEnd());
    }
#else
    assert(false);
#endif
    /* {
        std::vector<uint64_t> data(cc.N, 0);
        cudaMemcpyAsync(data.data(), buffer, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.ptr);
        cudaDeviceSynchronize();
        std::cout << "GPU " << id << " (" << data[0] << " " << data[1] << ")" << std::endl;
    } */

    {
        {
            constexpr ALGO algo = ALGO_SHOUP;
            constexpr int M = 4;

            const dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            const dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            const int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
            const int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
            const int size = getLimbSize(*level - 1);
            const int batch = cc.batch;
            const NTT_MODE mode = NTT_RESCALE;

            for (int i = 0; i < size; i += batch) {
                STREAM(limb.at(i)).wait(stream);
            }
            for (int i = 0; i < size; i += batch) {
                uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

                NTT_<false, algo, mode>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                       STREAM(limb.at(i)).ptr>>>(ptr.data, PARTITION(id, i), auxptr.data + i, nullptr, *level);

                stream.wait(STREAM(limb[i]));  // Data dependency on top_limb reaches only to here

                NTT_<true, algo, mode>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                       STREAM(limb.at(i)).ptr>>>(auxptr.data + i, PARTITION(id, i), limbptr.data + i, nullptr, *level);
            }
            for (int i = 0; i < size; i += batch) {
                s.wait(STREAM(limb[i]));
            }
        }
    }
}

void LimbPartition::doubleRescaleMGPU(LimbPartition& partition) {
    const int limbsize = getLimbSize(*level);
    assert(limbsize == getLimbSize(*partition.level));
    cudaSetDevice(device);

    Stream* stream[2] = {&cc.top_limb_stream[id], &cc.top_limb_stream2[id]};

    uint64_t* buffer[2] = {cc.top_limb_buffer[id], cc.top_limb_buffer2[id]};

    VectorGPU<void*>* ptr[2] = {&cc.top_limbptr[id], &cc.top_limbptr2[id]};

    LimbPartition* part[2] = {this, &partition};
    //parity = !parity;

    for (int i = 0; i < 2; ++i) {
        if (cc.limbGPUid[*part[i]->level].x == id) {
            if (part[i]->limb.size() > cc.limbGPUid[*part[i]->level].y &&
                PRIMEID(part[i]->limb[cc.limbGPUid[*part[i]->level].y]) == *part[i]->level) {

                LimbImpl& top = part[i]->limb.at(limbsize - 1);
                STREAM(top).wait(part[i]->s);
                SWITCH(top, INTT<ALGO_SHOUP>());
                STREAM(top).wait(*stream[i]);
                cudaMemcpyAsync(buffer[i], std::get<U64>(top).v.data, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                                STREAM(top).ptr);
                /*
            std::cout << "GPU: " << id << " ";
            SWITCH(top, printThisLimb(2));
            std::cout << std::endl;
*/
                stream[i]->wait(STREAM(top));
                while (part[i]->bufferLIMB == nullptr && part[i]->limb.size() > limbsize - 1) {
                    STREAM(part[i]->limb.back()).wait(*stream[i]);
                    part[i]->limb.pop_back();
                }
            } else {
                std::cout << "Cant find the top limb!!!" << part[i]->id << " " << *part[i]->level << " "
                          << cc.limbGPUid[*part[i]->level].y << " " << part[i]->limb.size() << std::endl;
            }
        } else {
            stream[i]->wait(part[i]->s);
        }
    }

#ifdef NCCL
    {
        NCCLCHECK(ncclGroupStart());
        if (id == cc.limbGPUid[*level].x) {
            for (int i = 0; i < cc.GPUid.size(); ++i) {
                if (i != id) {
                    ncclSend(buffer[0], cc.N, ncclUint64, i, rank, stream[0]->ptr);
                    ncclSend(buffer[1], cc.N, ncclUint64, i, rank, stream[1]->ptr);
                }
            }
        } else {
            ncclRecv(buffer[0], cc.N, ncclUint64, cc.limbGPUid[*part[0]->level].x, rank, stream[0]->ptr);
            ncclRecv(buffer[1], cc.N, ncclUint64, cc.limbGPUid[*part[1]->level].x, rank, stream[1]->ptr);
        }
        NCCLCHECK(ncclGroupEnd());
    }
#else
    assert(false);
#endif
    /* {
        std::vector<uint64_t> data(cc.N, 0);
        cudaMemcpyAsync(data.data(), buffer, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream.ptr);
        cudaDeviceSynchronize();
        std::cout << "GPU " << id << " (" << data[0] << " " << data[1] << ")" << std::endl;
    } */

    for (int j = 0; j < 2; ++j) {
        {
            constexpr ALGO algo = ALGO_SHOUP;
            constexpr int M = 4;

            const dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            const dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            const int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
            const int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == ALGO_SHOUP ? 1 : 0));
            const int size = getLimbSize(*part[j]->level - 1);
            const int batch = cc.batch;
            const NTT_MODE mode = NTT_RESCALE;

            for (int i = 0; i < size; i += batch) {
                STREAM(part[j]->limb.at(i)).wait(*stream[j]);
            }
            for (int i = 0; i < size; i += batch) {
                uint32_t num_limbs = std::min((uint32_t)batch, (uint32_t)(size - i));

                NTT_<false, algo, mode><<<dim3{cc.N / (blockDimFirst.x * M * 2), num_limbs}, blockDimFirst, bytesFirst,
                                          STREAM(part[j]->limb.at(i)).ptr>>>(
                    ptr[j]->data, PARTITION(part[j]->id, i), part[j]->auxptr.data + i, nullptr, *part[j]->level);

                stream[j]->wait(STREAM(part[j]->limb[i]));  // Data dependency on top_limb reaches only to here

                NTT_<true, algo, mode>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), num_limbs}, blockDimSecond, bytesSecond,
                       STREAM(part[j]->limb.at(i)).ptr>>>(part[j]->auxptr.data + i, PARTITION(part[j]->id, i),
                                                          part[j]->limbptr.data + i, nullptr, *part[j]->level);
            }
            for (int i = 0; i < size; i += batch) {
                part[j]->s.wait(STREAM(part[j]->limb[i]));
            }
        }
    }
}

void LimbPartition::dotKSKfusedMGPU(LimbPartition& out2, const LimbPartition& digitSrc, const LimbPartition& ksk_a,
                                    const LimbPartition& ksk_b, const LimbPartition& src) {
    cudaSetDevice(device);

    VectorGPU<void**> digits(s, cc.dnum * 6, device);
    std::vector<void**> h_digits(cc.dnum * 6, nullptr);
    LimbPartition& out1 = *this;
    cudaSetDevice(device);
    s.wait(ksk_a.s);
    s.wait(ksk_b.s);
    s.wait(src.s);
    s.wait(out2.s);
    const int limbsize = *level + 1;

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
            h_digits[i + 3 * cc.dnum] = src.limbptr.data;
            h_digits[i + 4 * cc.dnum] = ksk_a.limbptr.data;
            h_digits[i + 5 * cc.dnum] = ksk_b.limbptr.data;
        }

        int num_special = 0;
        while (num_special < (int)DIGITmeta.at(0).size() && DIGITmeta.at(0).at(num_special).id > cc.L)
            num_special++;
        int num_limbs = 0;
        while (num_limbs < (int)meta.size() && meta.at(num_limbs).id <= *level)
            num_limbs++;

        cudaMemcpyAsync(digits.data, h_digits.data(), cc.dnum * 6 * sizeof(void**), cudaMemcpyDefault, s.ptr);

        fusedDotKSK_2_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)num_special + num_limbs}, 128, 0, s.ptr>>>(
            out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data, digits.data, i,
            id, num_special, 0);
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
}

void LimbPartition::modup_ksk_moddown_mgpu(LimbPartition& c0, const LimbPartition& ksk_a, const LimbPartition& ksk_b,
                                           LimbPartition& auxLimbs1, LimbPartition& auxLimbs2, const bool moddown) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    bool SELECT = id == 1;
    LimbPartition& c1 = *this;
    int num_d = 0;
    {
        int start = 0;
        if constexpr (PRINT)
            std::cout << "/** Compute how many digits are used at this level*/" << std::endl;
        while (num_d < cc.dnum && start < *level + 1) {
            start += DECOMPmeta.at(num_d).size();
            num_d++;
        }
    }
    uint32_t limb_size = 0;
    while (limb_size < meta.size() && meta[limb_size].id <= *level)
        limb_size++;

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Input: ";
            for (size_t i = 0; i < limb_size; ++i) {
                std::cout << meta[i].id;
                SWITCH(limb[i], printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
        CudaCheckErrorMod;
    }

    if constexpr (PRINT)
        std::cout
            << "/** We try to pipeline the computation of each digit first, splitting independent groups of limbs*/"
            << std::endl;

    const int digits_per_it = num_d;  //cc.logN <= 15 ? num_d : cc.logN == 16 ? std::max((num_d + 1) / 2, 1) : 1;

    if constexpr (PRINT)
        std::cout << "GPU " << id << "compute " << num_d << " digits" << std::endl;

    if (1) {
        for (int d = 0; d < num_d; d += digits_per_it) {
            int ds = std::min(num_d - d, digits_per_it);

            uint32_t start_d = 0;
            while (start_d < limb_size && meta[start_d].digit < d)
                start_d++;
            uint32_t size_d = 0;
            while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
                size_d++;

            if constexpr (PRINT)
                if (SELECT) {
                    std::cout << "GPU " << id << " for digits " << d << ":" << d + digits_per_it << " INTT " << size_d
                              << " limbs starting at limb " << start_d << std::endl;
                }

            Stream& stream = cc.digitStream.at(d).at(id);
            stream.wait(s);
            if constexpr (PRINT)
                std::cout << "/** Intt */" << std::endl;
            if (size_d > 0) {
                constexpr ALGO algo = ALGO_SHOUP;
                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                int gather_offset = 0;
                for (int i = 0; i < id; ++i) {
                    gather_offset += cc.meta.at(i).size();
                }

                {
                    INTT_<false, algo, INTT_NONE>
                        <<<dim3{cc.N / (blockDimFirst.x * M * 2), size_d}, blockDimFirst, bytesFirst, stream.ptr>>>(
                            limbptr.data + start_d, PARTITION(id, start_d), auxptr.data + start_d);

                    INTT_<true, algo, INTT_NONE>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), size_d}, blockDimSecond, bytesSecond, stream.ptr>>>(
                            auxptr.data + start_d, PARTITION(id, start_d), GATHERptr.data + gather_offset + start_d);
                }
            }

            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU: " << id << "Out INTT: ";
                    for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                        for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
                            std::cout << DECOMPmeta[j][i].id;
                            SWITCH(DECOMPlimb[j][i], printThisLimb(2));
                        }
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
                CudaCheckErrorMod;
            }

            if constexpr (PRINT)
                std::cout << "/** Communicate */" << std::endl;
            {
                size_t j = id;
#ifdef NCCL
                NCCLCHECK(ncclGroupStart());

                int start = 0;
                for (size_t i = 0; i < cc.GPUid.size(); ++i) {

                    uint32_t limb_size_i = 0;
                    while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
                        limb_size_i++;
                    uint32_t start_d_i = 0;
                    while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
                        start_d_i++;
                    uint32_t size_d_i = 0;
                    while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
                        size_d_i++;

                    if constexpr (PRINT)
                        if (SELECT) {
                            std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it
                                      << " communicate " << size_d_i << " limbs" << std::endl;
                        }

                    if (size_d_i > 0) {
                        NCCLCHECK(
                            ncclBroadcast(/*bufferLIMB + cc.N * start_d*/ bufferGATHER + cc.N * (start + start_d_i),
                                          bufferGATHER + cc.N * (start + start_d_i), size_d_i * cc.N, ncclUint64, i,
                                          rank, stream.ptr));
                    }
                    start += cc.meta[i].size();
                }

                NCCLCHECK(ncclGroupEnd());
#else
                assert(false);
#endif
            }
        }
    }

    if (1) {
        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out INTT after communicate: ";
                for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                    for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
                        std::cout << DECOMPmeta[j][i].id;
                        SWITCH(DECOMPlimb[j][i], printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                cudaDeviceSynchronize();
            }
            CudaCheckErrorMod;
        }

        for (int d = 0; d < num_d; d += digits_per_it) {
            int ds = std::min(num_d - d, digits_per_it);

            uint32_t start_d = 0;
            while (start_d < limb_size && meta[start_d].digit < d)
                start_d++;
            uint32_t size_d = 0;
            while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
                size_d++;

            Stream& stream = cc.digitStream.at(d).at(id);

            if constexpr (PRINT)
                std::cout << "/** Conv */" << std::endl;

            for (int d_ = d; d_ < d + ds; ++d_) {
                Stream& stream1 = cc.digitStream.at(d_).at(id);
                stream1.wait(stream);
            }
            for (int d_ = d; d_ < d + ds; ++d_) {
                Stream& stream1 = cc.digitStream.at(d_).at(id);

                int start = 0;
                for (int j = 0; j < d_; ++j)
                    start += DECOMPlimb.at(j).size();
                int size = std::min((int)DECOMPlimb.at(d_).size(), *level + 1 - start);

                if (size <= 0) {
                    std::cerr << "void modup, aborting" << std::endl;
                    exit(-1);
                }

                if constexpr (PRINT)
                    if (SELECT) {
                        std::cout << host_constants_per_gpu[id].num_primeid_digit_to[d_][*level]
                                  << "<- num_prime_id_digit_to: " << d_ << std::endl;
                        std::cout << host_constants_per_gpu[id].num_primeid_digit_from[d_][*level]
                                  << "<- num_prime_id_digit_from: " << d_ << std::endl;
                        /*
                    std::cout << host_constants_per_gpu[id].num_primeid_digit_to[d_][level - 1]
                              << "<- num_prime_id_digit_to: " << d_ << std::endl;
                    std::cout << host_constants_per_gpu[id].num_primeid_digit_from[d_][level - 1]
                              << "<- num_prime_id_digit_from: " << d_ << std::endl;
                              */
                    }

                dim3 blockSize{64, 2};
                dim3 gridSize{(uint32_t)cc.N / blockSize.x};
                int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
                DecompAndModUpConv<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream1.ptr>>>(
                    DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_]);

                cc.digitStream2.at(d_).at(id).wait(stream1); /** Get dependency for limb NTTs later */
                if constexpr (PRINT)
                    std::cout << "/** NTT special limbs */" << std::endl;
                {
                    uint32_t size = cc.splitSpecialMeta.at(id).size();
                    constexpr ALGO algo = ALGO_SHOUP;
                    constexpr int M = 4;

                    dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                    dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                    int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                    int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                    if (size > 0) {
                        NTT_<false, algo, NTT_NONE>
                            <<<dim3{cc.N / (blockDimFirst.x * M * 2), size}, blockDimFirst, bytesFirst, stream1.ptr>>>(
                                DIGITlimbptr[d_].data, DIGIT(d_, 0), c0.DIGITlimbptr[d_].data);

                        NTT_<true, algo, NTT_NONE>
                            <<<dim3{cc.N / (blockDimSecond.x * M * 2), size}, blockDimSecond, bytesSecond,
                               stream1.ptr>>>(c0.DIGITlimbptr[d_].data, DIGIT(d_, 0), DIGITlimbptr[d_].data);
                    }
                }
            }
        }
        for (int d = 0; d < num_d; ++d) {
            s.wait(cc.digitStream.at(d).at(id));
        }

        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out ModUp after NTT specials: ";
                for (size_t j = 0; j < DIGITlimb.size(); ++j) {
                    for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
                        std::cout << DIGITmeta[j][i].id;
                        SWITCH(DIGITlimb[j][i], printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
            CudaCheckErrorMod;
        }
    }

    if constexpr (PRINT)
        std::cout << "/** We ksk only special limbs and start ModDown as soon as possible */" << std::endl;
    if (1) {
        if constexpr (PRINT)
            std::cout << "/** ksk */" << std::endl;
        {
            std::vector<void**> h_digits(cc.dnum * 6, nullptr);
            LimbPartition& out1 = *this;
            LimbPartition& out2 = c0;

            if constexpr (1) {
                for (size_t j = 0; j < num_d; ++j) {
                    h_digits[j] = DIGITlimbptr.at(j).data;
                    h_digits[j + cc.dnum] = ksk_a.DIGITlimbptr.at(j).data;
                    h_digits[j + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(j).data;
                    h_digits[j + 3 * cc.dnum] = limbptr.data;
                    h_digits[j + 4 * cc.dnum] = ksk_a.limbptr.data;
                    h_digits[j + 5 * cc.dnum] = ksk_b.limbptr.data;
                }

                int num_special = cc.splitSpecialMeta.at(id).size();

                VectorGPU<void**> digits(s, cc.dnum * 6, device, h_digits.data());

                s.wait(ksk_a.s);
                s.wait(ksk_b.s);
                s.wait(c0.s);

                if (num_special > 0) {
                    fusedDotKSK_2_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)num_special}, 128, 0, s.ptr>>>(
                        out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data,
                        digits.data, num_d, id, num_special, 0);
                }
                digits.free(s);
            }

            if constexpr (PRINT) {
                if (SELECT) {
                    cudaDeviceSynchronize();
                    std::cout << "GPU: " << id << "Out KSK specials: ";
                    for (const auto& j : {&out1, &out2}) {
                        for (auto& i : j->SPECIALlimb) {
                            SWITCH(i, printThisLimb(2));
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    cudaDeviceSynchronize();
                }
                CudaCheckErrorMod;
            }
        }

        c0.s.wait(s);

        if (moddown) {
            for (int i = 0; i < 2; ++i) {
                Stream& stream = i == 0 ? s : c0.s;
                LimbPartition& out = i == 0 ? c1 : c0;
                if constexpr (PRINT)
                    std::cout << "/** INTT specials*/" << std::endl;
                {
                    constexpr ALGO algo = ALGO_SHOUP;
                    constexpr int M = 4;

                    dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                    dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                    int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                    int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                    const int j = cc.splitSpecialMeta.at(id).at(0).id - cc.specialMeta.at(id).at(0).id;
                    const uint32_t limbs = cc.splitSpecialMeta.at(id).size();

                    if (limbs > 0) {
                        INTT_<false, algo, INTT_NONE>
                            <<<dim3{cc.N / (blockDimFirst.x * M * 2), limbs}, blockDimFirst, bytesFirst, stream.ptr>>>(
                                out.SPECIALlimbptr.data + j, SPECIAL(id, j), out.SPECIALauxptr.data + j);

                        INTT_<true, algo, INTT_NONE>
                            <<<dim3{cc.N / (blockDimSecond.x * M * 2), limbs}, blockDimSecond, bytesSecond,
                               stream.ptr>>>(out.SPECIALauxptr.data + j, SPECIAL(id, j), out.SPECIALlimbptr.data + j);
                    }
                }
            }

            if constexpr (PRINT)
                std::cout << "/** communicate */" << std::endl;
#ifdef NCCL
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < 2; ++i) {
                Stream& stream = i == 0 ? s : c0.s;
                LimbPartition& out = i == 0 ? c1 : c0;

                for (size_t j = 0; j < cc.splitSpecialMeta.size(); ++j) {
                    //Limb<uint64_t>& l = std::get<U64>(
                    //    this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
                    const uint32_t num_limbs = cc.splitSpecialMeta.at(j).size();
                    //uint64_t* ptr2 = l.v.data;
                    uint64_t* ptr =
                        out.bufferSPECIAL + (cc.splitSpecialMeta.at(j).at(0).id - SPECIALmeta.at(0).id) * cc.N;

                    if (num_limbs > 0)
                        NCCLCHECK(ncclBroadcast(ptr, ptr, cc.N * num_limbs, ncclUint64, (int)j, rank, stream.ptr));
                }
            }
            NCCLCHECK(ncclGroupEnd());
#else
            assert(false);
#endif
        }
    }

    if (moddown) {
        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "KSK specials after INTT and communicate: ";
                for (const auto& j : {&c1, &c0}) {
                    for (auto& i : j->SPECIALlimb) {
                        SWITCH(i, printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
            CudaCheckErrorMod;
        }

        for (int i = 0; i < 2; ++i) {
            Stream& stream = i == 0 ? c1.s : c0.s;
            LimbPartition& out = i == 0 ? c1 : c0;
            LimbPartition& auxLimbs = i == 0 ? auxLimbs1 : auxLimbs2;
            if constexpr (PRINT)
                std::cout << "/** Conv */" << std::endl;
            stream.wait(auxLimbs.s);
            {
                dim3 blockSize{64, 2};

                dim3 gridSize{(uint32_t)cc.N / blockSize.x};
                int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;
                if (limb_size > 0)
                    ModDown2<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr>>>(
                        auxLimbs.limbptr.data, limb_size, out.SPECIALlimbptr.data, PARTITION(id, 0));
            }
        }

        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out Moddown: ";
                for (const auto& j : {&auxLimbs1, &auxLimbs2}) {
                    for (auto& i : j->limb) {
                        SWITCH(i, printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            Stream& stream = i == 0 ? c1.s : c0.s;
            LimbPartition& auxLimbs = i == 0 ? auxLimbs1 : auxLimbs2;
            stream.wait(auxLimbs.s);
        }
    }

    if constexpr (PRINT)
        std::cout << "/**We delay the call of NTTs post-modup for non special limbs to here*/" << std::endl;
    for (int d = 0; d < num_d; ++d) {

        Stream& stream = cc.digitStream2.at(d).at(id);

        if (limb_size > 0) {
            uint32_t start = cc.splitSpecialMeta.at(id).size();
            uint32_t size = host_constants_per_gpu[id].num_primeid_digit_to[d][*level] - start;
            if (size > 0) {
                constexpr ALGO algo = ALGO_SHOUP;
                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                {
                    NTT_<false, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimFirst.x * M * 2), size}, blockDimFirst, bytesFirst, stream.ptr>>>(
                            DIGITlimbptr[d].data + start, DIGIT(d, start), c0.DIGITlimbptr[d].data + start);

                    NTT_<true, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), size}, blockDimSecond, bytesSecond, stream.ptr>>>(
                            c0.DIGITlimbptr[d].data + start, DIGIT(d, start), DIGITlimbptr[d].data + start);
                }
            }
        }
    }

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Out ModUp after NTT all limbs: ";
            for (size_t j = 0; j < DIGITlimb.size(); ++j) {
                for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
                    std::cout << DIGITmeta[j][i].id;
                    SWITCH(DIGITlimb[j][i], printThisLimb(2));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
        CudaCheckErrorMod;
    }

    if constexpr (PRINT)
        std::cout << "/** ksk remaining limbs*/" << std::endl;

    {
        Stream& stream = cc.digitStream2.at(0).at(id);
        for (int d = 1; d < num_d; ++d) {
            stream.wait(cc.digitStream2.at(d).at(id));
        }
        std::vector<void**> h_digits(cc.dnum * 6, nullptr);
        LimbPartition& out1 = *this;
        LimbPartition& out2 = c0;
        cudaSetDevice(device);
        stream.wait(ksk_a.s);
        stream.wait(ksk_b.s);
        stream.wait(c0.s);

        if constexpr (1) {
            size_t i = 0;
            for (; i < num_d; ++i) {
                h_digits[i] = DIGITlimbptr.at(i).data;
                h_digits[i + cc.dnum] = ksk_a.DIGITlimbptr.at(i).data;
                h_digits[i + 2 * cc.dnum] = ksk_b.DIGITlimbptr.at(i).data;
                h_digits[i + 3 * cc.dnum] = limbptr.data;
                h_digits[i + 4 * cc.dnum] = ksk_a.limbptr.data;
                h_digits[i + 5 * cc.dnum] = ksk_b.limbptr.data;
            }

            int num_special = cc.splitSpecialMeta.at(id).size();

            VectorGPU<void**> digits(stream, cc.dnum * 6, device);

            cudaMemcpyAsync(digits.data, h_digits.data(), cc.dnum * 6 * sizeof(void**), cudaMemcpyDefault, stream.ptr);

            if (limb_size > 0) {
                for (int start = 0; start < limb_size; start += cc.batch) {
                    int num = std::min(cc.batch, (int)limb_size - start);
                    STREAM(limb[start]).wait(stream);
                    fusedDotKSK_2_<<<dim3{(uint32_t)cc.N / 128, (uint32_t)num}, 128, 0, STREAM(limb[start]).ptr>>>(
                        out1.limbptr.data, out1.SPECIALlimbptr.data, out2.limbptr.data, out2.SPECIALlimbptr.data,
                        digits.data, i, id, num_special, num_special + start);
                }
                for (int start = 0; start < limb_size; start += cc.batch) {
                    stream.wait(STREAM(limb[start]));
                }
            }
            digits.free(stream);
        }

        ksk_a.s.wait(stream);
        ksk_b.s.wait(stream);

        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out KSK limbs: ";
                for (const auto& j : {&out1, &out2}) {
                    for (auto& i : j->limb) {
                        SWITCH(i, printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
        }
    }

    if (moddown) {
        for (int i = 0; i < 2; ++i) {
            if constexpr (PRINT)
                std::cout << "/** Last NTT step for moddown*/" << std::endl;
            Stream& stream = i == 0 ? c1.s : c0.s;
            LimbPartition& out = i == 0 ? c1 : c0;
            LimbPartition& auxLimbs = i == 0 ? auxLimbs1 : auxLimbs2;

            if (limb_size > 0) {
                constexpr ALGO algo = ALGO_SHOUP;
                constexpr int M = 4;

                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

                {
                    NTT_<false, algo, NTT_MODDOWN>
                        <<<dim3{cc.N / (blockDimFirst.x * M * 2), limb_size}, blockDimFirst, bytesFirst, stream.ptr>>>(
                            auxLimbs.limbptr.data, PARTITION(id, 0), out.auxptr.data);

                    stream.wait(cc.digitStream2.at(0).at(id));

                    NTT_<true, algo, NTT_MODDOWN>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), limb_size}, blockDimSecond, bytesSecond,
                           stream.ptr>>>(out.auxptr.data, PARTITION(id, 0), out.limbptr.data);
                }
            }
            auxLimbs.s.wait(stream);
        }

        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out Moddown after submult: ";
                for (const auto& j : {&c1, &c0}) {
                    for (auto& i : j->limb) {
                        SWITCH(i, printThisLimb(2));
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
            CudaCheckErrorMod;
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            Stream& stream = i == 0 ? c1.s : c0.s;
            LimbPartition& auxLimbs = i == 0 ? auxLimbs1 : auxLimbs2;

            stream.wait(cc.digitStream2.at(0).at(id));
            auxLimbs.s.wait(stream);
        }
    }
    if constexpr (PRINT) {
        std::cout << "Going out keyswitch" << std::endl;
    }
}

void LimbPartition::modupMGPU(LimbPartition& aux) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    bool SELECT = id == 1;
    LimbPartition& c1 = *this;
    LimbPartition& c0 = aux;
    int num_d = 0;
    {
        int start = 0;
        if constexpr (PRINT)
            std::cout << "/** Compute how many digits are used at this level*/" << std::endl;
        while (num_d < cc.dnum && start < *level + 1) {
            start += DECOMPmeta.at(num_d).size();
            num_d++;
        }
    }
    uint32_t limb_size = 0;
    while (limb_size < meta.size() && meta[limb_size].id <= *level)
        limb_size++;

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Input: ";
            for (size_t i = 0; i < limb_size; ++i) {
                std::cout << meta[i].id;
                SWITCH(limb[i], printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }
    if constexpr (PRINT)
        std::cout
            << "/** We try to pipeline the computation of each digit first, splitting independent groups of limbs*/"
            << std::endl;
    const int digits_per_it = num_d;  //cc.logN <= 15 ? num_d : cc.logN == 16 ? std::max((num_d + 1) / 2, 1) : 1;
    if constexpr (PRINT)
        std::cout << "GPU " << id << "compute " << num_d << " digits" << std::endl;
    for (int d = 0; d < num_d; d += digits_per_it) {
        int ds = std::min(num_d - d, digits_per_it);
        uint32_t start_d = 0;
        while (start_d < limb_size && meta[start_d].digit < d)
            start_d++;
        uint32_t size_d = 0;
        while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
            size_d++;
        if constexpr (PRINT)
            if (SELECT) {
                std::cout << "GPU " << id << " for digits " << d << ":" << d + digits_per_it << " INTT " << size_d
                          << " limbs starting at limb " << start_d << std::endl;
            }
        Stream& stream = cc.digitStream.at(d).at(id);
        stream.wait(s);
        if constexpr (PRINT)
            std::cout << "/** Intt */" << std::endl;
        if (size_d > 0) {
            constexpr ALGO algo = ALGO_SHOUP;
            constexpr int M = 4;
            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int gather_offset = 0;
            for (int i = 0; i < id; ++i) {
                gather_offset += cc.meta.at(i).size();
            }
            {
                INTT_<false, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), size_d}, blockDimFirst, bytesFirst, stream.ptr>>>(
                        limbptr.data + start_d, PARTITION(id, start_d), auxptr.data + start_d);
                INTT_<true, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), size_d}, blockDimSecond, bytesSecond, stream.ptr>>>(
                        auxptr.data + start_d, PARTITION(id, start_d), GATHERptr.data + gather_offset + start_d);
            }
        }
        if constexpr (PRINT) {
            if (SELECT) {
                cudaDeviceSynchronize();
                std::cout << "GPU: " << id << "Out INTT: ";
                for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                    for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
                        std::cout << DECOMPmeta[j][i].id;
                        SWITCH(DECOMPlimb[j][i], printThisLimb(2));
                    }
                }
                std::cout << std::endl;
                cudaDeviceSynchronize();
            }
        }
        if constexpr (PRINT)
            std::cout << "/** Communicate */" << std::endl;
        {
            size_t j = id;
#ifdef NCCL
            NCCLCHECK(ncclGroupStart());
            int start = 0;
            for (size_t i = 0; i < cc.GPUid.size(); ++i) {
                uint32_t limb_size_i = 0;
                while (limb_size_i < cc.meta[i].size() && cc.meta[i][limb_size_i].id <= *level)
                    limb_size_i++;
                uint32_t start_d_i = 0;
                while (start_d_i < limb_size_i && cc.meta[i][start_d_i].digit < d)
                    start_d_i++;
                uint32_t size_d_i = 0;
                while (start_d_i + size_d_i < limb_size_i && cc.meta[i][start_d_i + size_d_i].digit < d + ds)
                    size_d_i++;
                if constexpr (PRINT)
                    if (SELECT) {
                        std::cout << "GPU " << i << " for digits " << d << ":" << d + digits_per_it << " communicate "
                                  << size_d_i << " limbs" << std::endl;
                    }
                if (size_d_i > 0) {

                    NCCLCHECK(ncclBroadcast(/*bufferLIMB + cc.N * start_d*/ bufferGATHER + cc.N * (start + start_d_i),
                                            bufferGATHER + cc.N * (start + start_d_i), size_d_i * cc.N, ncclUint64, i,
                                            rank, stream.ptr));
                }
                start += cc.meta[i].size();
            }
            NCCLCHECK(ncclGroupEnd());
#else
            assert(false);
#endif
        }
    }
    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Out INTT after communicate: ";
            for (size_t j = 0; j < DECOMPlimb.size(); ++j) {
                for (size_t i = 0; i < DECOMPlimb[j].size(); ++i) {
                    std::cout << DECOMPmeta[j][i].id;
                    SWITCH(DECOMPlimb[j][i], printThisLimb(2));
                }
                std::cout << std::endl;
            }
            cudaDeviceSynchronize();
        }
    }
    for (int d = 0; d < num_d; d += digits_per_it) {
        int ds = std::min(num_d - d, digits_per_it);
        uint32_t start_d = 0;
        while (start_d < limb_size && meta[start_d].digit < d)
            start_d++;
        uint32_t size_d = 0;
        while (start_d + size_d < limb_size && meta[start_d + size_d].digit < d + ds)
            size_d++;
        Stream& stream = cc.digitStream.at(d).at(id);
        if constexpr (PRINT)
            std::cout << "/** Conv */" << std::endl;
        for (int d_ = d; d_ < d + ds; ++d_) {
            Stream& stream1 = cc.digitStream.at(d_).at(id);
            stream1.wait(stream);
        }
        for (int d_ = d; d_ < d + ds; ++d_) {
            Stream& stream1 = cc.digitStream.at(d_).at(id);
            int start = 0;
            for (int j = 0; j < d_; ++j)
                start += DECOMPlimb.at(j).size();
            int size = std::min((int)DECOMPlimb.at(d_).size(), *level + 1 - start);
            if (size <= 0) {
                std::cerr << "void modup, aborting" << std::endl;
                exit(-1);
            }
            if constexpr (PRINT)
                if (SELECT) {
                    std::cout << host_constants_per_gpu[id].num_primeid_digit_to[d_][*level]
                              << "<- num_prime_id_digit_to: " << d_ << std::endl;
                    std::cout << host_constants_per_gpu[id].num_primeid_digit_from[d_][*level]
                              << "<- num_prime_id_digit_from: " << d_ << std::endl;
                    /*
                        std::cout << host_constants_per_gpu[id].num_primeid_digit_to[d_][level - 1]
                                  << "<- num_prime_id_digit_to: " << d_ << std::endl;
                        std::cout << host_constants_per_gpu[id].num_primeid_digit_from[d_][level - 1]
                                  << "<- num_prime_id_digit_from: " << d_ << std::endl;
                                  */
                }
            dim3 blockSize{64, 2};
            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (size /*DECOMPlimb[d].size()*/) * blockSize.x;
            DecompAndModUpConv<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream1.ptr>>>(
                DECOMPlimbptr[d_].data, *level + 1, DIGITlimbptr[d_].data, digitid[d_]);
            cc.digitStream2.at(d_).at(id).wait(stream1); /** Get dependency for limb NTTs later */
            if constexpr (PRINT)
                std::cout << "/** NTT special limbs */" << std::endl;
            {
                uint32_t size = cc.splitSpecialMeta.at(id).size();
                constexpr ALGO algo = ALGO_SHOUP;
                constexpr int M = 4;
                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                if (size > 0) {
                    NTT_<false, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimFirst.x * M * 2), size}, blockDimFirst, bytesFirst, stream1.ptr>>>(
                            DIGITlimbptr[d_].data, DIGIT(d_, 0), c0.DIGITlimbptr[d_].data);
                    NTT_<true, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), size}, blockDimSecond, bytesSecond, stream1.ptr>>>(
                            c0.DIGITlimbptr[d_].data, DIGIT(d_, 0), DIGITlimbptr[d_].data);
                }
            }
        }
    }
    for (int d = 0; d < num_d; ++d) {
        s.wait(cc.digitStream.at(d).at(id));
    }
    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Out ModUp after NTT specials: ";
            for (size_t j = 0; j < DIGITlimb.size(); ++j) {
                for (size_t i = 0; i < DIGITlimb[j].size(); ++i) {
                    std::cout << DIGITmeta[j][i].id;
                    SWITCH(DIGITlimb[j][i], printThisLimb(2));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }
    c0.s.wait(s);
    if constexpr (PRINT)
        std::cout << "/** We delay the call of NTTs post-modup for non special limbs to here*/" << std::endl;
    for (int d = 0; d < num_d; ++d) {
        Stream& stream = cc.digitStream2.at(d).at(id);
        if (limb_size > 0) {
            uint32_t start = cc.splitSpecialMeta.at(id).size();
            uint32_t size = host_constants_per_gpu[id].num_primeid_digit_to[d][*level] - start;
            if (size > 0) {
                constexpr ALGO algo = ALGO_SHOUP;
                constexpr int M = 4;
                dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
                dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
                int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
                {
                    NTT_<false, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimFirst.x * M * 2), size}, blockDimFirst, bytesFirst, stream.ptr>>>(
                            DIGITlimbptr[d].data + start, DIGIT(d, start), c0.DIGITlimbptr[d].data + start);
                    NTT_<true, algo, NTT_NONE>
                        <<<dim3{cc.N / (blockDimSecond.x * M * 2), size}, blockDimSecond, bytesSecond, stream.ptr>>>(
                            c0.DIGITlimbptr[d].data + start, DIGIT(d, start), DIGITlimbptr[d].data + start);
                }
            }
            s.wait(stream);
        }
    }
    c0.s.wait(s);
}

void LimbPartition::moddownMGPU(LimbPartition& auxLimbs, bool ntt, bool free_special_limbs) {
    cudaSetDevice(device);
    constexpr bool PRINT = false;
    bool SELECT = id == 1;
    LimbPartition& c1 = *this;

    uint32_t limb_size = getLimbSize(*level);

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Input: ";
            for (size_t i = 0; i < limb_size; ++i) {
                std::cout << meta[i].id;
                SWITCH(limb[i], printThisLimb(2));
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }

    {

        Stream& stream = s;
        stream.wait(auxLimbs.s);
        LimbPartition& out = c1;
        if constexpr (PRINT)
            std::cout << "/** INTT specials*/" << std::endl;
        {
            constexpr ALGO algo = ALGO_SHOUP;
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            const int i = cc.splitSpecialMeta.at(id).at(0).id - cc.specialMeta.at(id).at(0).id;
            const uint32_t limbs = cc.splitSpecialMeta.at(id).size();

            if (limbs > 0) {
                INTT_<false, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), limbs}, blockDimFirst, bytesFirst, stream.ptr>>>(
                        out.SPECIALlimbptr.data + i, SPECIAL(id, i), out.SPECIALauxptr.data + i);

                INTT_<true, algo, INTT_NONE>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), limbs}, blockDimSecond, bytesSecond, stream.ptr>>>(
                        out.SPECIALauxptr.data + i, SPECIAL(id, i), out.SPECIALlimbptr.data + i);
            }

            if constexpr (PRINT)
                std::cout << "/** communicate */" << std::endl;
#ifdef NCCL
            NCCLCHECK(ncclGroupStart());
            for (size_t i = 0; i < cc.splitSpecialMeta.size(); ++i) {
                //Limb<uint64_t>& l = std::get<U64>(
                //    this->SPECIALlimb.at(cc.splitSpecialMeta.at(i).at(0).id - cc.specialMeta.at(id).at(0).id));
                const uint32_t num_limbs = cc.splitSpecialMeta.at(i).size();
                //uint64_t* ptr2 = l.v.data;
                uint64_t* ptr = out.bufferSPECIAL + (cc.splitSpecialMeta.at(i).at(0).id - SPECIALmeta.at(0).id) * cc.N;

                if (num_limbs > 0)
                    NCCLCHECK(ncclBroadcast(ptr, ptr, cc.N * num_limbs, ncclUint64, (int)i, rank, stream.ptr));
            }
            NCCLCHECK(ncclGroupEnd());
#else
            assert(false);
#endif
        }
    }

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "KSK specials after INTT and communicate: ";
            for (const auto& j : {&c1}) {
                for (auto& i : j->SPECIALlimb) {
                    SWITCH(i, printThisLimb(2));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }

    {
        Stream& stream = c1.s;
        LimbPartition& out = c1;
        if constexpr (PRINT)
            std::cout << "/** Conv */" << std::endl;
        stream.wait(auxLimbs.s);
        {
            dim3 blockSize{64, 2};

            dim3 gridSize{(uint32_t)cc.N / blockSize.x};
            int shared_bytes = sizeof(uint64_t) * (SPECIALlimb.size()) * blockSize.x;
            if (limb_size > 0)
                ModDown2<ALGO_SHOUP><<<gridSize, blockSize, shared_bytes, stream.ptr>>>(
                    auxLimbs.limbptr.data, limb_size, out.SPECIALlimbptr.data, PARTITION(id, 0));
        }
    }

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Out Moddown: ";
            for (const auto& j : {&auxLimbs}) {
                for (auto& i : j->limb) {
                    SWITCH(i, printThisLimb(2));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }

    {
        if constexpr (PRINT)
            std::cout << "/** Last NTT step for moddown*/" << std::endl;
        Stream& stream = c1.s;
        LimbPartition& out = c1;

        if (limb_size > 0) {
            constexpr ALGO algo = ALGO_SHOUP;
            constexpr int M = 4;

            dim3 blockDimFirst{(uint32_t)(1 << ((cc.logN) / 2 - 1))};
            dim3 blockDimSecond = dim3{(uint32_t)(1 << ((cc.logN + 1) / 2 - 1))};
            int bytesFirst = 8 * blockDimFirst.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));
            int bytesSecond = 8 * blockDimSecond.x * (2 * M + 1 + (algo == 2 || algo == 3 ? 1 : 0));

            {
                NTT_<false, algo, NTT_MODDOWN>
                    <<<dim3{cc.N / (blockDimFirst.x * M * 2), limb_size}, blockDimFirst, bytesFirst, stream.ptr>>>(
                        auxLimbs.limbptr.data, PARTITION(id, 0), out.auxptr.data);

                stream.wait(cc.digitStream2.at(0).at(id));

                NTT_<true, algo, NTT_MODDOWN>
                    <<<dim3{cc.N / (blockDimSecond.x * M * 2), limb_size}, blockDimSecond, bytesSecond, stream.ptr>>>(
                        out.auxptr.data, PARTITION(id, 0), out.limbptr.data);
            }
        }
        auxLimbs.s.wait(stream);
    }

    if constexpr (PRINT) {
        if (SELECT) {
            cudaDeviceSynchronize();
            std::cout << "GPU: " << id << "Out Moddown after submult: ";
            for (const auto& j : {&c1}) {
                for (auto& i : j->limb) {
                    SWITCH(i, printThisLimb(2));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            cudaDeviceSynchronize();
        }
    }
}

void LimbPartition::broadcastLimb0_mgpu() {
    cudaSetDevice(device);
    static bool parity = true;
    const int limbsize = getLimbSize(*level);

    Stream& stream = parity ? cc.top_limb_stream[id] : cc.top_limb_stream2[id];
    uint64_t* buffer = parity ? cc.top_limb_buffer[id] : cc.top_limb_buffer2[id];
    VectorGPU<void*>& ptr = parity ? cc.top_limbptr[id] : cc.top_limbptr2[id];

    stream.wait(s);
    bool skip0 = meta[0].id == 0;

    if (skip0) {
        uint64_t* src_ptr = std::get<U64>(limb[0]).v.data;
        cudaMemcpyAsync(buffer, src_ptr, cc.N * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream.ptr);
    }
    /*
#ifdef NCCL
    { NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr)); }
#else
    assert(false);
#endif
*/
#ifdef NCCL
    if constexpr (0) {
        NCCLCHECK(ncclBroadcast(buffer, buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr));
    } else {
        NCCLCHECK(ncclGroupStart());
        if (id == cc.limbGPUid[0].x) {
            for (int i = 0; i < cc.GPUid.size(); ++i) {
                if (i != id)
                    ncclSend(buffer, cc.N, ncclUint64, i, rank, stream.ptr);
            }
        } else {
            ncclRecv(buffer, cc.N, ncclUint64, cc.limbGPUid[0].x, rank, stream.ptr);
        }
        NCCLCHECK(ncclGroupEnd());
    }
#else
    assert(false);
#endif
    if (limbsize - skip0 > 0) {
        CKKS::broadcastLimb0_mgpu<<<dim3{(uint32_t)cc.N / 128, (uint32_t)limbsize - skip0}, 128, 0, stream.ptr>>>(
            limbptr.data + skip0, PARTITION(id, skip0), ptr.data);
    }
    s.wait(stream);
}

}  // namespace FIDESlib::CKKS
