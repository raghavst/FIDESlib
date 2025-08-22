//
// Created by carlosad on 27/09/24.
//

#ifndef GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
#define GPUCKKS_ELEMENWISEBATCHKERNELS_CUH

#include "AddSub.cuh"
#include "ConstantsGPU.cuh"
#include "ModMult.cuh"

namespace FIDESlib::CKKS {
__global__ void mult1AddMult23Add4_(const __grid_constant__ int partition, void** l, void** l1, void** l2, void** l3,
                                    void** l4);

__global__ void multnomoddownend_(const __grid_constant__ int primeid_init, void** c1, void** c0, void** bc0,
                                  void** bc1, void** in, void** aux);

__global__ void mult1Add2_(const __grid_constant__ int partition, void** l, void** l1, void** l2);

template <typename T>
__global__ void addMult_(T* l, const T* l1, const T* l2, const __grid_constant__ int primeid);

__global__ void addMult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void Mult_(void** l, void** l1, void** l2, const __grid_constant__ int primeid_init);

__global__ void square_(void** l, void** l1, const __grid_constant__ int primeid_init);

__global__ void binomial_square_fold_(void** c0_res, void** c2_key_switched_0, void** c1, void** c2_key_switched_1,
                                      const __grid_constant__ int primeid_init);
template <ALGO algo>
__global__ void Scalar_mult_(void** a, const uint64_t* b, const __grid_constant__ int primeid,
                             const uint64_t* shoup_mu);

__global__ void broadcastLimb0_(void** a);
__global__ void broadcastLimb0_mgpu(void** a, const __grid_constant__ int primeid_init, void** limb0);
__global__ void copy_(void** a, void** b);
__global__ void copy1D_(void* a, void* b);
__global__ void eval_linear_w_sum_(const __grid_constant__ int n, void** a, void*** bs, uint64_t* w,
                                   const __grid_constant__ int primeid_init);
__global__ void fusedDotKSK_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id,
                             int num_special);
__global__ void fusedDotKSK_2_(void** out1, void** sout1, void** out2, void** sout2, void*** digits, int num_d, int id,
                               int num_special, int init);
__global__ void dotProductPt_(void** c0, void** c1, void*** data, const size_t ptroffset, const int primeidInit,
                              const int n);

__global__ void addScaleB_(void** a, void** b, void** c, const int primeid_init);
__global__ void scaleByP_(void** a, const int primeid_init);

}  // namespace FIDESlib::CKKS

#endif  //GPUCKKS_ELEMENWISEBATCHKERNELS_CUH
