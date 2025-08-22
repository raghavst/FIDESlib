//
// Created by carlosad on 26/09/24.
//

#include <source_location>
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"

namespace FIDESlib::CKKS {
void KeySwitchingKey::Initialize(Context& cc, RawKeySwitchKey& rkk) {
    CudaNvtxRange r(std::string{std::source_location::current().function_name()}.substr(23 + strlen(loc)));

    //if (cc.GPUid.size() == 1) {
    for (int j = 0; j < cc.dnum; ++j) {
        a.generateDecompAndDigit(true);
        b.generateDecompAndDigit(true);
    }
    if (cc.GPUid.size() > 1) {
        a.grow(cc.L, true, true);
        b.grow(cc.L, true, true);
    }
    a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
    b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);
    /* } else {
        for (int j = 0; j < cc.dnum; ++j) {
            mgpu_a[j].grow(cc.L, true, true);
            mgpu_b[j].grow(cc.L, true, true);
            mgpu_a[j].generatePartialSpecialLimbs();
            mgpu_b[j].generatePartialSpecialLimbs();
            mgpu_a[j].loadConstant(rkk.r_key[0][j], rkk.r_key_moduli[0][j]);
            mgpu_b[j].loadConstant(rkk.r_key[1][j], rkk.r_key_moduli[1][j]);
        }
    }*/

    cudaDeviceSynchronize();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
    : my_range(loc, LIFETIME),
      cc((CudaNvtxStart(std::string{std::source_location::current().function_name()}.substr(18 + strlen(loc))), cc)),
      a(cc, -1),
      b(cc, -1) {
    CudaNvtxStop();
    /*
    if (cc.GPUid.size() > 1) {
        for (int j = 0; j < cc.dnum; ++j) {
            mgpu_a.emplace_back(cc, -1);
            mgpu_b.emplace_back(cc, -1);
        }
    }
     */
}
}  // namespace FIDESlib::CKKS
