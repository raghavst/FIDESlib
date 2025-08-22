//
// Created by carlosad on 4/11/24.
//

#ifndef GPUCKKS_FORWARDDEFS_CUH
#define GPUCKKS_FORWARDDEFS_CUH

namespace FIDESlib::CKKS {
class Context;
class Ciphertext;
class KeySwitchingKey;
class Plaintext;
class RNSPoly;
template <typename T>
class Limb;
class Parameters;
class BootstrapPrecomputation;
}  // namespace FIDESlib::CKKS

namespace FIDESlib {
enum ALGO { ALGO_NATIVE = 0, ALGO_NONE = 1, ALGO_SHOUP = 3, ALGO_BARRETT = 4, ALGO_BARRETT_FP64 = 5 };
constexpr ALGO DEFAULT_ALGO = ALGO_BARRETT;
}  // namespace FIDESlib
// namespace FIDESlib::CKKS

#endif  //GPUCKKS_FORWARDDEFS_CUH
