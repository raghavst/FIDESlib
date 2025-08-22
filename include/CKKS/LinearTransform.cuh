//
// Created by carlosad on 7/05/25.
//

#ifndef FIDESLIB_LINEARTRANSFORM_CUH
#define FIDESLIB_LINEARTRANSFORM_CUH

#include <vector>
#include "forwardDefs.cuh"

namespace FIDESlib::CKKS {

void LinearTransform(Ciphertext& ctxt, int rowSize, int bStep, std::vector<Plaintext*> pts, int stride = 1,
                     int offset = 0);

std::vector<int> GetLinearTransformRotationIndices(int bStep, int stride = 1, int offset = 0);
std::vector<int> GetLinearTransformPlaintextRotationIndices(int rowSize, int bStep, int stride = 1, int offset = 0);

void LinearTransformSpecial(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt2,
                            FIDESlib::CKKS::Ciphertext& ctxt3, int rowSize, int bStep, std::vector<Plaintext*> pts1,
                            std::vector<Plaintext*> pts2, int stride, int stride3);
/*
    void LinearTransformPt(FIDESlib::CKKS::Plaintext& ptxt, FIDESlib::CKKS::Context& cc, int rowSize, int bStep,
                                        std::vector<Plaintext*> pts, int stride, int offset);
*/
void LinearTransformSpecialPt(FIDESlib::CKKS::Ciphertext& ctxt1, FIDESlib::CKKS::Ciphertext& ctxt3,
                              FIDESlib::CKKS::Plaintext& ptxt, int rowSize, int bStep, std::vector<Plaintext*> pts1,
                              std::vector<Plaintext*> pts2, int stride, int stride3);

}  // namespace FIDESlib::CKKS
#endif  //FIDESLIB_LINEARTRANSFORM_CUH