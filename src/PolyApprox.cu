//
// Created by seyda on 5/19/25.
//

#include "PolyApprox.cuh"

#include "CKKS/AccumulateBroadcast.cuh"

// coefficient source: openfhe
// std::vector<double> cheb_coeff_inv_softmax = {0.000596892, -0.000558047, 0.000521437, -0.000486914, 0.000454341, -0.000423587, 0.000394529, -0.000367051, 0.000341042, -0.000316399, 0.000293023, -0.00027082, 0.000249701, -0.000229582, 0.000210383, -0.000192026, 0.000174437, -0.000157548, 0.000141289, -0.000125595, 0.000110405, -9.56565e-05, 8.12912e-05, -6.72513e-05, 5.34807e-05, -3.99243e-05, 2.65277e-05, -1.32374e-05};
std::vector<double> cheb_coeff_inv_softmax = {
    0.00110844,   -0.00106846, 0.00102853,   -0.000988633, 0.000948778,  -0.000908961, 0.000869181,
    -0.000829435, 0.000789722, -0.000750041, 0.00071039,   -0.000670768, 0.000631172,  -0.000591602,
    0.000552055,  -0.00051253, 0.000473026,  -0.000433541, 0.000394073,  -0.000354621, 0.000315183,
    -0.000275757, 0.000236343, -0.000196938, 0.000157541,  -0.00011815,  7.87641e-05,  -3.93813e-05};

std::vector<double> cheb_coeff_exp_softmax = {
    2.53213,     1.13032,      0.271495,     0.0443368,    0.00547424,   0.000542926,  4.49773e-05,
    3.19844e-06, 1.99212e-07,  1.10368e-08,  5.5059e-10,   2.49795e-11,  1.03914e-12,  3.99088e-14,
    1.76514e-15, 1.96862e-16,  -1.32079e-17, 1.40569e-16,  1.23913e-16,  -1.43947e-16, 3.08643e-17,
    -2.166e-16,  -3.91206e-17, -3.18494e-16, -6.44747e-16, -8.80072e-17, 1.77988e-16,  7.41657e-16};
// std::vector<double> cheb_coeff_exp_softmax = {855.128, 799.746, 655.192, 472.15, 301.079, 171.072, 87.2393, 40.2126, 16.8672, 6.47821, 2.29123, 0.750124, 0.228393, 0.0649468, 0.0173153, 0.00434323, 0.00102821, 0.000230402, 4.89976e-05, 9.91275e-06, 1.91208e-06, 3.52366e-07, 6.21554e-08, 1.05127e-08, 1.70769e-09, 2.66972e-10, 4.02587e-11, 5.58214e-12};

std::vector<double> cheb_coeff_inv_layernorm_1_1 = {
    127.344,    42.4221,   -8.46963,   3.61938,   -2.00263,   1.26769,   -0.871875,  0.634292,  -0.480453, 0.375074,
    -0.299673,  0.243803,  -0.201194,  0.167896,  -0.141319,  0.119709,  -0.101838,  0.0868283, -0.074037, 0.0629822,
    -0.0532966, 0.0446942, -0.0369485, 0.0298763, -0.0233261, 0.0171695, -0.0112952, 0.00560292};
std::vector<double> cheb_coeff_inv_layernorm_1_2 = {
    0.353553,     -0.247331,   0.173022,   -0.121039,    0.0846739,    -0.0592343,   0.0414378,
    -0.0289881,   0.0202789,   -0.0141862, 0.00992407,   -0.00694244,  0.00485661,   -0.00339744,
    0.00237665,   -0.00166253, 0.00116292, -0.000813371, 0.000568771,  -0.000397561, 0.000277649,
    -0.000193562, 0.000134452, -9.269e-05, 6.28881e-05,  -4.12008e-05, 2.48297e-05,  -1.16624e-05};

// 1-10000
// std::vector<double> cheb_coeff_inv_layernorm = {0.068943, -0.0434804, 0.034997, -0.0299093, 0.0262758, -0.0234492, 0.0211346, -0.0191734, 0.01747, -0.0159626, 0.0146086, -0.0133776, 0.0122469, -0.0111994, 0.0102214, -0.00930209, 0.00843266, -0.00760575, 0.00681518, -0.00605567, 0.00532264, -0.00461205, 0.00392031, -0.00324418, 0.00258069, -0.00192707, 0.00128073, -0.000639178};

// 1-1000
std::vector<double> cheb_coeff_inv_layernorm = {
    0.193447,    -0.113051,  0.086548,    -0.0709154, 0.0599841,   -0.0516867,  0.0450768,
    -0.0396408,  0.035068,   -0.0311551,  0.0277617,  -0.0247864,  0.0221533,   -0.0198042,
    0.0176931,   -0.0157829, 0.0140435,   -0.0124496, 0.01098,     -0.00961655, 0.00834341,
    -0.00714675, 0.00601424, -0.00493476, 0.00389816, -0.00289503, 0.00191654,  -0.000954249};

// relu
// std::vector<double> cheb_coeff_relu_27 = {0.636876, 0.5, 0.21195, 5.85982e-18, -0.0421821, -9.86934e-18, 0.0179255, 8.56513e-18, -0.00983515, 5.22852e-20, 0.00615224, 5.3345e-18, -0.00416308, -4.64829e-18, 0.00296292, 4.18764e-17, -0.002179, -1.03381e-17, 0.00163451, -3.85117e-17, -0.00123654, 4.37076e-17, 0.000932221, -1.60433e-16, -0.000689416, -1.58483e-16, 0.000487353, 1.9027e-16, -0.000311761, -2.70723e-16, 0.000152124, -1.28824e-16};
// sqrt
std::vector<double> cheb_coeff_relu = {
    1.80068,      0.600159,    -0.119991,    0.0513951,   -0.0285298,   0.0181365,   -0.01254,     0.00918198,
    -0.00700909,  0.00552228,  -0.00446019,  0.00367508,  -0.0030783,   0.00261405,  -0.00224576,  0.00194865,
    -0.00170547,  0.00150387,  -0.00133487,  0.00119176,  -0.00106949,  0.00096418,  -0.0008728,   0.000792975,
    -0.000722813, 0.00066079,  -0.000605671, 0.000556446, -0.000512281, 0.000472482, -0.000436469, 0.000403756,
    -0.000373929, 0.000346635, -0.000321573, 0.000298485, -0.000277145, 0.000257359, -0.000238958, 0.000221792,
    -0.00020573,  0.000190657, -0.000176469, 0.000163076, -0.000150395, 0.000138352, -0.000126882, 0.000115925,
    -0.000105425, 9.5332e-05,  -8.56009e-05, 7.61888e-05, -6.7056e-05,  5.81658e-05, -4.94833e-05, 4.09757e-05,
    -3.26115e-05, 2.43607e-05, -1.61941e-05, 8.08316e-06};

std::vector<double> cheb_coeff_relu_59 = {
    0.636693,     0.5,          0.212134,    -2.19406e-17, -0.0423683,   -1.7443e-17,  0.0181158,   3.13185e-17,
    -0.0100312,   2.5175e-18,   0.00635601,  3.0074e-18,   -0.0043766,   5.96322e-18,  0.00318849,  -2.37013e-17,
    -0.00241921,  -3.10727e-17, 0.00189235,  2.81345e-17,  -0.00151548,  -3.38055e-17, 0.00123636,  3.52938e-17,
    -0.00102365,  6.60064e-17,  0.000857588, 8.96365e-17,  -0.000725245, 2.1578e-17,   0.000617843, -4.97543e-17,
    -0.000529262, 2.05412e-19,  0.000455122, 3.00962e-17,  -0.000392217, 1.48616e-16,  0.000338156, -6.86962e-17,
    -0.000291121, 1.17556e-16,  0.000249705, 1.08019e-16,  -0.000212803, -1.1876e-16,  0.000179529, 7.56163e-17,
    -0.000149165, -1.67184e-16, 0.000121114, 7.1472e-17,   -9.48744e-05, 1.66073e-16,  7.00109e-05, 1.07589e-16,
    -4.61397e-05, 1.22269e-16,  2.29117e-05, 5.61582e-16};
std::vector<double> cheb_coeff_relu_59_2 = {
    0.636693,     0.5,          0.212134,    -2.19406e-17, -0.0423683,   -1.7443e-17,  0.0181158,   3.13185e-17,
    -0.0100312,   2.5175e-18,   0.00635601,  3.0074e-18,   -0.0043766,   5.96322e-18,  0.00318849,  -2.37013e-17,
    -0.00241921,  -3.10727e-17, 0.00189235,  2.81345e-17,  -0.00151548,  -3.38055e-17, 0.00123636,  3.52938e-17,
    -0.00102365,  6.60064e-17,  0.000857588, 8.96365e-17,  -0.000725245, 2.1578e-17,   0.000617843, -4.97543e-17,
    -0.000529262, 2.05412e-19,  0.000455122, 3.00962e-17,  -0.000392217, 1.48616e-16,  0.000338156, -6.86962e-17,
    -0.000291121, 1.17556e-16,  0.000249705, 1.08019e-16,  -0.000212803, -1.1876e-16,  0.000179529, 7.56163e-17,
    -0.000149165, -1.67184e-16, 0.000121114, 7.1472e-17,   -9.48744e-05, 1.66073e-16,  7.00109e-05, 1.07589e-16,
    -4.61397e-05, 1.22269e-16,  2.29117e-05, 5.61582e-16};

std::vector<double> cheb_coeff_tanh_27 = {
    6.34413e-17,  0.811676,     3.73497e-17,  -0.0542458,   6.31781e-17,  0.00451682,  4.92256e-17,
    -0.000382447, 5.41393e-17,  3.24439e-05,  3.83674e-17,  -2.75295e-06, 5.64483e-17, 2.33602e-07,
    -4.07099e-16, -1.98224e-08, 7.19972e-17,  1.68204e-09,  6.3352e-17,   -1.4273e-10, 8.77422e-17,
    1.21112e-11,  9.46603e-17,  -1.02738e-12, -1.16845e-16, 8.73335e-14,  2.54632e-16, -9.19349e-15};

namespace FIDESlib::CKKS {

void evalFunction(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey,
                  std::vector<double> cheb_coeff, int numSlots, double lower_bound, double upper_bound, bool bts) {
    // affine transformation to scale
    if (!(lower_bound == -1.0 && upper_bound == 1.0)) {
        double scale = 2.0 / (upper_bound - lower_bound);
        double shift = -(upper_bound + lower_bound) / (upper_bound - lower_bound);

        if (abs(scale - 1.0) > 1e-4) {
            ctxt.multScalar(scale);
        }
        ctxt.addScalar(shift);
        lower_bound = -1.0;
        upper_bound = 1.0;
        // std::cout << "# limbs at ch, 0: " << ctxt.getLevel() << std::endl;
    }
    //std::cout << "Input eval function level" << ctxt.getLevel() << " " << ctxt.NoiseLevel << std::endl;

    if (bts) {
        Bootstrap(ctxt, numSlots);
    }
    evalChebyshevSeries(ctxt, keySwitchingKey, cheb_coeff, lower_bound, upper_bound);
}

void evalTanh(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots,
              double lower_bound, double upper_bound, bool bts) {

    evalFunction(ctxt, keySwitchingKey, cheb_coeff_tanh_27, numSlots, lower_bound, upper_bound, bts);
}

std::vector<double> get_chebyshev_coefficients(const std::function<double(double)>& func, const double a,
                                               const double b, const uint32_t degree) {
    if (!degree) {
        OPENFHE_THROW("The degree of approximation can not be zero");
    }

    const size_t coeffTotal{degree + 1};
    const double bMinusA = 0.5 * (b - a);
    const double bPlusA = 0.5 * (b + a);
    const double PiByDeg = M_PI / static_cast<double>(coeffTotal);
    std::vector<double> functionPoints(coeffTotal);
    for (size_t i = 0; i < coeffTotal; ++i)
        functionPoints[i] = func(std::cos(PiByDeg * (i + 0.5)) * bMinusA + bPlusA);

    const double multFactor = 2.0 / static_cast<double>(coeffTotal);
    std::vector<double> coefficients(coeffTotal);
    for (size_t i = 0; i < coeffTotal; ++i) {
        for (size_t j = 0; j < coeffTotal; ++j)
            coefficients[i] += functionPoints[j] * std::cos(PiByDeg * i * (j + 0.5));
        coefficients[i] *= multFactor;
    }
    return coefficients;
}

void evalRelu(FIDESlib::CKKS::Ciphertext& ctxt, const KeySwitchingKey& keySwitchingKey, int numSlots) {

    auto scale = GetPreScaleFactor(ctxt.cc, numSlots);
    auto cheb_coeff_relu_scaled = get_chebyshev_coefficients(
        [scale](const double x) -> double {
            if (x <= 0.00001)
                return 0;
            return (scale)*sqrtf64(x);
        },
        0.0, 2.0, 27);

    //std::cout << cheb_coeff_relu_scaled.size() << std::endl;
    //cheb_coeff_relu_scaled.resize(59);

    evalFunction(ctxt, keySwitchingKey, cheb_coeff_relu_scaled, numSlots, 0.0, 2.0);
}

void EvalSoftmax(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, FIDESlib::CKKS::Ciphertext& ctxt,
                 const KeySwitchingKey& keySwitchingKey, int numSlots, int blockSize, int bStepAcc, bool bts) {

    static FIDESlib::CKKS::Plaintext* GPUpt_softmax = nullptr;
    static double scale;
    FIDESlib::CKKS::Context& cc = ctxt.cc;

    // Exponential
    evalFunction(ctxt, keySwitchingKey, cheb_coeff_exp_softmax, numSlots, -1, 1);

    if (bts) {
        Bootstrap(ctxt, numSlots);
    }

    for (int i = 0; i < 3; i++) {
        ctxt.mult(ctxt, ctxt, keySwitchingKey);
    }

    // auto scores_sum = rotsum_GPU(ctxt, blockSize, padding);
    Ciphertext scores_sum(ctxt.cc);
    scores_sum.copy(ctxt);
    FIDESlib::CKKS::Accumulate(scores_sum, bStepAcc, 1, blockSize);

    if (GPUpt_softmax == nullptr) {
        std::vector<double> mask(numSlots, 0.0);
        for (int i = 0; i < numSlots; i = i + blockSize) {
            mask[i] = 1;
        }

        auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask));
        GPUpt_softmax = new FIDESlib::CKKS::Plaintext(cc, raw_pt);

        scale = GetPreScaleFactor(cc, numSlots);
    }
    scores_sum.multPt(*GPUpt_softmax);

    if (scores_sum.NoiseLevel == 2)
        scores_sum.rescale();

    if (1) {
        Broadcast(scores_sum, bStepAcc, 1, blockSize);
    } else {
        for (int j = 1; j < blockSize; j = j * 2) {
            //std::cout << -j << std::endl;
            FIDESlib::CKKS::Ciphertext rotated(cc);
            rotated.copy(scores_sum);
            rotated.rotate(-j, cc.GetRotationKey(-j));
            scores_sum.add(rotated);
        }
    }

    // 1/x
    evalFunction(scores_sum, keySwitchingKey, cheb_coeff_inv_softmax, numSlots, 1, 100000, bts);

    /*
    if (bts) {
        Bootstrap(scores_sum, numSlots);
    }
    */
    ctxt.multScalar(scale);

    ctxt.mult(scores_sum, keySwitchingKey);
}

void EvalSoftmax_Matrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                        std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
                        const KeySwitchingKey& keySwitchingKey, int numSlots, int blockSize, int bStepAcc, bool bts) {

    for (size_t i = 0; i < ctxt.size(); i++) {
        for (size_t j = 0; j < ctxt[0].size(); j++) {
            EvalSoftmax(context, ctxt[i][j], keySwitchingKey, numSlots, blockSize, bStepAcc, bts);
        }
    }
}

void EvalRelu_Matrix(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt, const KeySwitchingKey& keySwitchingKey,
                     int numSlots) {

    FIDESlib::CKKS::Context& cc = ctxt[0][0].cc;

    for (size_t i = 0; i < ctxt.size(); i++) {
        for (size_t j = 0; j < ctxt[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext ctxt_abs(cc);
            ctxt_abs.mult(ctxt[i][j], ctxt[i][j], keySwitchingKey);
            evalRelu(ctxt_abs, keySwitchingKey, numSlots);
            ctxt[i][j].multScalar(GetPreScaleFactor(cc, numSlots));
            ctxt[i][j].add(ctxt_abs);  // output is scaled by 2
        }
    }
}

void EvalLayerNorm(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, FIDESlib::CKKS::Ciphertext& ctxt,
                   const KeySwitchingKey& keySwitchingKey, int numSlots, int blockSize,
                   FIDESlib::CKKS::Plaintext& weight, FIDESlib::CKKS::Plaintext& bias, bool bts, const int bStepAcc) {

    static FIDESlib::CKKS::Plaintext* GPUpt_norm = nullptr;
    Ciphertext sum(ctxt.cc);
    sum.copy(ctxt);

    FIDESlib::CKKS::Context& cc = ctxt.cc;

    if (GPUpt_norm == nullptr) {
        std::vector<double> mask(numSlots, 0.0);
        for (int i = 0; i < numSlots; i = i + 1) {
            mask[i] = 0.0078125;
        }

        auto raw_pt = FIDESlib::CKKS::GetRawPlainText(context, context->MakeCKKSPackedPlaintext(mask));
        GPUpt_norm = new FIDESlib::CKKS::Plaintext(cc, raw_pt);
    }

    // auto sum = rotsum_GPU(ctxt, blockSize, 1);  // sum
    Accumulate(sum, bStepAcc, 1, blockSize);

    sum.multPt(*GPUpt_norm);

    if (1) {
        Broadcast(sum, bStepAcc, 1, blockSize);
    } else {
        for (int j = 1; j < blockSize; j = j * 2) {
            FIDESlib::CKKS::Ciphertext rotated(cc);
            rotated.copy(sum);
            rotated.rotate(-j, cc.GetRotationKey(-j));
            sum.add(rotated);
        }
    }

    FIDESlib::CKKS::Ciphertext var(cc);

    var.copy(ctxt);
    var.sub(sum);  // ctxt - mean = var
    var.mult(var, var, keySwitchingKey);

    Ciphertext sum_var(ctxt.cc);
    sum_var.copy(var);
    // Accumulate(sum_var, bStepAcc, 1, blockSize);
    sum_var.multPt(*GPUpt_norm);

    if (1) {
        Broadcast(sum_var, bStepAcc, 1, blockSize);
    } else {
        for (int j = 1; j < blockSize; j = j * 2) {
            FIDESlib::CKKS::Ciphertext rotated(cc);
            rotated.copy(sum_var);
            rotated.rotate(-j, cc.GetRotationKey(-j));
            sum_var.add(rotated);
        }
    }
    // //std::cout << "# limbs LN sum_var " <<
    // evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm, numSlots, 10, 50000, bts);  // div_var

    // // sqrt
    // evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm_1_1, numSlots, 1, 1000, bts);
    // Bootstrap(sum_var, numSlots);

    // // 1 / x
    // evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm_1_2, numSlots, 1, 32, bts);
    // Bootstrap(sum_var, numSlots);
    //std::cout << sum_var.getLevel() << std::endl;
    evalFunction(sum_var, keySwitchingKey, cheb_coeff_inv_layernorm, numSlots, 1, 1000, bts);
    //std::cout << sum_var.getLevel() << std::endl;
    //if (bts)
    //    Bootstrap(sum_var, numSlots);

    // weight.multScalar(std::sqrt(128), false); // encoded with this scale

    ctxt.sub(sum);  // ctxt - mean

    ctxt.rescale();
    ctxt.dropToLevel(weight.c0.getLevel());  // TODO propper level adjustment
    ctxt.multPt(weight);                     // With prescale

    ctxt.mult(sum_var, keySwitchingKey);
    ctxt.rescale();

    ctxt.addPt(bias);  // With prescale
}

void EvalLayerNorm_Matrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                          std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& ctxt,
                          const KeySwitchingKey& keySwitchingKey,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& weight,
                          std::vector<std::vector<FIDESlib::CKKS::Plaintext>>& bias, int numSlots, int blockSize,
                          const int bStepAcc, bool bts) {

    for (size_t i = 0; i < ctxt.size(); i++) {
        for (size_t j = 0; j < ctxt[0].size(); j++) {
            EvalLayerNorm(context, ctxt[i][j], keySwitchingKey, numSlots, blockSize, weight[i][j], bias[i][j], bts,
                          bStepAcc);

            if (ctxt[i][j].NoiseLevel == 2)
                ctxt[i][j].rescale();

            // Plaintext aux(ctxt[i][j].cc);
            // aux.copy(weight[i][j]);
            // // aux.multScalar(std::sqrt(128), false);  // correcting the scale in layer norm
            // // aux.rescale();
            // if (ctxt[i][j].NoiseLevel == 2)
            //     ctxt[i][j].rescale();  //.dropToLevel(aux.c0.getLevel());
            // ctxt[i][j].dropToLevel(aux.c0.getLevel());
            // std::cout << "# limbs at LN: " << ctxt[i][j].getLevel() << " " << ctxt[i][j].NoiseLevel << std::endl;
            // ctxt[i][j].multPt(aux);
            // if (ctxt[i][j].NoiseLevel == 2)
            //     ctxt[i][j].rescale();  //.dropToLevel(aux.c0.getLevel());
            // ctxt[i][j].dropToLevel(bias[i][j].c0.getLevel());
            // ctxt[i][j].addPt(bias[i][j]);
            // std::cout << "# limbs at LN: " << ctxt[i][j].getLevel() << " " << ctxt[i][j].NoiseLevel << std::endl;
        }
    }
}

}  // namespace FIDESlib::CKKS
