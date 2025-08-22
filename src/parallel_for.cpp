//
// Created by carlosad on 27/03/25.
//
#include "parallel_for.hpp"
#include <omp.h>

void FIDESlib::parallel_for(int init, int end, int increment, const std::function<void(int)>& f) {
#pragma omp parallel for
    for (int i = init; i < end; i += increment) {
        f(i);
    }
}
