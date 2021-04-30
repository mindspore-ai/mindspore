/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
ComplexType ComplexInnerProduct(const Simulator::StateVector &v1, const Simulator::StateVector &v2, unsigned len) {
  CalcType real_part = 0;
  CalcType imag_part = 0;
  auto size = len / 2;
#pragma omp parallel for reduction(+ : real_part, imag_part)
  for (Index i = 0; i < size; i++) {
    real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
    imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
  }

  ComplexType result = {real_part, imag_part};
  return result;
}

ComplexType ComplexInnerProductWithControl(const Simulator::StateVector &v1, const Simulator::StateVector &v2,
                                           Index len, std::size_t ctrlmask) {
  CalcType real_part = 0;
  CalcType imag_part = 0;
  auto size = len / 2;
#pragma omp parallel for reduction(+ : real_part, imag_part)
  for (std::size_t i = 0; i < size; i++) {
    if ((i & ctrlmask) == ctrlmask) {
      real_part += v1[2 * i] * v2[2 * i] + v1[2 * i + 1] * v2[2 * i + 1];
      imag_part += v1[2 * i] * v2[2 * i + 1] - v1[2 * i + 1] * v2[2 * i];
    }
  }
  ComplexType result = {real_part, imag_part};
  return result;
}
}  // namespace mindquantum
}  // namespace mindspore
