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

#ifndef MINDQUANTUM_ENGINE_INTRINSIC_ONE_PARAGATE_H_
#define MINDQUANTUM_ENGINE_INTRINSIC_ONE_PARAGATE_H_
#include <string>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/parameter_gate.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
class IntrinsicOneParaGate : public ParameterGate {
  virtual Matrix GetIntrinsicMatrix(CalcType);
  virtual Matrix GetIntrinsicDiffMatrix(CalcType);

 public:
  IntrinsicOneParaGate();
  IntrinsicOneParaGate(const std::string &, const Indexes &, const Indexes &, const ParameterResolver &);
  CalcType LinearCombination(const ParameterResolver &, const ParameterResolver &);
  Matrix GetMatrix(const ParameterResolver &) override;
  Matrix GetDiffMatrix(const ParameterResolver &) override;
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_INTRINSIC_ONE_PARAGATE_H_
