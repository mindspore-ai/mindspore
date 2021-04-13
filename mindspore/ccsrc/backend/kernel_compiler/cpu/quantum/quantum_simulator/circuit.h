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

#ifndef MINDQUANTUM_ENGINE_CCIRCUIT_H_
#define MINDQUANTUM_ENGINE_CCIRCUIT_H_
#include <vector>
#include <string>
#include <memory>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/non_parameter_gate.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/gates.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
using GateBlock = std::vector<std::shared_ptr<BasicGate>>;
using GateBlocks = std::vector<GateBlock>;

class BasicCircuit {
 private:
  GateBlocks gate_blocks_;

 public:
  BasicCircuit();
  void AppendBlock();
  void AppendNoneParameterGate(const std::string &, Matrix, Indexes, Indexes);
  void AppendParameterGate(const std::string &, Indexes, Indexes, const ParameterResolver &);
  const GateBlocks &GetGateBlocks() const;
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_CCIRCUIT_H_
