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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/parameter_gate.h"

#include <string>

namespace mindspore {
namespace mindquantum {
ParameterGate::ParameterGate(const std::string &name, const Indexes &obj_qubits, const Indexes &ctrl_qubits,
                             const ParameterResolver &paras)
    : BasicGate(name, true, obj_qubits, ctrl_qubits, paras) {}
}  // namespace mindquantum
}  // namespace mindspore
