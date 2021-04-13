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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/non_parameter_gate.h"

#include <string>

namespace mindspore {
namespace mindquantum {
NoneParameterGate::NoneParameterGate(const std::string &name, const Matrix &gate_matrix, const Indexes &obj_qubits,
                                     const Indexes &ctrl_qubits)
    : BasicGate(name, false, obj_qubits, ctrl_qubits), gate_matrix_(gate_matrix) {}

Matrix &NoneParameterGate::GetBaseMatrix() { return gate_matrix_; }
}  // namespace mindquantum
}  // namespace mindspore
