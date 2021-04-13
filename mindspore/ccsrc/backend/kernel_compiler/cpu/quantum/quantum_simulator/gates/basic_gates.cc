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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/basic_gates.h"

#include <string>

namespace mindspore {
namespace mindquantum {
BasicGate::BasicGate(const std::string &name, bool is_parameter, const Indexes &obj_qubits, const Indexes &ctrl_qubits,
                     const ParameterResolver &paras)
    : name_(name), is_parameter_(is_parameter), obj_qubits_(obj_qubits), ctrl_qubits_(ctrl_qubits), paras_(paras) {}

Matrix BasicGate::GetMatrix(const ParameterResolver &paras_out) {
  Matrix gate_matrix_tmp;
  return gate_matrix_tmp;
}

Matrix BasicGate::GetDiffMatrix(const ParameterResolver &paras_out) {
  Matrix gate_matrix_tmp;
  return gate_matrix_tmp;
}

Matrix &BasicGate::GetBaseMatrix() { return gate_matrix_base_; }

const ParameterResolver &BasicGate::GetParameterResolver() const { return paras_; }

bool BasicGate::IsParameterGate() { return is_parameter_; }

Indexes BasicGate::GetObjQubits() { return obj_qubits_; }

Indexes BasicGate::GetCtrlQubits() { return ctrl_qubits_; }
}  // namespace mindquantum
}  // namespace mindspore
