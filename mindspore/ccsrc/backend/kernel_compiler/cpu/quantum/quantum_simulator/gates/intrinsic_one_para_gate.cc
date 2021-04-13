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
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/gates/intrinsic_one_para_gate.h"

#include <string>

namespace mindspore {
namespace mindquantum {
Matrix IntrinsicOneParaGate::GetIntrinsicMatrix(CalcType theta) {
  Matrix gate_matrix_tmp;
  return gate_matrix_tmp;
}

Matrix IntrinsicOneParaGate::GetIntrinsicDiffMatrix(CalcType theta) {
  Matrix gate_matrix_tmp;
  return gate_matrix_tmp;
}

IntrinsicOneParaGate::IntrinsicOneParaGate(const std::string &name, const Indexes &obj_qubits,
                                           const Indexes &ctrl_qubits, const ParameterResolver &paras)
    : ParameterGate(name, obj_qubits, ctrl_qubits, paras) {}

CalcType IntrinsicOneParaGate::LinearCombination(const ParameterResolver &paras_in,
                                                 const ParameterResolver &paras_out) {
  CalcType result = 0;
  auto &paras_in_data = paras_in.GetData();
  auto &paras_out_data = paras_out.GetData();
  for (ParaType::const_iterator i = paras_in_data.begin(); i != paras_in_data.end(); ++i) {
    result = result + paras_out_data.at(i->first) * (i->second);
  }
  return result;
}
Matrix IntrinsicOneParaGate::GetMatrix(const ParameterResolver &paras_out) {
  return GetIntrinsicMatrix(LinearCombination(GetParameterResolver(), paras_out));
}
Matrix IntrinsicOneParaGate::GetDiffMatrix(const ParameterResolver &paras_out) {
  return GetIntrinsicDiffMatrix(LinearCombination(GetParameterResolver(), paras_out));
}
}  // namespace mindquantum
}  // namespace mindspore
