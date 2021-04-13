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

#ifndef MINDQUANTUM_ENGINE_BASIC_GATES_H_
#define MINDQUANTUM_ENGINE_BASIC_GATES_H_
#include <string>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/parameter_resolver.h"
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {
class BasicGate {
 private:
  std::string name_;
  bool is_parameter_;
  Matrix gate_matrix_base_;
  Indexes obj_qubits_;
  Indexes ctrl_qubits_;
  ParameterResolver paras_;

 public:
  BasicGate();
  BasicGate(const std::string &, bool, const Indexes &, const Indexes &,
            const ParameterResolver &paras = ParameterResolver());
  virtual Matrix GetMatrix(const ParameterResolver &);
  virtual Matrix GetDiffMatrix(const ParameterResolver &);
  virtual Matrix &GetBaseMatrix();
  const ParameterResolver &GetParameterResolver() const;
  bool IsParameterGate();
  Indexes GetObjQubits();
  Indexes GetCtrlQubits();
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_BASIC_GATES_H_
