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

#ifndef MINDQUANTUM_ENGINE_PARAMETER_RESOLVER_H_
#define MINDQUANTUM_ENGINE_PARAMETER_RESOLVER_H_
#include <map>
#include <string>
#include <set>
#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/utils.h"

namespace mindspore {
namespace mindquantum {

class ParameterResolver {
 public:
  ParameterResolver();
  ParameterResolver(const ParaType &, const ParaSetType &, const ParaSetType &);
  const ParaType &GetData() const;
  const ParaSetType &GetRequiresGradParameters() const;
  void SetData(const std::string &, const CalcType &);
  void InsertNoGrad(const std::string &);
  void InsertRequiresGrad(const std::string &);

 private:
  ParaType data_;
  ParaSetType no_grad_parameters_;
  ParaSetType requires_grad_parameters_;
};
}  // namespace mindquantum
}  // namespace mindspore
#endif  // MINDQUANTUM_ENGINE_PARAMETER_RESOLVER_H_
