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

#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/parameter_resolver.h"

namespace mindspore {
namespace mindquantum {
ParameterResolver::ParameterResolver() {}

ParameterResolver::ParameterResolver(const ParaType &data, const ParaSetType &no_grad_parameters,
                                     const ParaSetType &requires_grad_parameters)
    : data_(data), no_grad_parameters_(no_grad_parameters), requires_grad_parameters_(requires_grad_parameters) {}

const ParaType &ParameterResolver::GetData() const { return data_; }
const ParaSetType &ParameterResolver::GetRequiresGradParameters() const { return requires_grad_parameters_; }
void ParameterResolver::SetData(const std::string &name, const CalcType &value) { data_[name] = value; }
void ParameterResolver::InsertNoGrad(const std::string &name) { no_grad_parameters_.insert(name); }
void ParameterResolver::InsertRequiresGrad(const std::string &name) { requires_grad_parameters_.insert(name); }
}  // namespace mindquantum
}  // namespace mindspore
