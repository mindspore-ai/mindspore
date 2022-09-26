/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/custom/custom_julia_cpu_kernel.h"

#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "plugin/device/cpu/kernel/custom/julia_api.h"
#include "utils/file_utils.h"
#include "mindspore/core/ops/custom.h"

namespace mindspore {
namespace kernel {
bool CustomJULIACpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Custom>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  const auto &exec_info = GetValue<std::string>(kernel_ptr->GetAttr("func_name"));
  auto pos1 = exec_info.find(":");
  auto pos2 = exec_info.rfind(":");
  if (pos1 == std::string::npos || pos2 == std::string::npos || pos1 == pos2) {
    MS_LOG(EXCEPTION) << "Wrong execute info: " << exec_info << ", it must be file:module:func";
  }
  auto path = exec_info.substr(0, pos1);
  auto real_path = FileUtils::GetRealPath(path.c_str());
  if (!real_path.has_value()) {
    MS_LOG(EXCEPTION) << "Invalid file path, " << path << " does not exist.";
  }
  file_path_ = real_path.value();
  module_name_ = exec_info.substr(pos1 + 1, (pos2 - pos1) - 1);
  func_name_ = exec_info.substr(pos2 + 1);

  for (size_t i = 0; i < inputs.size(); i++) {
    auto dtype = inputs[i]->GetDtype();
    auto in_shape = inputs[i]->GetShapeVector();
    ndims_.push_back(in_shape.size());
    shape_list_.push_back(in_shape);
    type_list_.push_back(TypeIdToString(dtype, true));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto dtype = outputs[i]->GetDtype();
    auto out_shape = outputs[i]->GetShapeVector();
    ndims_.push_back(out_shape.size());
    shape_list_.push_back(out_shape);
    type_list_.push_back(TypeIdToString(dtype, true));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  (void)std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                       [](auto &str) { return str.c_str(); });
  return true;
}

bool CustomJULIACpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  std::vector<void *> params;
  for (size_t i = 0; i < inputs.size(); i++) {
    params.push_back(reinterpret_cast<void *>(inputs[i]->addr));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    params.push_back(reinterpret_cast<void *>(outputs[i]->addr));
  }
  size_t nparam = params.size();
  JuliaAPI *julia = JuliaAPI::GetInstance();
  if (!julia->Init()) {
    MS_LOG(EXCEPTION) << "Julia kernel[" << file_path_ << ":" << module_name_ << ":" << func_name_ << "] init failed.";
  }
  if (!julia->Run(file_path_, module_name_, func_name_, nparam, params, ndims_, shapes_, type_pointer_list_)) {
    MS_LOG(EXCEPTION) << "Julia kernel[" << file_path_ << ":" << module_name_ << ":" << func_name_ << "] run failed.";
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
