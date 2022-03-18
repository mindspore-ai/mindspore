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

namespace mindspore {
namespace kernel {
void CustomJULIACpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  const auto &exec_info = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "func_name");
  auto pos1 = exec_info.find(":");
  auto pos2 = exec_info.rfind(":");
  if (pos1 == std::string::npos || pos2 == std::string::npos || pos1 == pos2) {
    MS_LOG(EXCEPTION) << "Wrong execute info: " << exec_info << ", it should be file:module:func";
  }
  auto path = exec_info.substr(0, pos1);
  auto real_path = FileUtils::GetRealPath(path.c_str());
  if (!real_path.has_value()) {
    MS_LOG(EXCEPTION) << "Invalid file path, " << path << " does not exist.";
  }
  file_path_ = real_path.value();
  module_name_ = exec_info.substr(pos1 + 1, (pos2 - pos1) - 1);
  func_name_ = exec_info.substr(pos2 + 1);

  num_input_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  auto input_type_list = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
  if (num_input_ != input_type_list.size()) {
    MS_LOG(EXCEPTION) << "Kernel[" << exec_info << "]'s input shapes'size is " << num_input_
                      << "is different from input types' size which is " << input_type_list.size();
  }

  for (size_t i = 0; i < num_input_; i++) {
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
    std::vector<int64_t> in_shape_tmp;
    (void)std::for_each(in_shape.begin(), in_shape.end(),
                        [&in_shape_tmp](size_t c) { in_shape_tmp.push_back(SizeToLong(c)); });
    ndims_.push_back(in_shape_tmp.size());
    shape_list_.push_back(in_shape_tmp);
    type_list_.push_back(TypeIdToString(input_type_list[i], true));
  }

  num_output_ = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  auto output_type_list = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);
  if (num_output_ != output_type_list.size()) {
    MS_LOG(EXCEPTION) << "Kernel[" << exec_info << "]'s output shapes'size is " << num_input_
                      << "is different from output types' size which is " << input_type_list.size();
  }

  for (size_t i = 0; i < num_output_; i++) {
    std::vector<size_t> out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
    std::vector<int64_t> out_shape_tmp;
    (void)std::for_each(out_shape.begin(), out_shape.end(),
                        [&out_shape_tmp](size_t c) { out_shape_tmp.push_back(SizeToLong(c)); });
    ndims_.push_back(out_shape_tmp.size());
    shape_list_.push_back(out_shape_tmp);
    type_list_.push_back(TypeIdToString(output_type_list[i], true));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  (void)std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                       [](auto &str) { return str.c_str(); });
}

bool CustomJULIACpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  std::vector<void *> params;
  for (size_t i = 0; i < num_input_; i++) {
    params.push_back(GetDeviceAddress<void>(inputs, i));
  }
  for (size_t i = 0; i < num_output_; i++) {
    params.push_back(GetDeviceAddress<void>(outputs, i));
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
