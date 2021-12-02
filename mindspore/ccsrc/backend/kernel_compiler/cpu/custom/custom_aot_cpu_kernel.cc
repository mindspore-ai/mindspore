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
#include "backend/kernel_compiler/cpu/custom/custom_aot_cpu_kernel.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "abstract/utils.h"
#include "runtime/device/cpu/cpu_common.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
CustomAOTCpuKernel::~CustomAOTCpuKernel() {
#if !defined(_WIN32) && !defined(_WIN64)
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
#endif
}

void CustomAOTCpuKernel::InitKernel(const CNodePtr &kernel_node) {
  const auto &exec_info = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "func_name");
  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "Invalid file path, " << path << " does not exist.";
    }
    file_path_ = real_path.value();
    func_name_ = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "Wrong execute info:" << exec_info;
  }

  num_input_ = AnfAlgo::GetInputTensorNum(kernel_node);
  auto input_type_list = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
  if (num_input_ != input_type_list.size()) {
    MS_LOG(EXCEPTION) << "Input shapes'size is " << num_input_ << ", while input types' size is "
                      << input_type_list.size();
  }

  for (size_t i = 0; i < num_input_; i++) {
    std::vector<size_t> in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
    std::vector<int64_t> in_shape_tmp;
    std::for_each(in_shape.begin(), in_shape.end(),
                  [&in_shape_tmp](size_t c) { in_shape_tmp.push_back(SizeToLong(c)); });
    shape_list_.emplace_back(in_shape_tmp);
    ndims_.push_back(SizeToInt(in_shape_tmp.size()));
    type_list_.emplace_back(TypeIdToString(input_type_list[i], true));
  }

  num_output_ = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto output_type_list = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);
  if (num_output_ != output_type_list.size()) {
    MS_LOG(EXCEPTION) << "Output shapes'size is " << num_output_ << ", while output types' size is "
                      << output_type_list.size();
  }

  for (size_t i = 0; i < num_output_; i++) {
    std::vector<size_t> out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
    std::vector<int64_t> out_shape_tmp;
    std::for_each(out_shape.begin(), out_shape.end(),
                  [&out_shape_tmp](size_t c) { out_shape_tmp.push_back(SizeToLong(c)); });
    shape_list_.emplace_back(out_shape_tmp);
    ndims_.push_back(SizeToInt(out_shape_tmp.size()));
    type_list_.emplace_back(TypeIdToString(output_type_list[i], true));
  }

  std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                 [](auto &v) { return &v[0]; });
  std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                 [](auto &str) { return str.c_str(); });
}

bool CustomAOTCpuKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  std::vector<void *> params;

  for (size_t i = 0; i < num_input_; i++) {
    params.push_back(GetDeviceAddress<void>(inputs, i));
  }
  for (size_t i = 0; i < num_output_; i++) {
    params.push_back(GetDeviceAddress<void>(outputs, i));
  }

#if !defined(_WIN32) && !defined(_WIN64)
  if (!handle_) {
    handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      MS_LOG(EXCEPTION) << "Open Error: " << dlerror();
    }
  }

  if (!aot_func_) {
    aot_func_ =
      reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
        dlsym(handle_, func_name_.c_str()));
    if (auto error_info = dlerror(); error_info != nullptr) {
      MS_LOG(EXCEPTION) << error_info;
    }
  }

  int nparam = SizeToInt(params.size());
  int ret = 0;
  try {
    if (nparam == 0) {
      ret = aot_func_(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      ret = aot_func_(nparam, &params[0], &ndims_[0], &shapes_[0], &type_pointer_list_[0], nullptr, nullptr);
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "CustomAOT operator failed when running user defined file " << file_path_ << "! "
                      << "Error message is " << e.what();
  }

  switch (ret) {
    case 0:
      break;
    case 1:
      MS_LOG(EXCEPTION) << "Number of parameters passed to AOT kernel is  " << nparam
                        << ", inconsistent with what the user wants";
    case 2:
      MS_LOG(EXCEPTION) << "Type of parameters passed to AOT kernel is inconsistent with what the user wants";
    default:
      MS_LOG(EXCEPTION) << "Error occurred when running AOT kernel, "
                        << "error id is " << ret;
  }

#else
  MS_LOG(EXCEPTION) << "Custom AOT Operator doesn't support Windows currently";
#endif

  return true;
}
}  // namespace kernel
}  // namespace mindspore
