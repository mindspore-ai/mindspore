/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/bisheng/custom_bisheng_kernel.h"

#include <dlfcn.h>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "abstract/utils.h"
#include "utils/file_utils.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
CustomBiShengKernel::~CustomBiShengKernel() {
  if (handle_ != nullptr) {
    dlclose(handle_);
  }

  attrs_.DestructKernelData();
}

bool CustomBiShengKernel::InitKernel(const AnfNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  const auto &exec_info = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "func_name");
  auto pos = exec_info.find(":");
  if (pos == std::string::npos) {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_ << "' on Ascend, user defined function path '" << exec_info
      << "' is illegal. Proper function path should follow the format of 'dir_path/file_name:func_name'";
  }
  auto path = exec_info.substr(0, pos);
  auto real_path = FileUtils::GetRealPath(path.c_str());
  if (!real_path.has_value()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, couldn't find the AOT binary file: " << path;
  }
  file_path_ = real_path.value();
  func_name_ = exec_info.substr(pos + 1);

  num_input_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
  auto input_type_list = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
  if (num_input_ != input_type_list.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, number of input types '" << input_type_list.size()
                      << "' doesn't match number of input shapes '" << num_input_ << "'";
  }

  for (size_t i = 0; i < num_input_; i++) {
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
    type_list_.emplace_back(TypeIdToString(input_type_list[i], true));
  }

  num_output_ = AnfAlgo::GetOutputTensorNum(kernel_node);
  auto output_type_list = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);
  if (num_output_ != output_type_list.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, number of outputs types '" << output_type_list.size()
                      << "' doesn't match number of output shapes '" << num_output_ << "'";
  }

  for (size_t i = 0; i < num_output_; i++) {
    auto out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
    type_list_.emplace_back(TypeIdToString(output_type_list[i], true));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  (void)std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                       [](auto &str) { return str.c_str(); });
  auto cnode = kernel_node->cast<CNodePtr>();
  attrs_.SetKernelPrim(common::AnfAlgo::GetCNodePrimitive(cnode));

  if (!handle_) {
    handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on Ascend, dlopen file '" << file_path_
                    << "' should be successful, but error occurs! Error message is: " << dlerror();
      return false;
    }
  }
  init_func_ = reinterpret_cast<std::add_pointer<int(int *, int64_t **, const char **, AotExtraDualABI *)>::type>(
    dlsym(handle_, (func_name_ + "Init").c_str()));
  if (init_func_ != nullptr) {
    // Init func exist in the custom aot file
    // Call this init func to set custom op attrs_
    int ret = 0;
    try {
      ret = init_func_(&ndims_[0], &shapes_[0], &type_pointer_list_[0], (&attrs_));
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on Ascend, operator failed when executing user defined file "
                    << file_path_ << "! "
                    << "Error message is " << e.what();
      return false;
    }

    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Return value from Ascend AOT kernel(" << file_path_ << ")'s function(" << func_name_
                        << ") is " << ret << ". "
                        << "Any return value not equal to 0 will be treated as user defined error code and we will "
                           "terminate execution. If termination is not your purpose, please set return value to 0.";
    }
  }
  InitSizeLists();
  return true;
}

void CustomBiShengKernel::InitSizeLists() {
  for (size_t i = 0; i < num_input_; i++) {
    size_t this_size =
      LongToSize(std::accumulate(shape_list_[i].begin(), shape_list_[i].end(), 1, std::multiplies<int64_t>()));
    this_size *= GetDtypeNbyte(type_list_[i]);
    input_size_list_.push_back(this_size);
  }
  for (size_t i = num_input_; i < (num_input_ + num_output_); i++) {
    size_t this_size =
      LongToSize(std::accumulate(shape_list_[i].begin(), shape_list_[i].end(), 1, std::multiplies<int64_t>()));

    this_size *= GetDtypeNbyte(type_list_[i]);
    output_size_list_.push_back(this_size);
  }
  workspace_size_list_.clear();
  workspace_size_list_ = attrs_.WorkSpace();
}

bool CustomBiShengKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> params;

  for (size_t i = 0; i < num_input_; i++) {
    params.push_back(GetDeviceAddress<void>(inputs, i));
  }
  for (size_t i = 0; i < num_output_; i++) {
    params.push_back(GetDeviceAddress<void>(outputs, i));
  }

  for (size_t i = 0; i < attrs_.WorkSpace().size(); i++) {
    params.push_back(GetDeviceAddress<void>(workspace, i));
  }

  if (!handle_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, the handle should be not nullptr since the file '"
                      << file_path_ << "' should be opened successfully in Init method.";
  }

  if (!aot_func_) {
    aot_func_ =
      reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
        dlsym(handle_, func_name_.c_str()));
    if (auto error_info = dlerror(); error_info != nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, error occurs when fetching function '" << func_name_
                        << "'. Error info: " << error_info;
    }
  }

  int nparam = SizeToInt(params.size());
  int ret = 0;
  try {
    if (nparam == 0) {
      ret = aot_func_(0, nullptr, nullptr, nullptr, nullptr, stream_ptr, nullptr);
    } else {
      ret = aot_func_(nparam, &params[0], &ndims_[0], &shapes_[0], &type_pointer_list_[0], stream_ptr,
                      reinterpret_cast<void *>(&attrs_));
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on Ascend, operator failed when executing user defined file "
                      << file_path_ << "! "
                      << "Error message is " << e.what();
  }

  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Return value from Ascend AOT kernel(" << file_path_ << ")'s function(" << func_name_
                      << ") is " << ret << ". "
                      << "Any return value not equal to 0 will be treated as user defined error code and we will "
                         "terminate execution. If termination is not your purpose, please set return value to 0.";
  }

  return true;
}

std::vector<TaskInfoPtr> CustomBiShengKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &,
                                                      const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  MS_LOG(ERROR) << "BiSheng OP does not support sink mode for now.";
  return {};
}
}  // namespace kernel
}  // namespace mindspore
