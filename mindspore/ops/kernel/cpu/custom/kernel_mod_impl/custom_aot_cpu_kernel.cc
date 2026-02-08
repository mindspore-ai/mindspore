/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/custom/kernel_mod_impl/custom_aot_cpu_kernel.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <functional>
#include "kernel/cpu/cpu_common.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
CustomAOTCpuKernelMod::~CustomAOTCpuKernelMod() {
#if !defined(_WIN32) && !defined(_WIN64)
  attrs_.DestructKernelData();

  if (handle_ != nullptr) {
    dlclose(handle_);
  }

#endif
}

void CustomAOTCpuKernelMod::SetKernelPath() {
  const auto &exec_info = GetValue<std::string>(primitive_->GetAttr("func_name"));

  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    if (primitive_->HasAttr("path_from_env") && GetValue<bool>(primitive_->GetAttr("path_from_env"))) {
      const char *path_in_env = common::EnvHelper::GetInstance()->GetEnv(path.c_str());
      if (path_in_env == nullptr) {
        MS_LOG(WARNING) << "For '" << kernel_name_ << "' on CPU, the attr path_from_env is set but the env var ["
                        << path << "] is empty. Use [" << path << "] as the path to the library instead.";
      } else {
        path = std::string(path_in_env);
      }
    }
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, couldn't find the AOT binary file: " << path;
    }
    file_path_ = real_path.value();
    func_name_ = exec_info.substr(pos + 1);

    constexpr auto kWhiteList = "MS_CUSTOM_AOT_WHITE_LIST";
    const char *value = common::EnvHelper::GetInstance()->GetEnv(kWhiteList);
    if (value == nullptr) {
      static bool print_cpu_warning_once = true;
      if (print_cpu_warning_once) {
        MS_LOG(INFO) << "For '" << kernel_name_ << "' on CPU, no white list is set and it might cause problems. "
                     << "Set the legal path of the file in MS_CUSTOM_AOT_WHITE_LIST.";
        print_cpu_warning_once = false;
      }
    } else {
      auto white_list = FileUtils::GetRealPath(value);
      if (!white_list.has_value()) {
        MS_LOG(EXCEPTION) << "Illegal white list path set in MS_CUSTOM_AOT_WHITE_LIST: " << std::string(value);
      }
      constexpr auto kKernelMeta = "akg_kernel_meta";
      if (file_path_.find(white_list.value()) == std::string::npos &&
          file_path_.find(kKernelMeta) == std::string::npos) {
        MS_LOG(EXCEPTION)
          << "For '" << kernel_name_
          << "' on CPU, the file is not place in the legal path file defined by MS_CUSTOM_AOT_WHITE_LIST: "
          << white_list.value() << ". The file path is: " << file_path_;
      }
    }
  } else {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_ << "' on CPU, user defined function path '" << exec_info
      << "' is illegal. Proper function path should follow the format of 'dir_path/file_name:func_name'";
  }
}

bool CustomAOTCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  kernel_name_ = primitive_->name();
  SetKernelPath();

  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    auto dtype = inputs[i]->dtype_id();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
    (void)type_list_.emplace_back(TypeIdToString(dtype, true));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    auto dtype = outputs[i]->dtype_id();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
    (void)type_list_.emplace_back(TypeIdToString(dtype, true));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  (void)std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                       [](auto &str) { return str.c_str(); });
  attrs_.SetKernelPrim(primitive_);

#if !defined(_WIN32) && !defined(_WIN64)
  if (!handle_) {
    handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!handle_) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on CPU, dlopen file '" << file_path_
                    << "' should be successful, but error occurs! Error message is: " << dlerror();
      return false;
    }
  }
  init_func_ = reinterpret_cast<std::add_pointer<int(int *, int64_t **, const char **, AotExtra *)>::type>(
    dlsym(handle_, (func_name_ + "Init").c_str()));
  if (init_func_ != nullptr) {
    // Init func exist in the custom aot file
    // Call this init func to set custom op attrs_
    int ret = 0;
    try {
      ret = init_func_(&ndims_[0], &shapes_[0], &type_pointer_list_[0], (&attrs_));
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on CPU, operator failed when executing user defined file "
                    << file_path_ << "! "
                    << "Error message is " << e.what();
      return false;
    }

    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Return value from CPU AOT kernel(" << file_path_ << ")'s function(" << func_name_ << ") is "
                        << ret << ". "
                        << "Any return value not equal to 0 will be treated as user defined error code and we will "
                           "terminate execution. If termination is not your purpose, please set return value to 0.";
    }
  }
#else
  MS_LOG(EXCEPTION) << "Custom AOT Operator doesn't support Windows currently";
#endif
  return true;
}

bool CustomAOTCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs) {
  std::vector<void *> params;

  for (size_t i = 0; i < inputs.size(); i++) {
    params.push_back(static_cast<void *>(inputs[i]->device_ptr()));
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    params.push_back(static_cast<void *>(outputs[i]->device_ptr()));
  }

  for (size_t i = 0; i < workspace.size(); i++) {
    params.push_back(static_cast<void *>(workspace[i]->device_ptr()));
  }

#if !defined(_WIN32) && !defined(_WIN64)

  if (!handle_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, dlopen file '" << file_path_
                      << "' must be successful, but error occurs! Error message is: " << dlerror();
  }

  if (!aot_func_) {
    aot_func_ =
      reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
        dlsym(handle_, func_name_.c_str()));
    if (auto error_info = dlerror(); error_info != nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, error occurs when fetching function '" << func_name_
                        << "'. Error info: " << error_info;
    }
  }

  int nparam = SizeToInt(params.size());
  int ret = 0;
  try {
    if (nparam == 0) {
      ret = aot_func_(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else {
      ret = aot_func_(nparam, &params[0], &ndims_[0], &shapes_[0], &type_pointer_list_[0], nullptr,
                      reinterpret_cast<void *>(&attrs_));
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on CPU, operator failed when executing user defined file "
                      << file_path_ << "! "
                      << "Error message is " << e.what();
  }

  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Return value from CPU AOT kernel(" << file_path_ << ")'s function(" << func_name_ << ") is "
                      << ret << ". "
                      << "Any return value not equal to 0 will be treated as user defined error code and we will "
                         "terminate execution. If termination is not your purpose, please set return value to 0.";
  }

#else
  MS_LOG(EXCEPTION) << "Custom AOT Operator doesn't support Windows currently";
#endif

  return true;
}

int CustomAOTCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  shapes_.clear();
  shape_list_.clear();
  ndims_.clear();

  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_shape = inputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(in_shape);
    ndims_.push_back(SizeToInt(in_shape.size()));
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto out_shape = outputs[i]->GetShapeVector();
    (void)shape_list_.emplace_back(out_shape);
    ndims_.push_back(SizeToInt(out_shape.size()));
  }

  (void)std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                       [](auto &v) { return &v[0]; });
  workspace_size_list_ = attrs_.WorkSpace();
  return static_cast<int>(KRET_OK);
}
}  // namespace kernel
}  // namespace mindspore
