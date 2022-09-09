/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUSTOM_CUSTOM_AOT_GPU_KERNEL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_CUSTOM_CUSTOM_AOT_GPU_KERNEL_H

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "kernel/common_utils.h"
#include "utils/custom_aot_extra.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
class CustomAOTGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  CustomAOTGpuKernelMod() : num_input_(0), num_output_(0), handle_(nullptr), aot_func_(nullptr) {}
  ~CustomAOTGpuKernelMod() override {
    if (handle_ != nullptr) {
#ifdef _MSC_VER
      FreeLibrary(handle_);
#else
      dlclose(handle_);
#endif
    }

    attrs_.DestructKernelData();
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
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
#ifdef _MSC_VER
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, dlopen file '" << file_path_
                    << "' should be successful, but error occurs! ";
#else
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, dlopen file '" << file_path_
                    << "' should be successful, but error occurs! Error message is: " << dlerror();
#endif
      return false;
    }

    if (!aot_func_) {
#ifdef _MSC_VER
      aot_func_ =
        reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
          GetProcAddress(handle_, func_name_.c_str()));
      if (aot_func_ == nullptr) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, error occurs when fetching function '" << func_name_;
        return false;
      }
#else
      aot_func_ =
        reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
          dlsym(handle_, func_name_.c_str()));
      if (auto error_info = dlerror(); error_info != nullptr) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, error occurs when fetching function '" << func_name_
                      << "'. Error info: " << error_info;
        return false;
      }
#endif
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
      MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, operator failed when executing user defined file "
                    << file_path_ << "! "
                    << "Error message is " << e.what();
      return false;
    }

    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Return value from GPU AOT kernel(" << file_path_ << ")'s function(" << func_name_ << ") is "
                        << ret << ". "
                        << "Any return value not equal to 0 will be treated as user defined error code and we will "
                           "terminate execution. If termination is not your purpose, please set return value to 0.";
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    const auto &exec_info = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "func_name");
    kernel_node_ = kernel_node;
    if (auto pos = exec_info.find(":"); pos != std::string::npos) {
      auto path = exec_info.substr(0, pos);
      auto real_path = FileUtils::GetRealPath(path.c_str());
      if (!real_path.has_value()) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on GPU, couldn't find the AOT binary file: " << path;
      }
      file_path_ = real_path.value();
      func_name_ = exec_info.substr(pos + 1);
    } else {
      MS_LOG(EXCEPTION)
        << "For '" << kernel_name_ << "' on GPU, user defined function path '" << exec_info
        << "' is illegal. Proper function path should follow the format of 'dir_path/file_name:func_name'";
    }

    num_input_ = common::AnfAlgo::GetInputTensorNum(kernel_node);
    auto input_type_list = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
    if (num_input_ != input_type_list.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on GPU, number of input types '" << input_type_list.size()
                        << "' doesn't match number of input shapes '" << num_input_ << "'";
    }

    for (size_t i = 0; i < num_input_; i++) {
      auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      type_list_.emplace_back(TypeIdToString(input_type_list[i], true));
      ndims_.push_back(SizeToInt(in_shape.size()));
      shape_list_.emplace_back(in_shape);
    }

    num_output_ = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    auto output_type_list = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);

    if (num_output_ != output_type_list.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' on GPU, number of outputs types '" << output_type_list.size()
                        << "' doesn't match number of output shapes '" << num_output_ << "'";
    }

    for (size_t i = 0; i < num_output_; i++) {
      auto out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, i);
      shape_list_.emplace_back(out_shape);
      ndims_.push_back(SizeToInt(out_shape.size()));
      type_list_.emplace_back(TypeIdToString(output_type_list[i], true));
    }

    std::transform(std::begin(shape_list_), std::end(shape_list_), std::back_inserter(shapes_),
                   [](auto &v) { return &v[0]; });
    std::transform(std::begin(type_list_), std::end(type_list_), std::back_inserter(type_pointer_list_),
                   [](auto &str) { return str.c_str(); });

    attrs_.SetKernelNode(kernel_node);

    if (!handle_) {
#ifdef _MSC_VER
      handle_ = LoadLibrary(file_path_.c_str());
      if (!handle_) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, dlopen file '" << file_path_
                      << "' should be successful, but error occurs! ";
        return false;
      }
#else
      handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (!handle_) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, dlopen file '" << file_path_
                      << "' should be successful, but error occurs! Error message is: " << dlerror();
        return false;
      }
#endif
    }
#ifdef _MSC_VER
    init_func_ = reinterpret_cast<std::add_pointer<int(int *, int64_t **, const char **, AotExtra *)>::type>(
      GetProcAddress(handle_, (func_name_ + "Init").c_str()));
#else
    init_func_ = reinterpret_cast<std::add_pointer<int(int *, int64_t **, const char **, AotExtra *)>::type>(
      dlsym(handle_, (func_name_ + "Init").c_str()));
#endif
    if (init_func_ != nullptr) {
      // Init func exist in the custom aot file
      // Call this init func to set custom op attrs
      int ret = 0;
      try {
        ret = init_func_(&ndims_[0], &shapes_[0], &type_pointer_list_[0], (&attrs_));
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "' on GPU, operator failed when executing user defined file "
                      << file_path_ << "! "
                      << "Error message is " << e.what();
        return false;
      }

      if (ret != 0) {
        MS_LOG(EXCEPTION) << "Return value from GPU AOT kernel(" << file_path_ << ")'s function(" << func_name_
                          << ") is " << ret << ". "
                          << "Any return value not equal to 0 will be treated as user defined error code and we will "
                             "terminate execution. If termination is not your purpose, please set return value to 0.";
      }
    }

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    for (size_t i = 0; i < num_input_; i++) {
      size_t this_size = LongToSizeClipNeg(
        std::accumulate(shape_list_[i].begin(), shape_list_[i].end(), int64_t(1), std::multiplies<int64_t>()));
      this_size *= GetDtypeNbyte(type_list_[i]);
      input_size_list_.push_back(this_size);
    }
    for (size_t i = num_input_; i < (num_input_ + num_output_); i++) {
      size_t this_size = LongToSizeClipNeg(
        std::accumulate(shape_list_[i].begin(), shape_list_[i].end(), int64_t(1), std::multiplies<int64_t>()));

      this_size *= GetDtypeNbyte(type_list_[i]);
      output_size_list_.push_back(this_size);
    }

    workspace_size_list_.clear();
    workspace_size_list_ = attrs_.WorkSpace();
  }

 private:
  size_t num_input_;
  size_t num_output_;
  std::string file_path_;
  std::string func_name_;
#ifdef _MSC_VER
  HMODULE handle_{nullptr};
#else
  void *handle_{nullptr};
#endif
  int (*init_func_)(int *, int64_t **, const char **, AotExtra *);
  int (*aot_func_)(int, void **, int *, int64_t **, const char **, void *, void *);

  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<int> ndims_;
  std::vector<std::string> type_list_;

  std::vector<int64_t *> shapes_;
  std::vector<const char *> type_pointer_list_;

  AotExtraImpl attrs_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUSTOM_CUSTOM_AOT_GPU_KERNEL_H
