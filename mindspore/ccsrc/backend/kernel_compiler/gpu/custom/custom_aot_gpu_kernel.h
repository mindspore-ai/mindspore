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

#include <dlfcn.h>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace kernel {
class CustomAOTGpuKernelMod : public NativeGpuKernelMod {
 public:
  CustomAOTGpuKernelMod() : num_input_(0), num_output_(0), handle_(nullptr), aot_func_(nullptr) {}
  ~CustomAOTGpuKernelMod() override {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    std::vector<void *> params;

    for (size_t i = 0; i < num_input_; i++) {
      params.push_back(GetDeviceAddress<void>(inputs, i));
    }
    for (size_t i = 0; i < num_output_; i++) {
      params.push_back(GetDeviceAddress<void>(outputs, i));
    }

    if (!handle_) {
      handle_ = dlopen(file_path_.c_str(), RTLD_LAZY | RTLD_LOCAL);
      if (!handle_) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', open should be successful, but error, " << dlerror();
        return false;
      }
    }

    if (!aot_func_) {
      aot_func_ =
        reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
          dlsym(handle_, func_name_.c_str()));
      if (auto error_info = dlerror(); error_info != nullptr) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', error info: " << error_info;
        return false;
      }
    }

    int nparam = SizeToInt(params.size());
    int ret = 0;
    try {
      if (nparam == 0) {
        ret = aot_func_(0, nullptr, nullptr, nullptr, nullptr, stream_ptr, nullptr);
      } else {
        ret = aot_func_(nparam, &params[0], &ndims_[0], &shapes_[0], &type_pointer_list_[0], stream_ptr, nullptr);
      }
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', operator failed when running user defined file " << file_path_
                    << "! "
                    << "Error message is " << e.what();
      return false;
    }

    switch (ret) {
      case 0:
        break;
      case 1:
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of parameters passed to AOT kernel is " << nparam
                      << ", inconsistent with what the user wants";
        return false;
      case 2:
        MS_LOG(ERROR) << "For '" << kernel_name_
                      << "', type of parameters passed to AOT kernel is inconsistent with what the user wants";
        return false;
      default:
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', error occurred when running AOT kernel, "
                      << "error id is " << ret;
        return false;
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    const auto &exec_info = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "func_name");
    if (auto pos = exec_info.find(":"); pos != std::string::npos) {
      auto path = exec_info.substr(0, pos);
      auto real_path = FileUtils::GetRealPath(path.c_str());
      if (!real_path.has_value()) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the file path should be exist, but got " << path;
      }
      file_path_ = real_path.value();
      func_name_ = exec_info.substr(pos + 1);
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', Wrong execute info:" << exec_info;
    }

    num_input_ = AnfAlgo::GetInputTensorNum(kernel_node);
    auto input_type_list = AnfAlgo::GetAllInputDeviceTypes(kernel_node);
    if (num_input_ != input_type_list.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << input_type_list.size()
                        << ", but got " << num_input_;
    }

    for (size_t i = 0; i < num_input_; i++) {
      auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i);
      std::vector<int64_t> in_shape_tmp;
      std::for_each(in_shape.begin(), in_shape.end(),
                    [&in_shape_tmp](size_t c) { in_shape_tmp.push_back(SizeToLong(c)); });
      type_list_.emplace_back(TypeIdToString(input_type_list[i], true));
      ndims_.push_back(SizeToInt(in_shape_tmp.size()));
      shape_list_.emplace_back(in_shape_tmp);
    }

    num_output_ = AnfAlgo::GetOutputTensorNum(kernel_node);
    auto output_type_list = AnfAlgo::GetAllOutputDeviceTypes(kernel_node);

    if (num_output_ != output_type_list.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be " << output_type_list.size()
                        << ", but got " << num_output_;
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

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
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
  }

 private:
  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<int> ndims_;
  std::vector<std::string> type_list_;

  std::vector<int64_t *> shapes_;
  std::vector<const char *> type_pointer_list_;

  size_t num_input_;
  size_t num_output_;
  std::string file_path_;
  std::string func_name_;
  void *handle_;
  int (*aot_func_)(int, void **, int *, int64_t **, const char **, void *, void *);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUSTOM_CUSTOM_AOT_GPU_KERNEL_H
