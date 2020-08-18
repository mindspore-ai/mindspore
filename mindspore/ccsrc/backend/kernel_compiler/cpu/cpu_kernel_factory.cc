/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

#include <memory>
#include <string>

#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace kernel {
CPUKernelFactory &CPUKernelFactory::GetInstance() {
  static CPUKernelFactory instance;
  return instance;
}

void CPUKernelFactory::Register(const std::string &kernel_name, const KernelAttr &kernel_attr,
                                CPUKernelCreator &&kernel_creator) {
  (void)name_to_attr_creator_[kernel_name].emplace_back(kernel_attr, kernel_creator);
#if !defined(_WIN32) && !defined(_WIN64)
  MS_LOG(DEBUG) << "CPUKernelFactory register operator: " << kernel_name;
#endif
}

std::shared_ptr<CPUKernel> CPUKernelFactory::Create(const std::string &kernel_name, const CNodePtr &apply_kernel) {
  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const KernelBuildInfo *kernel_build_Info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_Info);
  std::pair<bool, size_t> ret_pair = CPUKernelAttrCheck(kernel_name, *kernel_build_Info);
  if (ret_pair.first) {
    return (name_to_attr_creator_.find(kernel_name)->second)[ret_pair.second].second();
  }
  return nullptr;
}

std::pair<bool, size_t> CPUKernelFactory::CPUKernelAttrCheck(const std::string &kernel_name,
                                                             const KernelBuildInfo &kernel_info) {
  auto iter = name_to_attr_creator_.find(kernel_name);
  if (iter == name_to_attr_creator_.end()) {
    MS_LOG(INFO) << "Not registered CPU kernel: op[" << kernel_name << "]!";
    return std::make_pair(false, 0);
  }
  auto creators = iter->second;
  for (size_t index = 0; index < creators.size(); ++index) {
    auto attr_creator = creators[index];
    if (CPUKernelSingleAttrCheck(attr_creator.first, kernel_info)) {
      return std::make_pair(true, index);
    }
  }
  return std::make_pair(false, 0);
}

bool CPUKernelFactory::CPUKernelSingleAttrCheck(const KernelAttr &kernel_attr, const KernelBuildInfo &kernel_info) {
  for (size_t i = 0; i < kernel_info.GetInputNum(); ++i) {
    auto dtype = kernel_attr.GetAllSame() ? kernel_attr.GetInputAttr(0).first : kernel_attr.GetInputAttr(i).first;
    if (kernel_info.GetInputDeviceType(i) != dtype) {
      MS_LOG(DEBUG) << "input index:" << i << ", kernel info type:" << kernel_info.GetInputDeviceType(i)
                    << ", register type:" << dtype;
      return false;
    }
  }
  for (size_t i = 0; i < kernel_info.GetOutputNum(); ++i) {
    auto dtype = kernel_attr.GetAllSame() ? kernel_attr.GetOutputAttr(0).first : kernel_attr.GetOutputAttr(i).first;
    if (kernel_info.GetOutputDeviceType(i) != dtype) {
      MS_LOG(DEBUG) << "output index:" << i << ", kernel info type:" << kernel_info.GetOutputDeviceType(i)
                    << ", register type:" << dtype;
      return false;
    }
  }
  return true;
}

std::vector<KernelAttr> CPUKernelFactory::GetSupportedKernelAttrList(const std::string &kernel_name) {
  std::vector<KernelAttr> result;
  auto iter = name_to_attr_creator_.find(kernel_name);
  if (iter == name_to_attr_creator_.end()) {
    MS_LOG(WARNING) << "Not registered CPU kernel: op[" << kernel_name << "]!";
    return result;
  }
  auto creators = iter->second;
  for (size_t index = 0; index < creators.size(); ++index) {
    auto attr_creator = creators[index];
    result.push_back(attr_creator.first);
  }
  return result;
}
}  // namespace kernel
}  // namespace mindspore
