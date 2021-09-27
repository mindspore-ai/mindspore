/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>

#include "runtime/device/kernel_info.h"
#include "runtime/device/cpu/kernel_select_cpu.h"

namespace mindspore {
namespace kernel {
namespace {
const std::set<std::string> same_op_name = {"Concat", "Pack", "Stack", "Split", "Transpose", "Unpack", "AddN"};
}  // namespace

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
  MS_EXCEPTION_IF_NULL(apply_kernel);
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

void CPUKernelFactory::SetKernelAttrs(const std::shared_ptr<kernel::OpInfo> op_info,
                                      std::vector<KernelAttr> *kernel_attrs) {
  MS_EXCEPTION_IF_NULL(kernel_attrs);
  MS_EXCEPTION_IF_NULL(op_info);
  auto inputs_ptr = op_info->inputs_ptr();
  auto outputs_ptr = op_info->outputs_ptr();
  if (inputs_ptr.empty()) {
    MS_LOG(EXCEPTION) << "op " << op_info->op_name() << " input size is zero.";
  }
  auto first_input_dtypes = inputs_ptr[0]->dtypes();
  auto input_formats = inputs_ptr[0]->formats();

  for (size_t i = 0; i < first_input_dtypes.size(); i++) {
    KernelAttr kernel_attr;
    (void)kernel_attr.AddInputAttr(kernel::DtypeToTypeId(first_input_dtypes[i]), input_formats[i]);
    for (size_t j = 1; j < inputs_ptr.size(); j++) {
      auto input_dtypes = inputs_ptr[j]->dtypes();
      input_formats = inputs_ptr[j]->formats();
      (void)kernel_attr.AddInputAttr(kernel::DtypeToTypeId(input_dtypes[i]), input_formats[i]);
    }
    for (size_t j = 0; j < outputs_ptr.size(); j++) {
      auto output_dtypes = outputs_ptr[j]->dtypes();
      auto output_formats = outputs_ptr[j]->formats();
      (void)kernel_attr.AddOutputAttr(kernel::DtypeToTypeId(output_dtypes[i]), output_formats[i]);
    }
    if (same_op_name.count(op_info->op_name()) != 0) {
      (void)kernel_attr.SetAllSameAttr(true);
    }
    (void)kernel_attrs->emplace_back(kernel_attr);
  }
}

void CPUKernelFactory::UpdateKernelAttrs(const std::string &kernel_name, const std::vector<KernelAttr> &kernel_attrs) {
  size_t attr_size = kernel_attrs.size();
  std::vector<std::pair<KernelAttr, CPUKernelCreator>> attr_creators(attr_size);
  auto iter = name_to_attr_creator_.find(kernel_name);
  if (iter == name_to_attr_creator_.end()) {
    MS_LOG(EXCEPTION) << "CPUKernelFactory has not registered operator: " << kernel_name;
  }

  if (attr_size <= iter->second.size()) {
    for (size_t i = 0; i < attr_size; i++) {
      auto creator = name_to_attr_creator_.find(kernel_name)->second[i].second;
      attr_creators[i] = std::make_pair(kernel_attrs[i], creator);
    }
  } else {
    MS_LOG(INFO) << "attr size is not equal creators size " << kernel_name << " attr_size = " << attr_size
                 << " creator_size = " << iter->second.size();
    auto single_creator = name_to_attr_creator_.find(kernel_name)->second[0].second;
    for (size_t i = 0; i < attr_size; i++) {
      attr_creators[i] = std::make_pair(kernel_attrs[i], single_creator);
    }
  }
  name_to_attr_creator_[kernel_name] = attr_creators;
}

std::pair<bool, size_t> CPUKernelFactory::CPUKernelAttrCheck(const std::string &kernel_name,
                                                             const KernelBuildInfo &kernel_info) {
  auto iter = name_to_attr_creator_.find(kernel_name);
  if (iter == name_to_attr_creator_.end()) {
    MS_LOG(INFO) << "Not registered CPU kernel: op[" << kernel_name << "]!";
    return std::make_pair(false, 0);
  }

  if (device::cpu::IsDynamicParamKernel(kernel_name)) {
    return std::make_pair(true, 0);
  }

  auto kernel_attrs = GetSupportedKernelAttrList(kernel_name);
  if (kernel_attrs[0].GetInputSize() == 0 && kernel_attrs[0].GetOutputSize() == 0) {
    auto op_info_ptr = mindspore::kernel::OpLib::FindOp(kernel_name, kernel::OpImplyType::kCPU);
    if (op_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Not find op[" << kernel_name << "] in cpu";
    }
    kernel_attrs.clear();
    SetKernelAttrs(op_info_ptr, &kernel_attrs);
    kernel::CPUKernelFactory::GetInstance().UpdateKernelAttrs(kernel_name, kernel_attrs);
  }
  for (size_t index = 0; index < kernel_attrs.size(); ++index) {
    if (CPUKernelSingleAttrCheck(kernel_attrs[index], kernel_info)) {
      return std::make_pair(true, index);
    }
  }
  return std::make_pair(false, 0);
}

bool CPUKernelFactory::CPUKernelSingleAttrCheck(const KernelAttr &kernel_attr,
                                                const KernelBuildInfo &kernel_info) const {
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
    MS_LOG(EXCEPTION) << "Not registered CPU kernel: op[" << kernel_name << "]!";
  }
  auto creators = iter->second;
  result.reserve(creators.size());
  for (size_t index = 0; index < creators.size(); ++index) {
    auto attr_creator = creators[index];
    (void)result.emplace_back(attr_creator.first);
  }
  return result;
}
}  // namespace kernel
}  // namespace mindspore
