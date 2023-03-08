/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

#include "utils/ms_utils.h"
#include "include/backend/kernel_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
NativeGpuKernelModFactory &NativeGpuKernelModFactory::GetInstance() {
  static NativeGpuKernelModFactory instance;
  return instance;
}

void NativeGpuKernelModFactory::Register(const std::string &kernel_name, const KernelAttr &kernel_attr,
                                         NativeGpuKernelModCreater &&creator) {
  map_kernel_name_to_creater_[kernel_name].emplace_back(kernel_attr, creator);
}

bool NativeGpuKernelModFactory::CheckIOParam(const std::string &kernel_name, const KernelBuildInfo *kernel_info,
                                             std::vector<std::pair<KernelAttr, NativeGpuKernelModCreater>> *iter_second,
                                             size_t attr_index) {
  if (kernel_info->GetInputNum() != iter_second->at(attr_index).first.GetInputSize()) {
    if (!iter_second->at(attr_index).first.GetAllSame()) {
      return false;
    }
  }
  if (kernel_info->GetOutputNum() != iter_second->at(attr_index).first.GetOutputSize()) {
    if (!iter_second->at(attr_index).first.GetAllSame()) {
      return false;
    }
  }
  return true;
}

std::string NativeGpuKernelModFactory::SupportedTypeList(const std::string &kernel_name) {
  std::string type_lists = "";
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    return type_lists;
  }
  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    std::string type_list = "input[";
    auto attr = (iter->second)[attr_index].first;
    for (size_t input_index = 0; input_index < attr.GetInputSize(); ++input_index) {
      type_list = type_list + TypeIdToString(attr.GetInputAttr(input_index).dtype) +
                  ((input_index == (attr.GetInputSize() - 1)) ? "" : " ");
    }
    type_list = type_list + "], output[";
    for (size_t input_index = 0; input_index < attr.GetOutputSize(); ++input_index) {
      type_list = type_list + TypeIdToString(attr.GetOutputAttr(input_index).dtype) +
                  ((input_index == (attr.GetOutputSize() - 1)) ? "" : " ");
    }
    type_lists = type_lists + type_list + "]; ";
  }
  return type_lists;
}

std::vector<KernelAttr> NativeGpuKernelModFactory::GetGpuSupportedList(const std::string &kernel_name) {
  if (kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
    return kernel::NativeGpuKernelMod::GetGpuSupportedList(kernel_name);
  } else {
    std::vector<KernelAttr> kernel_attr_list;
    auto iter = map_kernel_name_to_creater_.find(kernel_name);
    if (map_kernel_name_to_creater_.end() == iter) {
      return kernel_attr_list;
    }

    for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
      auto attr = (iter->second)[attr_index].first;
      // Skip the invalid attr.
      if (attr.GetInputSize() > 0 || attr.GetOutputSize() > 0) {
        kernel_attr_list.push_back(attr);
      }
    }

    return kernel_attr_list;
  }
}

bool NativeGpuKernelModFactory::IsRegistered(const std::string &kernel_name) {
  // New kernel mod registered.
  if (kernel::Factory<kernel::NativeGpuKernelMod>::Instance().IsRegistered(kernel_name)) {
    return true;
  }

  // Old kernel mod registered.
  if (map_kernel_name_to_creater_.find(kernel_name) != map_kernel_name_to_creater_.end()) {
    return true;
  }

  return false;
}

bool NativeGpuKernelModFactory::ReducePrecision(
  const std::string &kernel_name, std::shared_ptr<mindspore::kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder) {
  MS_EXCEPTION_IF_NULL(builder);
  auto kernel_info = builder->Build();
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    MS_LOG(INFO) << "Not registered GPU kernel: op[" << kernel_name << "]!";
    return false;
  }
  reduce_flag_.first.clear();
  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    auto attr_size = (&(iter->second))->at(attr_index).first.GetInputSize();
    for (size_t input_index = 0; input_index < kernel_info->GetInputNum(); input_index++) {
      if (kernel_info->GetInputDeviceType(input_index) == kNumberTypeInt64 &&
          (iter->second)[attr_index].first.GetInputAttr(input_index % attr_size).dtype == kNumberTypeInt32) {
        builder->SetInputDeviceType(kNumberTypeInt32, input_index);
        reduce_flag_.first.push_back(input_index);
        MS_LOG(INFO) << "Kernel [" << kernel_name << "] does not support int64, cast input " << input_index
                     << " to int32.";
      }
    }
    for (size_t output_index = 0; output_index < kernel_info->GetOutputNum(); output_index++) {
      if (kernel_info->GetOutputDeviceType(output_index) == kNumberTypeInt64 &&
          (iter->second)[attr_index].first.GetOutputAttr(output_index % attr_size).dtype == kNumberTypeInt32) {
        builder->SetOutputDeviceType(kNumberTypeInt32, output_index);
        MS_LOG(INFO) << "Kernel [" << kernel_name << "] does not support int64, cast output " << output_index
                     << " to int32.";
      }
    }
  }
  return NativeGpuKernelModFactory::SearchRegistered(kernel_name, builder->Build());
}

void NativeGpuKernelModFactory::SetRefMapToKernelInfo(const std::string &kernel_name, size_t index,
                                                      device::KernelInfo *kernel_info) {
  MS_ERROR_IF_NULL_WO_RET_VAL(kernel_info);

  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    return;
  }

  const auto &kernel_attr = (iter->second)[index].first;
  if (!kernel_attr.GetOutInRefMap().empty()) {
    kernel_info->set_ref_map(kernel_attr.GetAllOutInRef(), kernel_attr.GetOutInRefMap());
  }
}

void NativeGpuKernelModFactory::CheckSM(const KernelBuildInfo *kernel_info, const size_t &input_index) {
  const int major_sm = GET_MAJOR_SM;
  const bool check_sm = mindspore::device::gpu::CudaCommon::GetInstance().check_sm();
  if (check_sm && major_sm < RECOMMEND_SM && kernel_info->GetInputDeviceType(input_index) == kNumberTypeFloat16) {
    if (major_sm < MINIUM_SM) {
      MS_LOG(EXCEPTION) << "Half precision ops must be used on Devices which computing capacity is >= " << MINIUM_SM
                        << ", but the current device's computing capacity is " << major_sm;
    }
    MS_LOG(WARNING) << "It is recommended to use devices with a computing capacity >= " << RECOMMEND_SM
                    << ", but the current device's computing capacity is " << major_sm << ". "
                    << "In this case, the computation may not be accelerated. Architectures with TensorCores can be "
                       "used to speed up half precision operations, such as Volta and Ampere.";
    mindspore::device::gpu::CudaCommon::GetInstance().set_check_sm(false);
  }
}

std::pair<bool, size_t> NativeGpuKernelModFactory::GpuKernelAttrCheck(const std::string &kernel_name,
                                                                      const KernelBuildInfo *kernel_info) {
  auto iter = map_kernel_name_to_creater_.find(kernel_name);
  if (map_kernel_name_to_creater_.end() == iter) {
    MS_LOG(INFO) << "Not registered GPU kernel: op[" << kernel_name << "]!";
    return std::make_pair(false, 0);
  }
  if ((iter->second).size() == 1 && (iter->second)[0].first.GetInputSize() == 0) {
    return std::make_pair(true, 0);
  }

  for (size_t attr_index = 0; attr_index < (iter->second).size(); ++attr_index) {
    if (!CheckIOParam(kernel_name, kernel_info, &(iter->second), attr_index)) {
      continue;
    }
    bool flag = true;
    auto attr_size = (&(iter->second))->at(attr_index).first.GetInputSize();
    if (kernel_info->GetInputNum() > 0) {
      MS_EXCEPTION_IF_ZERO("attr size", attr_size);
    }
    // data type matching check of all input parameters of kernel
    for (size_t input_index = 0; input_index < kernel_info->GetInputNum(); input_index++) {
      NativeGpuKernelModFactory::CheckSM(kernel_info, input_index);
      if (kernel_info->GetInputDeviceType(input_index) !=
          (iter->second)[attr_index].first.GetInputAttr(input_index % attr_size).dtype) {
        flag = false;
        break;
      }
    }
    if (!flag) {
      continue;
    }
    attr_size = (&(iter->second))->at(attr_index).first.GetOutputSize();
    if (kernel_info->GetOutputNum() > 0) {
      MS_EXCEPTION_IF_ZERO("attr size", attr_size);
    }
    // data type matching check of all output parameters of kernel
    for (size_t output_index = 0; output_index < kernel_info->GetOutputNum(); output_index++) {
      if (kernel_info->GetOutputDeviceType(output_index) !=
          (iter->second)[attr_index].first.GetOutputAttr(output_index % attr_size).dtype) {
        flag = false;
        break;
      }
    }
    // finish data type matching check and return a pair maintain the whether matching is success,
    // if first is true, second is index of matching KernelAttr and creator pair in vector;
    if (flag) {
      size_t match_index = attr_index;
      return std::make_pair(true, match_index);
    }
  }
  return std::make_pair(false, 0);
}

NativeGpuKernelMod *NativeGpuKernelModFactory::Create(const std::string &kernel_name, const CNodePtr &apply_kernel) {
  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const KernelBuildInfo *kernel_build_Info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_Info);
  std::pair<bool, size_t> ret_pair = GpuKernelAttrCheck(kernel_name, kernel_build_Info);
  if (ret_pair.first) {
    SetRefMapToKernelInfo(kernel_name, ret_pair.second, kernel_info);
    return (map_kernel_name_to_creater_.find(kernel_name)->second)[ret_pair.second].second();
  }
  return nullptr;
}

bool NativeGpuKernelModFactory::SearchRegistered(const std::string &kernel_name,
                                                 const KernelBuildInfoPtr &kernel_build_info) {
  std::pair<bool, size_t> ret_pair = GpuKernelAttrCheck(kernel_name, kernel_build_info.get());
  return ret_pair.first;
}
}  // namespace kernel
}  // namespace mindspore
