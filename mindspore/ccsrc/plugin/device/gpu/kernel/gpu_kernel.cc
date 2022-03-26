/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include <tuple>
#include <set>

namespace mindspore {
namespace kernel {
namespace {
void CheckDeviceSm(const KernelAttr &kernel_attr) {
  const int major_sm = GET_MAJOR_SM;
  if (!mindspore::device::gpu::CudaCommon::GetInstance().check_sm() || major_sm >= RECOMMEND_SM) {
    return;
  }

  for (size_t i = 0; i < kernel_attr.GetInputSize(); ++i) {
    if (kernel_attr.GetInputAttr(i).first != kNumberTypeFloat16) {
      continue;
    }

    if (major_sm < MINIUM_SM) {
      MS_LOG(EXCEPTION) << "Half precision ops can be used on Devices which computing capacity is >= " << MINIUM_SM
                        << ", but the current device's computing capacity is " << major_sm;
    }
    MS_LOG(WARNING) << "It is recommended to use devices with a computing capacity >= " << RECOMMEND_SM
                    << ", but the current device's computing capacity is " << major_sm;
    mindspore::device::gpu::CudaCommon::GetInstance().set_check_sm(false);
    return;
  }
}
}  // namespace
void NativeGpuKernelMod::InferOp() {
  anf_node_ = kernel_node_.lock();
  if (common::AnfAlgo::IsDynamicShape(kernel_node_.lock())) {
    auto cnode = kernel_node_.lock();
    if (NeedSkipExecute(cnode)) {
      std::vector<TypeId> dtypes{common::AnfAlgo::GetOutputInferDataType(cnode, 0)};
      common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, {common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0)},
                                                  cnode.get());
    } else {
      KernelMod::InferShape();
    }
  }
}

void NativeGpuKernelMod::InitOp() {
  auto cnode = kernel_node_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  KernelMod::GetDepndLists(cnode);
  if (!common::AnfAlgo::GetBooleanAttr(cnode, kAttrInputIsDynamicShape) &&
      common::AnfAlgo::GetBooleanAttr(cnode, kAttrOutputIsDynamicShape) && depend_list_.empty()) {
    return;
  }

  MS_LOG(INFO) << "Update Args: " << cnode->fullname_with_scope();
  DestroyResource();
  ResetResource();
  Init(cnode);
}

void NativeGpuKernelMod::SetGpuRefMapToKernelInfo(const CNodePtr &apply_kernel) {
  MS_EXCEPTION_IF_NULL(apply_kernel);
  auto kernel_attrs = GetOpSupport();
  if (kernel_attrs.empty()) {
    return;
  }

  auto index = GetMatchKernelAttrIdxWithException(apply_kernel, kernel_attrs);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(apply_kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const KernelBuildInfo *kernel_build_Info = kernel_info->select_kernel_build_info();
  MS_EXCEPTION_IF_NULL(kernel_build_Info);
  const auto &matched_kernel_attr = kernel_attrs[index];
  if (!matched_kernel_attr.GetOutInRefMap().empty()) {
    kernel_info->set_ref_map(matched_kernel_attr.GetOutInRefMap());
  }
}

bool NativeGpuKernelMod::GpuCheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr) {
  return kernel::Factory<NativeGpuKernelMod>::Instance().Create(kernel_name)->CheckSupport(kernel_name, kernel_attr);
}

std::vector<KernelAttr> NativeGpuKernelMod::GetAllSupportedList(const std::string &kernel_name) {
  if (initialize_.count(kernel_name) == 0) {
    auto kernel_support = GetOpSupport();
    (void)support_map_.emplace(kernel_name, kernel_support);
    (void)initialize_.insert(kernel_name);
  }

  return support_map_[kernel_name];
}

bool NativeGpuKernelMod::CheckSupport(const std::string &kernel_name, const KernelAttr &kernel_attr_to_check) {
  CheckDeviceSm(kernel_attr_to_check);
  auto kernel_attrs = GetAllSupportedList(kernel_name);
  bool is_match;
  std::tie(is_match, std::ignore) = MatchKernelAttr(kernel_attr_to_check, kernel_attrs);
  return is_match;
}

NativeGpuKernelMod::ReducePrecisonRes NativeGpuKernelMod::ReducePrecisionCheck(const std::string &kernel_name,
                                                                               const KernelAttr &kernel_attr_to_check) {
  KernelAttr reduce_kernel_attr;
  std::vector<ReduceDetail> input_reduce_index;
  std::vector<ReduceDetail> output_reduce_index;
  std::vector<KernelAttr> kernel_attr_list = this->GetOpSupport();

  const TypeId from_precision = kNumberTypeInt64;
  const TypeId to_precision = kNumberTypeInt32;
  for (size_t attr_index = 0; attr_index < kernel_attr_list.size(); ++attr_index) {
    auto &cur_kernel_attr = kernel_attr_list[attr_index];
    auto attr_size = cur_kernel_attr.GetInputSize();
    MS_EXCEPTION_IF_ZERO("kernel attr input size", attr_size);
    for (size_t iidx = 0; iidx < kernel_attr_to_check.GetInputSize(); iidx++) {
      auto [type_id, format] = kernel_attr_to_check.GetInputAttr(iidx);
      if (type_id == from_precision && cur_kernel_attr.GetInputAttr(iidx % attr_size).first == to_precision) {
        (void)input_reduce_index.emplace_back(iidx, from_precision, to_precision);
        type_id = to_precision;
        MS_LOG(WARNING) << "Kernel [" << kernel_name << "] does not support int64, cast input " << iidx << " to int32.";
        reduce_kernel_attr.AddInputAttr(type_id, format);
      }
    }
    for (size_t oidx = 0; oidx < kernel_attr_to_check.GetOutputSize(); oidx++) {
      auto [type_id, format] = kernel_attr_to_check.GetOutputAttr(oidx);
      if (type_id == from_precision && cur_kernel_attr.GetOutputAttr(oidx % attr_size).first == to_precision) {
        (void)output_reduce_index.emplace_back(oidx, from_precision, to_precision);
        type_id = to_precision;
        MS_LOG(WARNING) << "Kernel [" << kernel_name << "] does not support int64, cast output " << oidx
                        << " to int32.";
        reduce_kernel_attr.AddOutputAttr(type_id, format);
      }
    }
  }

  return std::make_tuple(CheckSupport(kernel_name, reduce_kernel_attr), input_reduce_index, output_reduce_index);
}

std::map<std::string, std::vector<KernelAttr>> NativeGpuKernelMod::support_map_{};
std::set<std::string> NativeGpuKernelMod::initialize_{};
}  // namespace kernel
}  // namespace mindspore
