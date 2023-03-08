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

#include "plugin/device/ascend/kernel/rts/getnext_dynamic.h"

#include <memory>
#include <numeric>
#include <functional>
#include <string>
#include <map>
#include "abstract/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/common_utils.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "runtime/device/kernel_runtime.h"
#include "utils/ms_context.h"
#include "include/backend/data_queue/data_queue_mgr.h"

namespace mindspore {
namespace kernel {
GetNextDynamic::GetNextDynamic() {}

GetNextDynamic::~GetNextDynamic() {}

bool GetNextDynamic::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                            const std::vector<AddressPtr> &, void *) {
  return true;
}

bool GetNextDynamic::Init(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  anf_node_ = anf_node;
  auto output_num = AnfAlgo::GetOutputTensorNum(anf_node);
  std::vector<size_t> size_list;
  for (size_t idx = 0; idx < output_num; ++idx) {
    size_list.push_back(0);
  }
  SetOutputSizeList(size_list);
  return true;
}

int GetNextDynamic::Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &,
                           const std::vector<KernelTensorPtr> &, const std::map<uint32_t, tensor::TensorPtr> &) {
  auto data_kernel = anf_node_.lock();
  device::UpdateGetNextNode(data_kernel);
  return KernelErrorCode::KRET_OK;
}

GetNextDynamicDesc::GetNextDynamicDesc() {}

GetNextDynamicDesc::~GetNextDynamicDesc() {}

// GetNextDynamic KernelInfo Register
std::vector<std::shared_ptr<kernel::KernelBuildInfo>> GetNextDynamicDesc::GetKernelInfo(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<std::shared_ptr<kernel::KernelBuildInfo>> get_next_dynamic_build_info{};
  auto output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  std::vector<string> output_format;
  std::vector<TypeId> output_type;
  for (size_t idx = 0; idx < output_num; ++idx) {
    auto data_type = common::AnfAlgo::GetOutputInferDataType(kernel_node, idx);
    output_type.emplace_back(data_type);
    output_format.emplace_back(kOpFormat_DEFAULT);
  }
  auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
  builder.SetOutputsFormat(output_format);
  builder.SetOutputsDeviceType(output_type);
  builder.SetProcessor(AICORE);
  builder.SetKernelType(RT_KERNEL);
  builder.SetFusionType(kPatternOpaque);
  get_next_dynamic_build_info.emplace_back(builder.Build());
  return get_next_dynamic_build_info;
}
}  // namespace kernel
}  // namespace mindspore
