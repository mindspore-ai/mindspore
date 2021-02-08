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

#include "backend/kernel_compiler/hccl/hccl_kernel_metadata.h"
#include <memory>
#include <set>
#include "utils/utils.h"
#include "backend/kernel_compiler/hccl/hcom_util.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/parallel/context.h"

namespace mindspore {
namespace kernel {
namespace {
std::string GetKernelFormat(const CNodePtr &kernel_node, size_t index) {
  const std::set<std::string> kReduceNoSupportedSet = {kOpFormat_FRAC_Z, kOpFormat_FRACTAL_Z_C04, kOpFormat_C1HWNCoC0};
  auto op_name = AnfAlgo::GetCNodeName(kernel_node);
  auto parallel_context_instance = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context_instance);
  if (parallel_context_instance->enable_parallel_optimizer() && op_name == kBroadcast) {
    return kOpFormat_DEFAULT;
  }
  if (op_name == kReceive || op_name == kHcomSend) {
    return kOpFormat_DEFAULT;
  }
  auto format = AnfAlgo::GetPrevNodeOutputFormat(kernel_node, index);
  if (op_name != kReduceScatter && op_name != kAllGatherOpName) {
    return format;
  }
  if (format == kOpFormat_FRAC_NZ && AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index).size() <= 2) {
    return kOpFormat_DEFAULT;
  }
  if (kReduceNoSupportedSet.find(format) != kReduceNoSupportedSet.end()) {
    return kOpFormat_DEFAULT;
  }
  return format;
}
}  // namespace
void HcclMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  const std::vector<TypeId> kHcclSupportTypes = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16,
                                                 kNumberTypeFloat32, kNumberTypeInt16};
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (op_name != kAllGather && op_name != kAllReduce && op_name != kBroadcast && op_name != kReduceScatter &&
      op_name != kHcomSend && op_name != kReceive) {
    MS_LOG(DEBUG) << "Hccl does not have op [" << op_name << "]";
    return;
  }
  for (const auto &type : kHcclSupportTypes) {
    std::vector<std::string> inputs_format{};
    std::vector<TypeId> inputs_type{};
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      inputs_format.emplace_back(GetKernelFormat(kernel_node, input_index));
      inputs_type.push_back(type);
    }
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      if (op_name == kReduceScatter && AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrFusion) > 0) {
        outputs_format.emplace_back(GetKernelFormat(kernel_node, 0));
      } else {
        outputs_format.emplace_back(GetKernelFormat(kernel_node, output_index));
      }
      outputs_type.push_back(type);
    }
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    builder.SetInputsFormat(inputs_format);
    builder.SetInputsDeviceType(inputs_type);
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_type);
    builder.SetKernelType(HCCL_KERNEL);
    kernel_info_list->push_back(builder.Build());
  }
}
}  // namespace kernel
}  // namespace mindspore
