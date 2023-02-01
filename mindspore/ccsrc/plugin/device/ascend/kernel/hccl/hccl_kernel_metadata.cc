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

#include "plugin/device/ascend/kernel/hccl/hccl_kernel_metadata.h"
#include <memory>
#include <algorithm>
#include <set>
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/kernel/hccl/hcom_util.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t N_nchw = 0;
constexpr size_t C_nchw = 1;
std::string GetKernelFormat(const CNodePtr &kernel_node, size_t index) {
  static const std::set<std::string> kReduceNoSupportedSet = {kOpFormat_FRAC_Z, kOpFormat_FRACTAL_Z_C04,
                                                              kOpFormat_C1HWNCoC0};
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto parallel_context_instance = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context_instance);
  if (parallel_context_instance->enable_parallel_optimizer() && op_name == kBroadcastOpName) {
    return kOpFormat_DEFAULT;
  }
  if (op_name == kReceiveOpName || op_name == kHcomSendOpName || op_name == kAllToAllvOpName) {
    return kOpFormat_DEFAULT;
  }
  auto format = AnfAlgo::GetPrevNodeOutputFormat(kernel_node, index);
  if (op_name != kReduceScatterOpName && op_name != kAllGatherOpName) {
    return format;
  }
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, index);
  if (op_name == kAllGatherOpName && input_shape.size() <= kShape4dDims) {
    auto pad_shape = trans::PaddingShapeTo4dDefault(input_shape, kernel_node);
    if (pad_shape[N_nchw] % kCubeSize != 0 || pad_shape[C_nchw] % kCubeSize != 0) {
      return kOpFormat_DEFAULT;
    }
  }
  if (format == kOpFormat_FRAC_NZ && input_shape.size() <= kShape2dDims) {
    return kOpFormat_DEFAULT;
  }
  if (kReduceNoSupportedSet.find(format) != kReduceNoSupportedSet.end()) {
    return kOpFormat_DEFAULT;
  }
  return format;
}
}  // namespace
void HcclMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  static const std::vector<TypeId> kHcclSupportTypes = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16,
                                                        kNumberTypeFloat32, kNumberTypeInt16};
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (op_name != kAllGatherOpName && op_name != kAllReduceOpName && op_name != kBroadcastOpName &&
      op_name != kReduceScatterOpName && op_name != kHcomSendOpName && op_name != kReceiveOpName &&
      op_name != kAllToAllvOpName) {
    MS_LOG(DEBUG) << "Hccl does not have op [" << op_name << "]";
    return;
  }
  TypeId recv_type;
  if (op_name == kReceiveOpName) {
    if (!HcomUtil::GetHcomReceiveType(kernel_node, &recv_type)) {
      MS_LOG(EXCEPTION) << "GetHcomReceiveType fail!";
    }
    auto res = std::find(kHcclSupportTypes.begin(), kHcclSupportTypes.end(), recv_type);
    if (res == kHcclSupportTypes.end()) {
      MS_LOG(EXCEPTION) << "HcclReceive cannot support data type: " << TypeIdToType(recv_type);
    }
  }
  for (const auto &type : kHcclSupportTypes) {
    std::vector<std::string> inputs_format{};
    std::vector<TypeId> inputs_type{};
    std::vector<KernelObjectType> input_object_type{};
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      (void)inputs_format.emplace_back(GetKernelFormat(kernel_node, input_index));
      (void)inputs_type.push_back(type);
      (void)input_object_type.push_back(KernelObjectType::TENSOR);
    }
    std::vector<std::string> outputs_format;
    std::vector<TypeId> outputs_type;
    std::vector<KernelObjectType> output_object_type{};
    size_t output_num = AnfAlgo::GetOutputElementNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      (void)outputs_format.emplace_back(GetKernelFormat(kernel_node, output_index));
      if (op_name == kReceiveOpName) {
        outputs_type.push_back(recv_type);
      } else {
        outputs_type.push_back(type);
      }
      (void)output_object_type.push_back(KernelObjectType::TENSOR);
    }
    auto builder = KernelBuildInfo::KernelBuildInfoBuilder();
    builder.SetInputsFormat(inputs_format);
    builder.SetInputsDeviceType(inputs_type);
    builder.SetOutputsFormat(outputs_format);
    builder.SetOutputsDeviceType(outputs_type);
    builder.SetKernelType(HCCL_KERNEL);
    builder.SetInputsKernelObjectType(input_object_type);
    builder.SetOutputsKernelObjectType(output_object_type);
    kernel_info_list->push_back(builder.Build());
  }
}
}  // namespace kernel
}  // namespace mindspore
