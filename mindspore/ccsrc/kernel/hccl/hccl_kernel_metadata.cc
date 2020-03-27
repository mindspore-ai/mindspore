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

#include "kernel/hccl/hccl_kernel_metadata.h"
#include <memory>
#include "utils/utils.h"
#include "kernel/hccl/hcom_util.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
void HcclMetadataInfo(const CNodePtr &kernel_node, std::vector<std::shared_ptr<KernelBuildInfo>> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_info_list);
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string op_name = AnfAlgo::GetCNodeName(kernel_node);
  if (op_name != kAllGather && op_name != kAllReduce && op_name != kBroadcast && op_name != kReduceScatter) {
    MS_LOG(INFO) << "Hccl can't find op [" << op_name << "]";
    return;
  }

  std::vector<TypeId> data_type_list{kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeInt8, kNumberTypeInt32};
  std::vector<std::string> input_format, output_format;
  std::vector<TypeId> input_type, output_type;
  for (const auto &data_type : data_type_list) {
    for (const auto &format : k4DSupportFormat) {
      auto builder = std::make_shared<KernelBuildInfo::KernelBuildInfoBuilder>();
      input_format.clear();
      input_format.push_back(format);
      input_type.clear();
      input_type.push_back(data_type);
      output_format.clear();
      output_format.push_back(format);
      output_type.clear();
      output_type.push_back(data_type);

      builder->SetInputsFormat(input_format);
      builder->SetInputsDeviceType(input_type);
      builder->SetOutputsFormat(output_format);
      builder->SetOutputsDeviceType(output_type);
      builder->SetKernelType(HCCL_KERNEL);
      kernel_info_list->emplace_back(builder->Build());
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
