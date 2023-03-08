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
#include <string>
#include "plugin/device/ascend/kernel/akg/akg_kernel_metadata.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/oplib/oplib.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
void AkgMetadataInfo(const CNodePtr &kernel_node, std::vector<KernelBuildInfoPtr> *kernel_info_list) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_info_list);

  std::string op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  const std::vector<std::string> support_devices = {"aicore", "aicpu", "cuda"};
  for (size_t i = 0; i < support_devices.size(); i++) {
    auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, OpImplyType::kImplyAKG);
    if (op_info_ptr == nullptr) {
      continue;
    }

    if (!ParseMetadata(kernel_node, op_info_ptr, Processor(i), kernel_info_list)) {
      MS_LOG(WARNING) << "Akg parsed metadata of op[" << op_name << "], device[" << support_devices[i] << "] failed.";
    } else {
      MS_LOG(DEBUG) << "Akg parsed metadata of op[" << op_name << "], device[" << support_devices[i] << "].";
      break;
    }
  }

  if (kernel_info_list->empty()) {
    MS_LOG(WARNING) << "Akg dose not has metadata of op[" << op_name << "].";
  }
}
}  // namespace kernel
}  // namespace mindspore
