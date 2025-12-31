/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/kernel_executor/rts/send.h"
#include <vector>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
SendKernel::~SendKernel() {}

bool SendKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode) && cnode->GetAttr(kAttrEventId) == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "SendKernel has no attr kAttrEventId";
  }

  if (common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode)) {
    event_id_ = GetValue<uint32_t>(primitive->GetAttr(kAttrEventId));
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrRecordEvent, cnode)) {
    event_ = reinterpret_cast<aclrtEvent>(GetValue<uintptr_t>(primitive->GetAttr(kAttrRecordEvent)));
  }
  MS_LOG(INFO) << "StreamSend op event id:" << event_id_ << ", event ptr: " << event_;

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  Resize(input_kernel_tensors, output_kernel_tensors);
  return true;
}

bool SendKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                        const std::vector<KernelTensor *> &, void *stream_ptr) {
  if (event_ == nullptr) {
    MS_LOG(INFO) << "Skip launching StreamSend.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto status = CALL_ASCEND_API(aclrtRecordEvent, event_, stream_ptr);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "StreamSend op aclrtRecordEvent failed!";
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
