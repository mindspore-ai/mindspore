/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/rts/res_limit.h"
#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
ResLimitKernel::~ResLimitKernel() {}

bool ResLimitKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (common::AnfAlgo::HasNodeAttr(kAttrCubeNum, anf_node->cast<CNodePtr>())) {
    ResLimitInfo limit_info;
    limit_info.type = aclrtDevResLimitType::ACL_RT_DEV_RES_CUBE_CORE;
    limit_info.core_num = GetValue<uint32_t>(primitive->GetAttr(kAttrCubeNum));
    limit_info.stream_id = GetValue<uint32_t>(primitive->GetAttr(kAttrStreamId));
    res_limit_infos_.push_back(limit_info);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrVectorNum, anf_node->cast<CNodePtr>())) {
    ResLimitInfo limit_info;
    limit_info.type = aclrtDevResLimitType::ACL_RT_DEV_RES_VECTOR_CORE;
    limit_info.core_num = GetValue<uint32_t>(primitive->GetAttr(kAttrVectorNum));
    limit_info.stream_id = GetValue<uint32_t>(primitive->GetAttr(kAttrStreamId));
    res_limit_infos_.push_back(limit_info);
  }
  if (anf_node->cast<CNodePtr>()->HasAttr(mindspore::ops::kHasDynamicValue)) {
    is_dyn_graph_ = true;
  }
  return true;
}

int ResLimitKernel::Resize(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &) {
  if (!is_exec_resize_ || is_dyn_graph_) {
    for (const auto &iter : res_limit_infos_) {
      auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(iter.stream_id);
      auto ret = CALL_ASCEND_API(aclrtSetStreamResLimit, stream_ptr, iter.type, iter.core_num);
      if (ret != ACL_SUCCESS) {
        MS_LOG(EXCEPTION) << "Call aclrtSetStreamResLimit failed! Error flag is " << ret;
      }
    }
    is_exec_resize_ = true;
  }
  return 0;
}

bool ResLimitKernel::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                            const std::vector<KernelTensor *> &, void *stream_ptr) {
  return true;
}
}  // namespace kernel
}  // namespace mindspore
