/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/format/to_nchw_format.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
STATUS ToNCHWFormat::GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) {
  MS_ERROR_IF_NULL_W_RET_VAL(cnode, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(trans_info, lite::RET_ERROR);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ERROR_IF_NULL_W_RET_VAL(prim, lite::RET_ERROR);
  if (sensitive_ops_.find(prim->name()) == sensitive_ops_.end()) {
    MS_LOG(DEBUG) << "node " << prim->name() << " do not need to change format !";
    return lite::RET_OK;
  }
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    auto node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
    if (node_format == mindspore::NCHW || node_format == mindspore::KCHW) {
      MS_LOG(DEBUG) << "node's format has been nchw, no need to transfer, " << cnode->fullname_with_scope();
      return lite::RET_OK;
    }
    if (node_format != mindspore::NHWC && node_format != mindspore::KHWC) {
      MS_LOG(ERROR) << "node's format is invalid, which must be nhwc or nchw, now is " << node_format
                    << ", node name is " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  trans_info->pre_ = opt::kNHWC2NCHW;
  trans_info->post_ = opt::kNCHW2NHWC;
  return lite::RET_OK;
}

STATUS ToNCHWFormat::DecideConvWeightSrcAndDstFormat(const CNodePtr &cnode, schema::Format *src_format,
                                                     schema::Format *dst_format) {
  MS_ERROR_IF_NULL_W_RET_VAL(cnode, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(src_format, lite::RET_ERROR);
  MS_ERROR_IF_NULL_W_RET_VAL(dst_format, lite::RET_ERROR);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ERROR_IF_NULL_W_RET_VAL(prim, lite::RET_ERROR);
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    auto node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
    if (node_format == mindspore::NCHW) {
      MS_LOG(DEBUG) << "node's format has been nchw, no need to transfer, " << cnode->fullname_with_scope();
      return lite::RET_OK;
    }
    if (node_format != mindspore::NHWC) {
      MS_LOG(ERROR) << "node's format is invalid, which must be nhwc or nchw, now is " << node_format
                    << ", node name is " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  *src_format = schema::Format_KHWC;
  *dst_format = schema::Format_KCHW;
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
