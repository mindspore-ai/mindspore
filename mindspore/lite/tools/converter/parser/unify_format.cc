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

#include "tools/converter/parser/unify_format.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kInputChannal = 3;
}
void UnifyFormatToNHWC::GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) {
  MS_ASSERT(cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  auto &specify_nhwc_op_map = opt::GetNHWCOpMap();
  auto &specify_nchw_op_map = opt::GetNCHWOpMap();
  if (fmk_type_ == lite::converter::FmkType_TFLITE) {
    if (specify_nchw_op_map.find(prim->name()) == specify_nchw_op_map.end()) {
      return;
    }
    trans_info->pre_ = opt::kNHWC2NCHW;
    trans_info->post_ = opt::kNCHW2NHWC;
  } else if (fmk_type_ == lite::converter::FmkType_TF) {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end() && opt::GetFormat(cnode) == NCHW) {
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
    if (specify_nchw_op_map.find(prim->name()) != specify_nchw_op_map.end()) {
      trans_info->pre_ = opt::kNHWC2NCHW;
      trans_info->post_ = opt::kNCHW2NHWC;
    }
  } else {
    if (specify_nhwc_op_map.find(prim->name()) != specify_nhwc_op_map.end()) {
      if (fmk_type_ == lite::converter::FmkType_ONNX && prim->GetAttr(ops::kFormat) != nullptr &&
          GetValue<int64_t>(prim->GetAttr(ops::kFormat)) == NHWC) {
        return;
      }
      trans_info->pre_ = opt::kNCHW2NHWC;
      trans_info->post_ = opt::kNHWC2NCHW;
    }
  }
}

void UnifyFormatToNHWC::SetSensitiveOps() {
  auto &sensitive_nhwc_ops = opt::GetNHWCOpMap();
  auto &sensitive_nchw_ops = opt::GetNCHWOpMap();
  sensitive_ops_.insert(sensitive_nhwc_ops.begin(), sensitive_nhwc_ops.end());
  sensitive_ops_.insert(sensitive_nchw_ops.begin(), sensitive_nchw_ops.end());
}

bool UnifyFormatToNHWC::DecideWhetherHandleGraphInput(const FuncGraphPtr &func_graph, const ShapeVector &shape) {
  if (fmk_type_ == converter::FmkType_TF || fmk_type_ == converter::FmkType_TFLITE) {
    return false;
  }
  if (func_graph->get_inputs().size() == 1 && fmk_type_ == lite::converter::FmkType_ONNX &&
      shape[opt::kInputIndexThree] == kInputChannal && shape[1] == -1) {
    return false;
  }
  return true;
}

bool UnifyFormatToNHWC::DecideWhetherInferShapeForNewNode() { return false; }
}  // namespace lite
}  // namespace mindspore
