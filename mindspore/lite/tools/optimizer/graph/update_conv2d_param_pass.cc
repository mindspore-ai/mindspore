/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include <memory>
#include <utility>
#include <vector>
#include "ops/fusion/conv2d_fusion.h"
#include "mindspore/lite/include/errorcode.h"

namespace mindspore::opt {
namespace {
void SetConvAttr(const PrimitivePtr &prim, const std::vector<int64_t> &kernel_size, int64_t in_channel,
                 int64_t out_channel) {
  MS_ASSERT(prim != nullptr);
  if (prim->GetAttr(ops::kKernelSize) == nullptr) {
    prim->AddAttr(ops::kKernelSize, MakeValue(kernel_size));
  } else {
    auto origin_kernel_size = GetValue<std::vector<int64_t>>(prim->GetAttr(ops::kKernelSize));
    if (std::any_of(origin_kernel_size.begin(), origin_kernel_size.end(), [](int64_t size) { return size <= 0; })) {
      prim->AddAttr(ops::kKernelSize, MakeValue(kernel_size));
    }
  }
  if (prim->GetAttr(ops::kInChannel) == nullptr || GetValue<int64_t>(prim->GetAttr(ops::kInChannel)) <= 0) {
    prim->AddAttr(ops::kInChannel, MakeValue(in_channel));
  }
  if (prim->GetAttr(ops::kOutChannel) == nullptr || GetValue<int64_t>(prim->GetAttr(ops::kOutChannel)) <= 0) {
    prim->AddAttr(ops::kOutChannel, MakeValue(out_channel));
  }
}
}  // namespace

STATUS UpdateConv2DParamPass::UpdateConv2DAttr(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (cnode->size() < kInputSizeThree) {
    MS_LOG(ERROR) << "conv2d's input size is invalid, now is " << (cnode->size() - 1);
    return lite::RET_ERROR;
  }
  auto weight = cnode->input(kInputIndexTwo);
  if (weight == nullptr) {
    MS_LOG(ERROR) << "conv2d's weight is invalid, now is nullptr.";
    return lite::RET_ERROR;
  }
  auto abstract = weight->abstract();
  ShapeVector shape;
  if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch shape from abstract failed.";
    return lite::RET_ERROR;
  }
  if (shape.empty()) {
    return lite::RET_OK;
  }
  if (shape.size() != kInputSizeFour) {
    MS_LOG(ERROR) << "conv2d weight shape size is invalid.";
    return lite::RET_ERROR;
  }
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_ASSERT(prim != nullptr);
  if (prim->GetAttr(ops::kFormat) == nullptr) {
    MS_LOG(ERROR) << "current conv2d's format is undefined.";
    return lite::RET_ERROR;
  }
  auto format = static_cast<mindspore::Format>(GetValue<int64_t>(prim->GetAttr(ops::kFormat)));
  if (format != mindspore::NHWC && format != mindspore::NCHW) {
    MS_LOG(ERROR) << "conv2d's format only support nhwc or nchw, now is " << format;
    return lite::RET_ERROR;
  }
  auto kernel_size = format == mindspore::NHWC ? ShapeVector{shape[1], shape[kInputIndexTwo]}
                                               : ShapeVector{shape[kInputIndexTwo], shape[kInputIndexThree]};
  int64_t in_channel = format == mindspore::NHWC ? shape[kInputIndexThree] : shape[1];
  int64_t out_channel = shape[0];
  if (prim->GetAttr(ops::kGroup) == nullptr) {
    bool is_depth_wise =
      prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(prim->GetAttr(ops::kIsDepthWise));
    prim->AddAttr(ops::kGroup, MakeValue(is_depth_wise ? out_channel : 1));
  }
  MS_ASSERT(prim->GetAttr(ops::kGroup) != nullptr);
  auto group = GetValue<int64_t>(prim->GetAttr(ops::kGroup));
  if (CheckPrimitiveType(cnode, prim::kPrimConv2dTransposeFusion)) {
    std::swap(in_channel, out_channel);
  }
  if (CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
    in_channel *= group;
  } else {
    out_channel *= group;
  }

  SetConvAttr(prim, kernel_size, in_channel, out_channel);
  return lite::RET_OK;
}

bool UpdateConv2DParamPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(node, prim::kPrimConv2DFusion) ||
        CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
      if (UpdateConv2DAttr(cnode) != lite::RET_OK) {
        MS_LOG(ERROR) << "update conv2d attr failed.";
        return false;
      }
    }
  }
  return true;
}
}  // namespace mindspore::opt
