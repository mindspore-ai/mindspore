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
#include <vector>
#include "ops/fusion/conv2d_fusion.h"
#include "mindspore/lite/include/errorcode.h"

namespace mindspore::opt {
namespace {
constexpr int kAnfPopulaterInputNumTwo = 2;
}

lite::STATUS UpdateConv2DParamPass::UpdateCommonConv2D(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  if (fmk_type_ != lite::converter::FmkType_TF) {
    return lite::RET_OK;
  }
  auto conv = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode->input(0));
  if (conv == nullptr) {
    MS_LOG(DEBUG) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  if (conv->GetAttr(ops::kFormat) == nullptr || conv->get_format() != mindspore::NHWC) {
    return lite::RET_OK;
  }
  auto weight_node = cnode->input(kAnfPopulaterInputNumTwo);
  if (weight_node == nullptr) {
    MS_LOG(DEBUG) << "Conv2D weight node is nullptr.";
    return lite::RET_ERROR;
  }
  if (!weight_node->isa<Parameter>()) {
    MS_LOG(DEBUG) << "Conv2D weight node is not parameter.";
    return lite::RET_NO_CHANGE;
  }
  auto weight_param = weight_node->cast<ParameterPtr>();
  if (!weight_param->has_default()) {
    MS_LOG(DEBUG) << "Conv2D weight node is not parameter.";
    return lite::RET_NO_CHANGE;
  }
  auto default_param = weight_param->default_param();
  auto weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(default_param);
  auto weight_shape = weight_tensor->tensor_shape();
  std::vector<int64_t> kernel_size = {weight_shape[0], weight_shape[1]};
  conv->set_kernel_size(kernel_size);
  conv->set_in_channel(weight_shape[2]);
  conv->set_out_channel(weight_shape[3]);
  return lite::RET_OK;
}

lite::STATUS UpdateConv2DParamPass::UpdateDepthWiseConv2D(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto conv = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(cnode->input(0));
  if (conv == nullptr) {
    MS_LOG(ERROR) << "cnode is invalid.";
    return lite::RET_ERROR;
  }
  int64_t channel_in = conv->GetAttr(ops::kInChannel) != nullptr ? conv->get_in_channel() : -1;
  if (channel_in == -1) {
    auto input_node = cnode->input(kAnfPopulaterInputNumTwo);
    MS_ASSERT(input_node != nullptr);
    if (input_node->isa<Parameter>()) {
      auto param_node = input_node->cast<ParameterPtr>();
      auto param = param_node->default_param();
      auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
      conv->set_in_channel(static_cast<int64_t>(weight->tensor_shape().at(0)));
    }
  }
  return lite::RET_OK;
}

bool UpdateConv2DParamPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  int status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimConv2DFusion)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto conv = GetValueNode<std::shared_ptr<mindspore::ops::Conv2DFusion>>(cnode->input(0));
    if (conv == nullptr) {
      MS_LOG(ERROR) << "Depthwise conv2D node has no primitiveC.";
      return RET_ERROR;
    }
    if (conv->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(conv->GetAttr(ops::kIsDepthWise))) {
      status = UpdateDepthWiseConv2D(cnode);
    } else {
      status = UpdateCommonConv2D(cnode);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "update con2d failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
