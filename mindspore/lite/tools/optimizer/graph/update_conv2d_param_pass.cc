/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "mindspore/lite/include/errorcode.h"
#include "src/ops/primitive_c.h"

namespace mindspore::opt {
bool UpdateConv2DParamPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  int status = RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto type = opt::GetCNodeType(node);
    if (type == schema::PrimitiveType_DepthwiseConv2D) {
      auto dwconv2d_cnode = node->cast<CNodePtr>();
      auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(dwconv2d_cnode->input(0));
      if (primitive_c == nullptr) {
        MS_LOG(ERROR) << "Depthwise conv2D node has no primitiveC.";
        return RET_ERROR;
      }
      auto primT = primitive_c->primitiveT();
      if (primT == nullptr) {
        MS_LOG(ERROR) << "Depthwise conv2D node has no primitiveT.";
        return RET_ERROR;
      }
      int channel_in = primT->value.AsDepthwiseConv2D()->channelIn;
      if (channel_in == -1) {
        auto input_node = node->cast<CNodePtr>()->input(lite::kAnfPopulaterInputNumTwo);
        MS_ASSERT(input_node != nullptr);
        if (input_node->isa<Parameter>()) {
          auto param_node = input_node->cast<ParameterPtr>();
          auto param = param_node->default_param();
          auto weight = std::dynamic_pointer_cast<ParamValueLite>(param);
          primT->value.AsDepthwiseConv2D()->channelIn = weight->tensor_shape().at(0);
        }
      }
    } else if (type == schema::PrimitiveType_Conv2D) {
      auto conv2d_cnode = node->cast<CNodePtr>();
      auto primitive_c = GetValueNode<std::shared_ptr<lite::PrimitiveC>>(conv2d_cnode->input(0));
      if (primitive_c == nullptr) {
        MS_LOG(DEBUG) << "Conv2D node has no primitiveC.";
        continue;
      }
      auto primT = primitive_c->primitiveT();
      if (primT == nullptr) {
        MS_LOG(DEBUG) << "Conv2D node has no primitiveT.";
        continue;
      }
      auto conv2d_primt = primT->value.AsConv2D();
      auto weight_node = conv2d_cnode->input(lite::kAnfPopulaterInputNumTwo);
      if (weight_node == nullptr) {
        MS_LOG(DEBUG) << "Conv2D weight node is nullptr.";
        continue;
      }
      if (!weight_node->isa<Parameter>()) {
        MS_LOG(DEBUG) << "Conv2D weight node is not parameter.";
        continue;
      }
      auto weight_param = weight_node->cast<ParameterPtr>();
      if (!weight_param->has_default()) {
        MS_LOG(DEBUG) << "Conv2D weight node is not parameter.";
        continue;
      }
      auto default_param = weight_param->default_param();
      auto weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(default_param);
      auto weight_shape = weight_tensor->tensor_shape();
      if (fmk_type == lite::converter::FmkType_TF && conv2d_primt->format == schema::Format_NHWC) {
        conv2d_primt->kernelH = weight_shape[0];
        conv2d_primt->kernelW = weight_shape[1];
        conv2d_primt->channelIn = weight_shape[2];
        conv2d_primt->channelOut = weight_shape[3];
      }
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "remove identity pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::opt
