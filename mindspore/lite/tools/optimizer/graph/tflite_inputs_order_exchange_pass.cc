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
#include "tools/optimizer/graph/tflite_inputs_order_exchange_pass.h"
#include <vector>
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/quantizer/quant_cast.h"

using mindspore::lite::PrimitiveC;
namespace mindspore::opt {
namespace {
constexpr size_t split_inputs_size = 3;
}  // namespace
bool TfliteInputsOrderExchangePass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
    if (opt::GetCNodeType(node) == schema::PrimitiveType_DeConv2D) {
      cnode->set_input(1, cnode->input(3));
      auto inputs = cnode->inputs();
      inputs.pop_back();
      cnode->set_inputs(inputs);

      auto input_quant_params = primitive_c->input_quant_params();
      input_quant_params[0] = input_quant_params.at(2);
      input_quant_params.pop_back();
      primitive_c->set_input_quant_params(input_quant_params);
      continue;
    }

    if (opt::GetCNodeType(node) == schema::PrimitiveType_Split && cnode->inputs().size() == split_inputs_size) {
      cnode->set_input(1, cnode->input(2));
      auto inputs = cnode->inputs();
      inputs.pop_back();
      cnode->set_inputs(inputs);

      auto input_quant_params = primitive_c->input_quant_params();
      input_quant_params[0] = input_quant_params.at(1);
      input_quant_params.pop_back();
      primitive_c->set_input_quant_params(input_quant_params);
      continue;
    }

    if (opt::GetCNodeType(node) == schema::PrimitiveType_Reduce ||
        opt::GetCNodeType(node) == schema::PrimitiveType_ArgMin ||
        opt::GetCNodeType(node) == schema::PrimitiveType_ArgMax ||
        opt::GetCNodeType(node) == schema::PrimitiveType_SpaceToBatch ||
        opt::GetCNodeType(node) == schema::PrimitiveType_BatchToSpace ||
        opt::GetCNodeType(node) == schema::PrimitiveType_SpaceToBatchND ||
        opt::GetCNodeType(node) == schema::PrimitiveType_BatchToSpaceND ||
        opt::GetCNodeType(node) == schema::PrimitiveType_SpaceToDepth ||
        (opt::GetCNodeType(node) == schema::PrimitiveType_Pad && primitive_c->primitiveT()->value.AsPad() != nullptr &&
         primitive_c->primitiveT()->value.AsPad()->paddingMode == schema::PaddingMode_CONSTANT) ||
        (opt::GetCNodeType(node) == schema::PrimitiveType_Resize &&
         primitive_c->primitiveT()->value.AsResize() != nullptr &&
         primitive_c->primitiveT()->value.AsResize()->newHeight != 0 &&
         primitive_c->primitiveT()->value.AsResize()->newWidth != 0)) {
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.emplace_back(cnode->input(0));
      new_inputs.emplace_back(cnode->input(1));
      cnode->set_inputs(new_inputs);
      continue;
    }
  }
  return true;
}
}  // namespace mindspore::opt
