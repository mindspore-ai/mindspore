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
#include "src/common/utils.h"

using mindspore::lite::PrimitiveC;
namespace mindspore::opt {
namespace {
constexpr size_t split_inputs_size = 3;
const std::vector<schema::PrimitiveType> single_input_ops = {
  schema::PrimitiveType_Reduce,         schema::PrimitiveType_ArgMin,       schema::PrimitiveType_ArgMax,
  schema::PrimitiveType_SpaceToBatch,   schema::PrimitiveType_BatchToSpace, schema::PrimitiveType_SpaceToBatchND,
  schema::PrimitiveType_BatchToSpaceND, schema::PrimitiveType_SpaceToDepth};
}  // namespace

STATUS ReorderCnodeInputs(CNode *cnode, const std::vector<size_t> &perm) {
  // add primitive first
  std::vector<AnfNodePtr> new_inputs = {cnode->input(0)};
  auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));
  auto old_quant_params = primitive_c->input_quant_params();
  std::vector<std::vector<schema::QuantParamT>> new_quant_params;
  // add inputs as perm order
  for (size_t idx : perm) {
    if (idx > cnode->inputs().size() - 1) {
      MS_LOG(ERROR) << "Idx " << idx << " is larger than inputs size: " << cnode->inputs().size() - 1;
      return RET_ERROR;
    }
    new_inputs.emplace_back(cnode->input(idx));
    new_quant_params.emplace_back(old_quant_params.at(idx - 1));
  }
  cnode->set_inputs(new_inputs);
  primitive_c->set_input_quant_params(new_quant_params);
  return RET_OK;
}

bool TfliteInputsOrderExchangePass::Run(const FuncGraphPtr &graph) {
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto primitive_c = GetValueNode<std::shared_ptr<PrimitiveC>>(cnode->input(0));

    if (opt::GetCNodeType(node) == schema::PrimitiveType_Fill) {
      // dims, value => value, dims
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {2, 1})) {
        MS_LOG(ERROR) << "Reorder fill inputs failed";
        return false;
      }
      continue;
    }

    if (opt::GetCNodeType(node) == schema::PrimitiveType_DeConv2D) {
      // output_shape, weights, input => input, weight
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {3, 2})) {
        MS_LOG(ERROR) << "Reorder deconv inputs failed";
        return false;
      }
      continue;
    }

    if (opt::GetCNodeType(node) == schema::PrimitiveType_Split && cnode->inputs().size() == split_inputs_size) {
      // axis, input, ??? => input, axis
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {2, 1})) {
        MS_LOG(ERROR) << "Reorder split inputs failed";
        return false;
      }
      continue;
    }

    bool is_single_input_pad = opt::GetCNodeType(node) == schema::PrimitiveType_Pad &&
                               primitive_c->primitiveT()->value.AsPad() != nullptr &&
                               primitive_c->primitiveT()->value.AsPad()->paddingMode == schema::PaddingMode_CONSTANT;
    bool is_single_input_resize = opt::GetCNodeType(node) == schema::PrimitiveType_Resize &&
                                  primitive_c->primitiveT()->value.AsResize() != nullptr &&
                                  primitive_c->primitiveT()->value.AsResize()->newHeight != 0 &&
                                  primitive_c->primitiveT()->value.AsResize()->newWidth != 0;
    if (lite::IsContain(single_input_ops, opt::GetCNodeType(node)) || is_single_input_pad || is_single_input_resize) {
      if (RET_OK != ReorderCnodeInputs(cnode.get(), {1})) {
        MS_LOG(ERROR) << "Reorder single input failed";
        return false;
      }
      continue;
    }
  }
  return true;
}
}  // namespace mindspore::opt
