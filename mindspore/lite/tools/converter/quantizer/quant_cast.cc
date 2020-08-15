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


#include "mindspore/lite/tools/converter/quantizer/quant_cast.h"
#include <memory>
#include <vector>
#include "mindspore/lite/src/ir/primitive_t_value.h"

namespace mindspore::lite::quant {

ValueNodePtr NewQuantCastValueNode(int src_type, int dst_type, const std::vector<schema::QuantParamT> &quant_params) {
  std::unique_ptr<schema::PrimitiveT> primitive = std::make_unique<schema::PrimitiveT>();
  schema::QuantDTypeCastT quant_dtype_cast;
  quant_dtype_cast.srcT = src_type;  // kNumberTypeInt8;
  quant_dtype_cast.dstT = dst_type;  // kNumberTypeFloat32;
  primitive->value.Set(quant_dtype_cast);
  auto primTValue = std::make_shared<PrimitiveTValue>(primitive.release());
  primTValue->SetQuantType(schema::QuantType_PostTraining);
  for (auto &quant_param : quant_params) {
    std::vector<schema::QuantParamT> quant_params_in = {quant_param};
    primTValue->AddInputQuantParam(quant_params_in);
  }
  return NewValueNode(primTValue);
}

STATUS QuantCast::Run(FuncGraphPtr graph) {
  MS_ASSERT(graph != nullptr);

  auto cnodes = graph->GetOrderedCnodes();
  bool first = true;

  for (auto &cnode : cnodes) {
    auto primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
    auto curnode_quant_type = schema::QuantType_QUANT_NONE;
    if (primitiveT_value == nullptr) {
      MS_LOG(WARNING) << "PrimitiveT_value is nullptr: " << cnode->fullname_with_scope();
    } else {
      curnode_quant_type = primitiveT_value->GetQuantType();
    }
    if (first) {
      if (curnode_quant_type == schema::QuantType_PostTraining && inputDataDType == kNumberTypeFloat32) {
        auto value_node =
          NewQuantCastValueNode(kNumberTypeFloat32, kNumberTypeInt8, primitiveT_value->GetInputQuantParams().front());
        std::vector<AnfNodePtr> op_inputs = {value_node, cnode->input(1)};
        auto quant_cast_cnode = graph->NewCNode(op_inputs);
        quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_quant_cast");
        cnode->set_input(1, quant_cast_cnode);
        MS_LOG(DEBUG) << "Add quant cast at front. "
                      << "cur_node: " << cnode->fullname_with_scope() << " quant_type: " << curnode_quant_type;
      }
      first = false;
      continue;
    }

    for (int i = 1; i < cnode->inputs().size(); i++) {
      auto input_node = cnode->input(i);
      if (!input_node->isa<CNode>()) {
        continue;
      }
      auto input_cnode = std::dynamic_pointer_cast<CNode>(input_node);
      auto input_cnode_primitiveT_value = GetValueNode<std::shared_ptr<PrimitiveTValue>>(input_cnode->input(0));
      if (input_cnode_primitiveT_value == nullptr) {
        MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                      << " PrimitiveTValue is null";
        continue;
      }
      auto input_cnode_quant_type = input_cnode_primitiveT_value->GetQuantType();

      if (curnode_quant_type != input_cnode_quant_type) {
        ValueNodePtr value_node = nullptr;
        if (curnode_quant_type == schema::QuantType_PostTraining &&
            input_cnode_quant_type == schema::QuantType_QUANT_NONE) {
          value_node = NewQuantCastValueNode(kNumberTypeFloat32, kNumberTypeInt8,
                                             primitiveT_value->GetInputQuantParams().front());
        } else if (curnode_quant_type == schema::QuantType_QUANT_NONE &&
                   input_cnode_quant_type == schema::QuantType_PostTraining) {
          value_node = NewQuantCastValueNode(kNumberTypeInt8, kNumberTypeFloat32,
                                             input_cnode_primitiveT_value->GetInputQuantParams().front());
        }
        if (value_node == nullptr) {
          MS_LOG(WARNING) << "value_node is null! "
                          << "cur_node: " << cnode->fullname_with_scope() << " quant_type: "
                          << " input_" << i << ": " << input_cnode->fullname_with_scope()
                          << " quant_type:" << input_cnode_quant_type;
          continue;
        }
        std::vector<AnfNodePtr> op_inputs = {value_node, input_cnode};
        auto quant_cast_cnode = graph->NewCNode(op_inputs);
        quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_quant_cast");
        cnode->set_input(i, quant_cast_cnode);
        MS_LOG(DEBUG) << "Add quant cast. "
                      << "cur_node: " << cnode->fullname_with_scope() << " quant_type: " << curnode_quant_type
                      << " input_" << i << ": " << input_cnode->fullname_with_scope()
                      << " quant_type:" << input_cnode_quant_type;
      } else {
        MS_LOG(DEBUG) << "No need to add quant cast. "
                      << "cur_node: " << cnode->fullname_with_scope() << " quant_type: " << curnode_quant_type
                      << " input_" << i << ": " << input_cnode->fullname_with_scope()
                      << " quant_type:" << input_cnode_quant_type;
      }
    }
  }
  return RET_OK;
}

}  // namespace mindspore::lite::quant
