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

#include "mindspore/lite/tools/converter/quantizer/quant_cast.h"
#include <memory>
#include <vector>
#include "ops/gather.h"
#include "ops/quant_dtype_cast.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
ValueNodePtr NewQuantCastValueNode(int src_type, int dst_type, const std::vector<schema::QuantParamT> &quant_params) {
  auto prim_c = std::make_shared<ops::QuantDTypeCast>();
  prim_c->Init(src_type, dst_type);
  auto quant_params_holder = std::make_shared<QuantParamHolder>();
  quant_params_holder->set_quant_type(schema::QuantType_PostTraining);
  for (auto &quant_param : quant_params) {
    std::vector<schema::QuantParamT> quant_params_in = {quant_param};
    quant_params_holder->AddInputQuantParam(quant_params_in);
    quant_params_holder->AddOutputQuantParam(quant_params_in);
  }
  prim_c->AddAttr("quant_params", quant_params_holder);
  return NewValueNode(prim_c);
}

STATUS QuantCast::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto cnodes = graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
    auto primitive_quant_param_holder = GetCNodeQuantHolder(primitive_c);
    MS_ASSERT(primitive_quant_param_holder != nullptr);
    auto curnode_quant_type = schema::QuantType_QUANT_NONE;
    if (primitive_c == nullptr) {
      MS_LOG(WARNING) << "primitive_c is nullptr: " << cnode->fullname_with_scope();
    } else {
      curnode_quant_type = primitive_quant_param_holder->quant_type();
    }
    if (primitive_c->name() == ops::kNameGather) {
      continue;
    }
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto input_node = cnode->input(i);
      auto is_graph_input = false;
      if (input_node->isa<Parameter>()) {
        if (!input_node->cast<ParameterPtr>()->has_default()) {
          is_graph_input = true;
        }
      }
      if (!input_node->isa<mindspore::CNode>() && !is_graph_input) {
        continue;
      }
      auto input_cnode_quant_type = schema::QuantType_QUANT_NONE;
      std::shared_ptr<ops::PrimitiveC> input_cnode_primitive_c = nullptr;
      if (!is_graph_input) {
        auto input_cnode = std::dynamic_pointer_cast<mindspore::CNode>(input_node);
        input_cnode_primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(input_cnode->input(0));
        if (input_cnode_primitive_c == nullptr) {
          MS_LOG(DEBUG) << "input: " << i << " " << input_cnode->fullname_with_scope() << ": "
                        << " PrimitiveC is null";
          continue;
        }
        auto input_primitive_quant_holder = GetCNodeQuantHolder(input_cnode_primitive_c);
        MS_ASSERT(input_primitive_quant_holder != nullptr);
        input_cnode_quant_type = input_primitive_quant_holder->quant_type();
      }

      if (curnode_quant_type != input_cnode_quant_type) {
        ValueNodePtr value_node = nullptr;
        if (curnode_quant_type == schema::QuantType_PostTraining &&
            input_cnode_quant_type == schema::QuantType_QUANT_NONE) {
          if (primitive_quant_param_holder->input_quant_params().size() < i) {
            MS_LOG(ERROR) << "quant param is invalid.";
            return RET_ERROR;
          }
          value_node = NewQuantCastValueNode(kNumberTypeFloat32, kNumberTypeInt8,
                                             primitive_quant_param_holder->input_quant_params()[i - 1]);
        } else if (curnode_quant_type == schema::QuantType_QUANT_NONE &&
                   input_cnode_quant_type == schema::QuantType_PostTraining) {
          auto input_primitive_quant_param_holder = GetCNodeQuantHolder(input_cnode_primitive_c);
          value_node = NewQuantCastValueNode(kNumberTypeInt8, kNumberTypeFloat32,
                                             input_primitive_quant_param_holder->output_quant_params().front());
        }
        if (value_node == nullptr) {
          MS_LOG(WARNING) << "value_node is null! "
                          << "cur_node: " << cnode->fullname_with_scope() << " quant_type: "
                          << " input_" << i << ": "
                          << " quant_type:" << input_cnode_quant_type;
          continue;
        }
        std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
        auto quant_cast_cnode = graph->NewCNode(op_inputs);
        quant_cast_cnode->set_fullname_with_scope(cnode->fullname_with_scope() + "_quant_cast_" + std::to_string(i));
        cnode->set_input(i, quant_cast_cnode);
        MS_LOG(DEBUG) << "Add quant cast. "
                      << "cur_node: " << cnode->fullname_with_scope() << " quant_type: " << curnode_quant_type
                      << " input_" << i << ": "
                      << " quant_type:" << input_cnode_quant_type;
      } else {
        MS_LOG(DEBUG) << "No need to add quant cast. "
                      << "cur_node: " << cnode->fullname_with_scope() << " quant_type: " << curnode_quant_type
                      << " input_" << i << ": "
                      << " quant_type:" << input_cnode_quant_type;
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
