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
#include "tools/converter/parser/tf/tf_ragged_range_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "tools/converter/parser/tf/tf_util.h"
#include "ops/range.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFRaggedRangeParser::Parse(const tensorflow::NodeDef &tf_op,
                                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                            std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Range>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "starts", &attr_value)) {
    prim->set_start(attr_value.i());
  } else {
    auto start_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(0)));
    if (TensorFlowUtils::FindAttrValue(*start_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange start node value attr, means it has default value";
      prim->set_start(attr_value.i());
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "limits", &attr_value)) {
    prim->set_limit(attr_value.i());
  } else {
    auto limit_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(1)));
    if (TensorFlowUtils::FindAttrValue(*limit_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange limit node value attr, means it has default value";
      prim->set_limit(attr_value.i());
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "deltas", &attr_value)) {
    prim->set_delta(attr_value.i());
  } else {
    auto delta_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(2)));
    if (TensorFlowUtils::FindAttrValue(*delta_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange delta node value attr, means it has default value";
    }
    prim->set_delta(attr_value.i());
  }

  *output_size = 1;
  for (int i = 0; i < 3; i++) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "add op input " << i << " failed!";
      return nullptr;
    }
  }

  return prim.release();
}

TFNodeRegistrar g_tfRaggedRangeParser("RaggedRange", new TFRaggedRangeParser());
}  // namespace lite
}  // namespace mindspore
