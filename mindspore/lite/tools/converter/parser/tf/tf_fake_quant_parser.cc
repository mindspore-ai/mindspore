/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_fake_quant_parser.h"
#include "nnacl/op_base.h"
#include "tools/converter/ops/ops_def.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFFakeQuantParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<FakeQuantWithMinMaxVars>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  MS_CHECK_GE(tf_op.input_size(), kInputSize2, nullptr);
  tensorflow::AttrValue attr_value;
  // min param
  auto min_node = GetConstInputNode(tf_node_map, tf_op.input(SECOND_INPUT));
  if (min_node == nullptr) {
    MS_LOG(ERROR) << "Find FakeQuant input min node failed.";
    return nullptr;
  }
  if (!TensorFlowUtils::FindAttrValue(*min_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attribute min should be specified.";
    return nullptr;
  }
  auto min_value = attr_value.tensor().float_val(0);

  // max param
  auto max_node = GetConstInputNode(tf_node_map, tf_op.input(THIRD_INPUT));
  if (max_node == nullptr) {
    MS_LOG(ERROR) << "Find FakeQuant input max node failed.";
    return nullptr;
  }
  if (!TensorFlowUtils::FindAttrValue(*max_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attribute max should be specified.";
    return nullptr;
  }
  auto max_value = attr_value.tensor().float_val(0);

  prim->AddAttr("min", MakeValue(min_value));
  prim->AddAttr("max", MakeValue(max_value));

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add op input failed.";
    return nullptr;
  }
  return prim;
}

TFNodeRegistrar g_tfFakeQuantParser("FakeQuantWithMinMaxVars", new TFFakeQuantParser());
}  // namespace lite
}  // namespace mindspore
