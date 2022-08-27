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

  bool narrow_range = false;
  if (ParseNarrowRange(tf_op, &narrow_range) != RET_OK) {
    MS_LOG(ERROR) << "parse narrow_range failed";
    return nullptr;
  }
  prim->AddAttr("narrow_range", MakeValue(narrow_range));

  int num_bits = 8;
  if (ParseNumBits(tf_op, &num_bits) != RET_OK) {
    MS_LOG(ERROR) << "parse num_bits failed";
    return nullptr;
  }
  prim->AddAttr("num_bits", MakeValue(num_bits));

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add op input failed.";
    return nullptr;
  }
  return prim;
}

STATUS TFFakeQuantParser::ParseNumBits(const tensorflow::NodeDef &node_def, int *num_bits) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "num_bits", &attr_value)) {
    MS_LOG(ERROR) << "The attr num_bits should be specified";
    return RET_ERROR;
  }
  *num_bits = attr_value.i();
  return RET_OK;
}

STATUS TFFakeQuantParser::ParseNarrowRange(const tensorflow::NodeDef &node_def, bool *narrow_range) {
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(node_def, "narrow_range", &attr_value)) {
    MS_LOG(ERROR) << "The attr narrow_range should be specified";
    return RET_ERROR;
  }
  *narrow_range = attr_value.b();
  return RET_OK;
}

TFNodeRegistrar g_tfFakeQuantParser("FakeQuantWithMinMaxVars", new TFFakeQuantParser());
}  // namespace lite
}  // namespace mindspore
