/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_argmax_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/arg_max_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFArgMaxParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::ArgMaxFusion>();

  tensorflow::AttrValue attr_value;
  auto axis_node = tf_node_map.at(tf_op.input(tf_op.input_size() - 1));
  if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attr value should be specified.";
    return nullptr;
  }
  auto &axis_tensor = attr_value.tensor();
  prim->set_axis(axis_tensor.int_val(0));
  prim->set_out_max_value(false);

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim.release();
}
TFNodeRegistrar g_tfArgMaxParser("ArgMax", new TFArgMaxParser());
}  // namespace lite
}  // namespace mindspore
