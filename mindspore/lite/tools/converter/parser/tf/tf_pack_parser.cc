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
#include "tools/converter/parser/tf/tf_pack_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/stack.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFPackParser::Parse(const tensorflow::NodeDef &tf_op,
                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Stack>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "axis", &attr_value)) {
    MS_LOG(ERROR) << "The axis attr should be specified";
    return nullptr;
  }
  prim->set_axis({attr_value.i()});

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); ++i) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "add op input " << i << " failed";
      return nullptr;
    }
  }

  return prim.release();
}

TFNodeRegistrar g_tfPackParser("Pack", new TFPackParser());
}  // namespace lite
}  // namespace mindspore
