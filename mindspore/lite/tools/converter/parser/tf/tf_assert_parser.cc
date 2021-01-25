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
#include "tools/converter/parser/tf/tf_assert_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/assert.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFAssertParser::Parse(const tensorflow::NodeDef &tf_op,
                                       const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                       std::vector<std::string> *inputs, int *output_size) {
  auto primitive_c = new (std::nothrow) ops::Assert;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "New Assert failed";
    return nullptr;
  }

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "summarize", &attr_value)) {
    MS_LOG(ERROR) << "The keep_dims attr should be specified";
    return nullptr;
  }
  primitive_c->set_summarize((int64_t)(attr_value.i()));

  *output_size = 0;  // Assert not have output
  for (int i = 0; i < tf_op.input_size(); ++i) {
    auto status = AddOpInput(tf_op, i, inputs);
    if (status != RET_OK) {
      return nullptr;
    }
  }

  return primitive_c;
}

TFNodeRegistrar g_tfAssertParser("Assert", new TFAssertParser());
}  // namespace lite
}  // namespace mindspore
