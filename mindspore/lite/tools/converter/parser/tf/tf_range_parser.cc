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
#include "tools/converter/parser/tf/tf_range_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/range.h"

namespace mindspore {
namespace lite {

ops::PrimitiveC *TFRangeParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF RangeParser";
  if (output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return nullptr;
  }

  auto primitive_c = new (std::nothrow) ops::Range;
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "New Range failed";
    return nullptr;
  }

  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "start", &attr_value)) {
    primitive_c->set_start(static_cast<int64_t>(attr_value.i()));
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "limit", &attr_value)) {
    primitive_c->set_limit(static_cast<int64_t>(attr_value.i()));
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "delta", &attr_value)) {
    primitive_c->set_delta(static_cast<int64_t>(attr_value.i()));
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  status |= AddOpInput(tf_op, 1, inputs);
  status |= AddOpInput(tf_op, 2, inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add op input failed!";
    return nullptr;
  }
  return primitive_c;
}

TFNodeRegistrar g_tfRangeParser("Range", new TFRangeParser());
}  // namespace lite
}  // namespace mindspore
