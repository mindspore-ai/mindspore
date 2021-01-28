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
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "ops/range.h"
#include "tools/converter/parser/tf/tf_ragged_range_parser.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFRaggedRangeParser::Parse(const tensorflow::NodeDef &tf_op,
                                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                            std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF RaggedRangeParser";
  if (output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return nullptr;
  }

  auto prim = std::make_unique<ops::Range>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "starts", &attr_value)) {
    MS_LOG(ERROR) << "The starts attr should be specified";
    return nullptr;
  }
  prim->set_start(static_cast<int64_t>(attr_value.i()));

  if (!TensorFlowUtils::FindAttrValue(tf_op, "limits", &attr_value)) {
    MS_LOG(ERROR) << "The limits attr should be specified";
    return nullptr;
  }
  prim->set_limit(static_cast<int64_t>(attr_value.i()));

  if (!TensorFlowUtils::FindAttrValue(tf_op, "deltas", &attr_value)) {
    MS_LOG(ERROR) << "The deltas attr should be specified";
    return nullptr;
  }
  prim->set_delta(static_cast<int64_t>(attr_value.i()));

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "add op input is failed!";
    return nullptr;
  }
  return prim.release();
}

TFNodeRegistrar g_tfRaggedRangeParser("RaggedRange", new TFRaggedRangeParser());
}  // namespace lite
}  // namespace mindspore
