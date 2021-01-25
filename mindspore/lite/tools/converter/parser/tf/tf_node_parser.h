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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_NODE_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_NODE_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "tools/converter/parser/tf/tf_util.h"
#include "proto/graph.pb.h"
#include "ops/primitive_c.h"
#include "mindspore/core/utils/check_convert_utils.h"

namespace mindspore {
namespace lite {
class TFNodeParser {
 public:
  TFNodeParser() = default;

  virtual ~TFNodeParser() = default;

  virtual ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
    return nullptr;
  }

  STATUS AddOpInput(const tensorflow::NodeDef &tf_op, const int idx, std::vector<std::string> *inputs);

  const tensorflow::NodeDef *GetConstInputNode(const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                               const std::string &input_name);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TF_NODE_PARSER_H
