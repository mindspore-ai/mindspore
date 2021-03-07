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
#include "tools/converter/parser/tf/tf_linspace_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "ops/lin_space.h"
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFLinSpaceParser::Parse(const tensorflow::NodeDef &tf_op,
                                         const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                         std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(DEBUG) << "TF LinSpaceParser";
  auto prim = std::make_unique<ops::LinSpace>();
  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); ++i) {
    auto status = AddOpInput(tf_op, i, inputs);
    if (status != RET_OK) {
      return nullptr;
    }
  }
  return prim.release();
}
TFNodeRegistrar g_tfLinSpaceParser("LinSpace", new TFLinSpaceParser());
}  // namespace lite
}  // namespace mindspore
