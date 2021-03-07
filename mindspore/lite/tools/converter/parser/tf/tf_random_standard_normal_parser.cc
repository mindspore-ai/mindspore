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
#include "tools/converter/parser/tf/tf_random_standard_normal_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/random_standard_normal.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFRandomStandardNormalParser::Parse(const tensorflow::NodeDef &tf_op,
                                                     const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                                     std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::RandomStandardNormal>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "seed", &attr_value)) {
    MS_LOG(ERROR) << "The seed attr should be specified";
    return nullptr;
  }
  prim->set_seed(attr_value.i());
  if (!TensorFlowUtils::FindAttrValue(tf_op, "seed2", &attr_value)) {
    MS_LOG(ERROR) << "The seed2 attr should be specified";
    return nullptr;
  }
  prim->set_seed2(attr_value.i());

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}
TFNodeRegistrar g_tfRandomStandardNormalParser("RandomStandardNormal", new TFRandomStandardNormalParser());
}  // namespace lite
}  // namespace mindspore
