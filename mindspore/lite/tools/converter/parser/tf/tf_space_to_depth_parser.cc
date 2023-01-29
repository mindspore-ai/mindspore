/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/tf/tf_space_to_depth_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/space_to_depth.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFSpaceToDepthParser::Parse(const tensorflow::NodeDef &tf_op,
                                          const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                          std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SpaceToDepth>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "block_size", &attr_value)) {
    MS_LOG(ERROR) << "The block_size attr should be specified";
    return nullptr;
  }
  prim->set_block_size(attr_value.i());
  if (!TensorFlowUtils::FindAttrValue(tf_op, "data_format", &attr_value)) {
    MS_LOG(ERROR) << "The data_format attr should be specified";
    return nullptr;
  }
  mindspore::Format format = mindspore::Format::NHWC;
  if (attr_value.s() == "NCHW") {
    format = mindspore::Format::NCHW;
  } else if (attr_value.s() != "NHWC") {
    MS_LOG(WARNING) << "Unsupported data format: " << attr_value.s() << "which would be treated as NHWC.";
  }
  prim->set_format(format);

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    inputs->emplace_back(tf_op.input(i));
  }
  return prim->GetPrim();
}
TFNodeRegistrar g_tfSpaceToDepthParser("SpaceToDepth", new TFSpaceToDepthParser());
}  // namespace lite
}  // namespace mindspore
