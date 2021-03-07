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
#include "tools/converter/parser/tf/tf_pad_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/pad_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFPadParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::PadFusion>();

  if (tf_op.op() == "Pad") {
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    prim->set_constant_value(0.0f);

  } else if (tf_op.op() == "MirrorPad") {
    tensorflow::AttrValue attr_value;
    if (!TensorFlowUtils::FindAttrValue(tf_op, "mode", &attr_value)) {
      MS_LOG(ERROR) << "The axis attr should be specified";
      return nullptr;
    }

    if (attr_value.s() == "SYMMETRIC") {
      prim->set_padding_mode(mindspore::PaddingMode::SYMMETRIC);
    } else if (attr_value.s() == "REFLECT") {
      prim->set_padding_mode(mindspore::PaddingMode::REFLECT);
    } else {
      MS_LOG(ERROR) << "padding mode:" << attr_value.s() << " don't support";
      return nullptr;
    }
  }

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim.release();
}
TFNodeRegistrar g_tfPadParser("Pad", new TFPadParser());
TFNodeRegistrar g_tfMirrorPadParser("MirrorPad", new TFPadParser());
}  // namespace lite
}  // namespace mindspore
