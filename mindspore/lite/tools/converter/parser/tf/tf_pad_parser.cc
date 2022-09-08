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
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kInputIndexTwo = 2;
constexpr int kInputSizeThree = 3;
}  // namespace
PrimitiveCPtr TFPadParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::PadFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  if (tf_op.op() == "Pad") {
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    prim->set_constant_value(0.0f);
    (void)prim_c->AddAttr(ops::kOriginalOpName, MakeValue("Pad"));
  } else if (tf_op.op() == "PadV2") {
    prim->set_padding_mode(mindspore::PaddingMode::CONSTANT);
    if (tf_op.input_size() < kInputSizeThree) {
      MS_LOG(ERROR) << "tf padv2 input size less than 3, which is " << tf_op.input_size();
      return nullptr;
    }
    auto &const_value_name = tf_op.input(kInputIndexTwo);
    if (tf_node_map.find(const_value_name) == tf_node_map.end()) {
      MS_LOG(ERROR) << "cannot find the input.";
      return nullptr;
    }
    tensorflow::AttrValue attr_value;
    if (!TensorFlowUtils::FindAttrValue(*tf_node_map.at(const_value_name), "value", &attr_value)) {
      MS_LOG(ERROR) << "the input may be not const, which is not support now.";
      return nullptr;
    }
    auto &tensor_proto = attr_value.tensor();
    if (tensor_proto.dtype() != tensorflow::DT_FLOAT) {
      MS_LOG(ERROR) << "input data type only support float now.";
      return nullptr;
    }
    prim->set_constant_value(tensor_proto.float_val(0));
    (void)prim_c->AddAttr(ops::kOriginalOpName, MakeValue("PadV2"));
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
    (void)prim_c->AddAttr(ops::kOriginalOpName, MakeValue("MirrorPad"));
  }

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}
TFNodeRegistrar g_tfPadParser("Pad", new TFPadParser());
TFNodeRegistrar g_tfPadV2Parser("PadV2", new TFPadParser());
TFNodeRegistrar g_tfMirrorPadParser("MirrorPad", new TFPadParser());
}  // namespace lite
}  // namespace mindspore
