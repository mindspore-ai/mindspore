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

namespace mindspore {
namespace lite {
STATUS TFPadParser::Parse(const tensorflow::NodeDef &tf_op,
                          const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                          std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF PadParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::PadT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }
  if (tf_op.op() == "Pad") {
    attr->paddingMode = schema::PaddingMode_CONSTANT;
    attr->constantValue = 0.0f;

  } else if (tf_op.op() == "MirrorPad") {
    tensorflow::AttrValue attr_value;
    if (!TensorFlowUtils::FindAttrValue(tf_op, "mode", &attr_value)) {
      MS_LOG(ERROR) << "The axis attr should be specified";
      return RET_ERROR;
    }

    if (attr_value.s() == "SYMMETRIC") {
      attr->paddingMode = schema::PaddingMode_SYMMETRIC;
    } else if (attr_value.s() == "REFLECT") {
      attr->paddingMode = schema::PaddingMode_REFLECT;
    } else {
      MS_LOG(ERROR) << "padding mode:" << attr_value.s() << " don't support";
      return RET_ERROR;
    }
  }
  primitive->value.type = schema::PrimitiveType_Pad;
  primitive->value.value = attr.release();

  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return status;
  }
  status = AddOpInput(tf_op, 1, inputs);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return status;
  }
  return status;
}
TFNodeRegistrar g_tfPadParser("Pad", new TFPadParser());
TFNodeRegistrar g_tfMirrorPadParser("MirrorPad", new TFPadParser());
}  // namespace lite
}  // namespace mindspore
