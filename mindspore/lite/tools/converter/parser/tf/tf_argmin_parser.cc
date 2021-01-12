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
#include "tools/converter/parser/tf/tf_argmin_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFArgMinParser::Parse(const tensorflow::NodeDef &tf_op,
                             const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                             std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(DEBUG) << "TF ArgMinParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::ArgMinT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }
  tensorflow::AttrValue attr_value;
  auto axis_node = tf_node_map.at(tf_op.input(tf_op.input_size() - 1));
  if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attr value should be specified.";
    return RET_ERROR;
  }
  auto &axis_tensor = attr_value.tensor();
  attr->axis = axis_tensor.int_val(0);
  attr->outMaxValue = false;
  primitive->value.type = schema::PrimitiveType_ArgMin;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }
  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  if (status != RET_OK) {
    return status;
  }
  return RET_OK;
}
TFNodeRegistrar g_tfArgMinParser("ArgMin", new TFArgMinParser());
}  // namespace lite
}  // namespace mindspore
