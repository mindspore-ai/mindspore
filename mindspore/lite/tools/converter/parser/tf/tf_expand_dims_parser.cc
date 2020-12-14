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
#include "tools/converter/parser/tf/tf_expand_dims_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFExpandDimsParser::Parse(const tensorflow::NodeDef &tf_op,
                                 const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                 PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF ExpandDimsParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::ExpandDimsT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  auto axis_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "Find ExpandDims input axis failed";
    return RET_ERROR;
  }
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The value attr should be specified";
    return RET_ERROR;
  }
  auto tensor_proto = attr_value.tensor();
  if (tensor_proto.int_val_size() > 0) {
    attr->dim = tensor_proto.int_val(0);
  } else {
    attr->dim = (reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data()))[0];
  }

  primitive->value.type = schema::PrimitiveType_ExpandDims;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  auto status = AddOpInput(tf_op, 0, inputs);
  return status;
}
TFNodeRegistrar g_tfExpandDimsParser("ExpandDims", new TFExpandDimsParser());
}  // namespace lite
}  // namespace mindspore
