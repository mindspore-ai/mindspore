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
#include "tools/converter/parser/tf/tf_range_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFRangeParser::Parse(const tensorflow::NodeDef &tf_op,
                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map, PrimitiveC **primitiveC,
                            std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF RangeParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::RangeT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "starts", &attr_value)) {
    attr->start = static_cast<int32_t>(attr_value.i());
  } else {
    auto start_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(0)));
    if (TensorFlowUtils::FindAttrValue(*start_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange start node value attr, means it has default value";
      attr->start = static_cast<int32_t>(attr_value.i());
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "limits", &attr_value)) {
    attr->limit = static_cast<int32_t>(attr_value.i());
  } else {
    auto limit_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(1)));
    if (TensorFlowUtils::FindAttrValue(*limit_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange limit node value attr, means it has default value";
      attr->limit = static_cast<int32_t>(attr_value.i());
    }
  }

  if (TensorFlowUtils::FindAttrValue(tf_op, "deltas", &attr_value)) {
    attr->delta = static_cast<int32_t>(attr_value.i());
  } else {
    auto delta_node = tf_node_map.at(TensorFlowUtils::GetFlattenNodeName(tf_op.input(2)));
    if (TensorFlowUtils::FindAttrValue(*delta_node, "value", &attr_value)) {
      MS_LOG(INFO) << "Found raggedrange delta node value attr, means it has default value";
    }
    attr->delta = static_cast<int32_t>(attr_value.i());
  }

  primitive->value.type = schema::PrimitiveType_Range;
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
  status = AddOpInput(tf_op, 1, inputs);
  if (status != RET_OK) {
    return status;
  }
  status = AddOpInput(tf_op, 2, inputs);
  return status;
}
TFNodeRegistrar g_tfRangeParser("Range", new TFRangeParser());
}  // namespace lite
}  // namespace mindspore
