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
#include "tools/converter/parser/tf/tf_stride_slice_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"

namespace mindspore {
namespace lite {
STATUS TFStrideSliceParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  PrimitiveC **primitiveC, std::vector<std::string> *inputs, int *output_size) {
  MS_LOG(INFO) << "TF StrideSliceParser";
  if (primitiveC == nullptr || output_size == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_NULL_PTR;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "New PrimitiveT failed";
    return RET_NULL_PTR;
  }
  auto attr = std::make_unique<schema::StridedSliceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return RET_NULL_PTR;
  }

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "begin_mask", &attr_value)) {
    MS_LOG(ERROR) << "The begin_mask attr should be specified";
    return RET_ERROR;
  }
  attr->beginMask = attr_value.i();

  if (!TensorFlowUtils::FindAttrValue(tf_op, "end_mask", &attr_value)) {
    MS_LOG(ERROR) << "The end_mask attr should be specified";
    return RET_ERROR;
  }
  attr->endMask = attr_value.i();

  if (!TensorFlowUtils::FindAttrValue(tf_op, "ellipsis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The ellipsis_mask attr should be specified";
    return RET_ERROR;
  }
  attr->ellipsisMask = attr_value.i();

  if (!TensorFlowUtils::FindAttrValue(tf_op, "new_axis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The new_axis_mask attr should be specified";
    return RET_ERROR;
  }
  attr->newAxisMask = attr_value.i();

  if (!TensorFlowUtils::FindAttrValue(tf_op, "shrink_axis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The shrink_axis_mask attr should be specified";
    return RET_ERROR;
  }
  attr->shrinkAxisMask = attr_value.i();
  primitive->value.type = schema::PrimitiveType_StridedSlice;
  primitive->value.value = attr.release();
  *primitiveC = PrimitiveC::Create(primitive.release());
  if (*primitiveC == nullptr) {
    MS_LOG(ERROR) << "primitiveC is nullptr";
    return RET_ERROR;
  }

  *output_size = 1;
  STATUS status = RET_OK;
  for (int i = 0; i < tf_op.input_size(); i++) {
    status = AddOpInput(tf_op, i, inputs);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Add Op input failed.";
      return status;
    }
  }
  return status;
}
TFNodeRegistrar g_tfStrideSliceParser("StridedSlice", new TFStrideSliceParser());
}  // namespace lite
}  // namespace mindspore
