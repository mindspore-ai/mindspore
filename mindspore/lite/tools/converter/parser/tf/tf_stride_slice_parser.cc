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
#include "ops/strided_slice.h"

namespace mindspore {
namespace lite {

ops::PrimitiveC *TFStrideSliceParser::Parse(const tensorflow::NodeDef &tf_op,
                                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                            std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::StridedSlice>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "begin_mask", &attr_value)) {
    MS_LOG(ERROR) << "The begin_mask attr should be specified";
    return nullptr;
  }
  prim->set_begin_mask(attr_value.i());

  if (!TensorFlowUtils::FindAttrValue(tf_op, "end_mask", &attr_value)) {
    MS_LOG(ERROR) << "The end_mask attr should be specified";
    return nullptr;
  }
  prim->set_end_mask(attr_value.i());

  if (!TensorFlowUtils::FindAttrValue(tf_op, "ellipsis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The ellipsis_mask attr should be specified";
    return nullptr;
  }
  prim->set_ellipsis_mask(attr_value.i());

  if (!TensorFlowUtils::FindAttrValue(tf_op, "new_axis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The new_axis_mask attr should be specified";
    return nullptr;
  }
  prim->set_new_axis_mask(attr_value.i());

  if (!TensorFlowUtils::FindAttrValue(tf_op, "shrink_axis_mask", &attr_value)) {
    MS_LOG(ERROR) << "The shrink_axis_mask attr should be specified";
    return nullptr;
  }
  prim->set_shrink_axis_mask(attr_value.i());

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size(); i++) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "Add Op input " << i << " failed.";
      return nullptr;
    }
  }

  return prim.release();
}

TFNodeRegistrar g_tfStrideSliceParser("StridedSlice", new TFStrideSliceParser());
}  // namespace lite
}  // namespace mindspore
