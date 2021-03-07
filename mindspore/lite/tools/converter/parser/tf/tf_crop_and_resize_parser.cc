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
#include "tools/converter/parser/tf/tf_crop_and_resize_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/crop_and_resize.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFCropAndResizeParser::Parse(const tensorflow::NodeDef &tf_op,
                                              const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                              std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::CropAndResize>();

  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "extrapolation_value", &attr_value)) {
    MS_LOG(ERROR) << "The align_corners attr should be specified";
    return nullptr;
  }
  prim->set_extrapolation_value(attr_value.f());

  if (!TensorFlowUtils::FindAttrValue(tf_op, "method", &attr_value)) {
    MS_LOG(ERROR) << "The align_corners attr should be specified";
    return nullptr;
  }
  if (attr_value.s() == "bilinear") {
    prim->set_method(mindspore::ResizeMethod::LINEAR);
  } else if (attr_value.s() == "nearest_neighbor") {
    prim->set_method(mindspore::ResizeMethod::NEAREST);
  } else {
    MS_LOG(ERROR) << "Do not support method: " << attr_value.s();
  }

  *output_size = 1;
  for (int i = 0; i < 4; ++i) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "Add Op input-" << i << " failed.";
      return nullptr;
    }
  }

  return prim.release();
}
TFNodeRegistrar g_tfCropAndResizeParser("CropAndResize", new TFCropAndResizeParser());
}  // namespace lite
}  // namespace mindspore
