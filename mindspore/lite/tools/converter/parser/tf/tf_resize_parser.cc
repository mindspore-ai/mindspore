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
#include "tools/converter/parser/tf/tf_resize_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/op_utils.h"
#include "ops/resize.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFResizeParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NHWC));
  prim->set_cubic_coeff(-0.75f);
  if (!TensorFlowUtils::FindAttrValue(tf_op, "align_corners", &attr_value)) {
    MS_LOG(ERROR) << "The align_corners attr should be specified";
    return nullptr;
  }
  if (attr_value.b()) {
    prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ALIGN_CORNERS);
    (void)prim_c->AddAttr("align_corners", MakeValue(true));
  } else if (TensorFlowUtils::FindAttrValue(tf_op, "half_pixel_centers", &attr_value) && attr_value.b()) {
    prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::HALF_PIXEL);
    prim->set_cubic_coeff(-0.5f);
    (void)prim_c->AddAttr("half_pixel_centers", MakeValue(true));
  } else {
    prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ASYMMETRIC);
  }

  if (tf_op.op() == "ResizeBilinear") {
    prim->set_method(mindspore::ResizeMethod::LINEAR);
  } else if (tf_op.op() == "ResizeNearestNeighbor") {
    prim->set_method(mindspore::ResizeMethod::NEAREST);
  } else if (tf_op.op() == "ResizeBicubic") {
    prim->set_method(mindspore::ResizeMethod::CUBIC);
  } else {
    prim->set_method(mindspore::ResizeMethod::UNKNOWN);
  }
  auto size_node = tf_node_map.at(tf_op.input(SECOND_INPUT));
  if (size_node == nullptr) {
    MS_LOG(ERROR) << "Find size input failed.";
    return nullptr;
  }
  if (!TensorFlowUtils::FindAttrValue(tf_op, "value", &attr_value)) {
    MS_LOG(WARNING) << "The value attr should be specified";
  }
  auto tensor_content = attr_value.tensor().tensor_content();
  if (tensor_content.size() >= kInputSize1 * sizeof(int32_t)) {
    auto size_ptr = reinterpret_cast<const int32_t *>(tensor_content.data());
    prim->set_new_height(size_ptr[0]);
    prim->set_new_width(size_ptr[1]);
  }

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "Add Op input failed.";
    return nullptr;
  }

  return prim->GetPrim();
}
TFNodeRegistrar g_tfResizeBilinearParser("ResizeBilinear", new TFResizeParser());
TFNodeRegistrar g_tfResizeNearestNeighborParser("ResizeNearestNeighbor", new TFResizeParser());
TFNodeRegistrar g_tfResizeBicubicParser("ResizeBicubic", new TFResizeParser());
}  // namespace lite
}  // namespace mindspore
