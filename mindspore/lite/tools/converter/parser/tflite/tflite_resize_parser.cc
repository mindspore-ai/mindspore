/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tflite/tflite_resize_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ops/resize.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteResizeParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize1, nullptr);
  auto prim = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_format(mindspore::Format::NHWC);
  prim->set_preserve_aspect_ratio(false);
  prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ASYMMETRIC);
  prim->set_cubic_coeff(-0.75f);
  prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ASYMMETRIC);
  auto &opcode = tflite_model->operator_codes[tflite_op->opcode_index];
  MS_CHECK_TRUE_RET(opcode != nullptr, nullptr);
  auto tflite_op_type = opcode->builtin_code;
  if (tflite_op_type == tflite::BuiltinOperator_RESIZE_BILINEAR) {
    MS_LOG(DEBUG) << "parse TfliteResizeBilinearParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeBilinearOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op ResizeBilinear attr failed";
      return nullptr;
    }
    if (tfliteAttr->align_corners) {
      prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ALIGN_CORNERS);
    }
    if (tfliteAttr->half_pixel_centers) {
      prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::HALF_PIXEL);
      prim->set_cubic_coeff(-0.5f);
    }
    prim->set_method(mindspore::ResizeMethod::LINEAR);
  } else if (tflite_op_type == tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR) {
    MS_LOG(DEBUG) << "parse TfliteResizeNearestNeighborParser";
    const auto &tfliteAttr = tflite_op->builtin_options.AsResizeNearestNeighborOptions();
    if (tfliteAttr == nullptr) {
      MS_LOG(ERROR) << "get op ResizeNearestNeighbor attr failed";
      return nullptr;
    }
    if (tfliteAttr->align_corners) {
      prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ALIGN_CORNERS);
    }
    if (tfliteAttr->half_pixel_centers) {
      prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::HALF_PIXEL);
      prim->set_cubic_coeff(-0.5f);
      (void)prim->AddAttr("half_pixel_centers", api::MakeValue(true));
    }
    prim->set_method(mindspore::ResizeMethod::NEAREST);
    prim->set_nearest_mode(mindspore::NearestMode::NORMAL);
  } else {
    MS_LOG(ERROR) << "wrong resize type";
    return nullptr;
  }

  std::vector<int64_t> dims;
  auto ret = GetTfliteData(tflite_op->inputs[SECOND_INPUT], tflite_subgraph->tensors, tflite_model->buffers, &dims);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get axes value failed.";
    return nullptr;
  }

  if (dims.size() > 1) {
    prim->set_new_height(dims.at(0));
    prim->set_new_width(dims.at(1));
  }

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteResizeBilinearParser(tflite::BuiltinOperator_RESIZE_BILINEAR, new TfliteResizeParser());
TfliteNodeRegister g_tfliteResizeNearestNeighborParser(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                                                       new TfliteResizeParser());
}  // namespace lite
}  // namespace mindspore
