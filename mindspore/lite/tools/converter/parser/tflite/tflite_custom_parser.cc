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

#include "tools/converter/parser/tflite/tflite_custom_parser.h"
#include <vector>
#include <memory>
#include <map>
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/flexbuffers.h"

namespace mindspore {
namespace lite {
STATUS TfliteCustomParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                 const std::vector<std::unique_ptr<tflite::TensorT>> &tflite_tensors,
                                 const std::vector<std::unique_ptr<tflite::BufferT>> &tflite_model_buffer,
                                 schema::CNodeT *op, std::vector<int32_t> *tensors_id,
                                 std::vector<schema::Format> *tensors_format, std::map<int, int> *tensors_id_map) {
  MS_LOG(DEBUG) << "parse TfliteCustomParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::DetectionPostProcessT> attr = std::make_unique<schema::DetectionPostProcessT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const auto &custom_attr = tflite_op->custom_options;
  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  attr->format = schema::Format_NHWC;
  attr->inputSize = tflite_op->inputs.size();
  attr->hScale = attr_map["h_scale"].AsFloat();
  attr->wScale = attr_map["w_scale"].AsFloat();
  attr->xScale = attr_map["x_scale"].AsFloat();
  attr->yScale = attr_map["y_scale"].AsFloat();
  attr->NmsIouThreshold = attr_map["nms_iou_threshold"].AsFloat();
  attr->NmsScoreThreshold = attr_map["nms_score_threshold"].AsFloat();
  attr->MaxDetections = attr_map["max_detections"].AsInt32();
  if (attr_map["detections_per_class"].IsNull()) {
    attr->DetectionsPreClass = 100;
  } else {
    attr->DetectionsPreClass = attr_map["detections_per_class"].AsInt32();
  }
  attr->MaxClassesPreDetection = attr_map["max_classes_per_detection"].AsInt32();
  attr->NumClasses = attr_map["num_classes"].AsInt32();
  if (attr_map["use_regular_nms"].IsNull()) {
    attr->UseRegularNms = false;
  } else {
    attr->UseRegularNms = attr_map["use_regular_nms"].AsBool();
  }
  if (attr_map["_output_quantized"].IsNull()) {
    attr->OutQuantized = false;
  } else {
    attr->OutQuantized = attr_map["_output_quantized"].AsBool();
  }

  op->primitive->value.type = schema::PrimitiveType_DetectionPostProcess;
  op->primitive->value.value = attr.release();

  for (size_t i = 0; i < tflite_op->inputs.size(); ++i) {
    AddOpInput(op, tensors_id, tensors_format, tensors_id_map, tflite_op->inputs[i], tensors_id->size(),
               tflite_tensors.size(), schema::Format_NHWC);
  }
  for (size_t i = 0; i < tflite_op->outputs.size(); ++i) {
    AddOpOutput(op, tensors_id, tensors_format, tensors_id_map, tflite_op->outputs[i], tensors_id->size(),
                tflite_tensors.size(), schema::Format_NHWC);
  }
  return RET_OK;
}

TfliteNodeRegister g_tfliteCustomParser("Custom", new TfliteCustomParser());
}  // namespace lite
}  // namespace mindspore
