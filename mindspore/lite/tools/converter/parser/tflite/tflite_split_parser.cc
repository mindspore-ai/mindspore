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

#include "tools/converter/parser/tflite/tflite_split_parser.h"
#include <vector>
#include <memory>
#include <map>
#include "ops/split.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteSplitParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize1, nullptr);
  auto prim = std::make_unique<ops::Split>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  const auto &tflite_attr = tflite_op->builtin_options.AsSplitOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op split attr failed";
    return nullptr;
  }
  auto num_splits = tflite_attr->num_splits;
  const auto &shape_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(1));
  if (shape_tensor == nullptr) {
    MS_LOG(ERROR) << "shape_tensor is null";
    return nullptr;
  }
  const auto tensor_shape = shape_tensor->shape;
  const auto &axis_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(0));
  if (axis_tensor == nullptr) {
    MS_LOG(ERROR) << "axis_tensor is null";
    return nullptr;
  }
  auto &axis_buf_data = tflite_model->buffers.at(axis_tensor->buffer);
  if (axis_buf_data == nullptr) {
    MS_LOG(ERROR) << "buf_data is null";
    return nullptr;
  }
  auto axis = *(reinterpret_cast<int32_t *>(axis_buf_data->data.data()));
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= static_cast<int32_t>(tensor_shape.size())) {
    MS_LOG(ERROR) << "axis value is too large";
    return nullptr;
  }
  prim->set_axis(axis);
  MS_CHECK_TRUE_MSG(num_splits != 0, nullptr, "split num should not be 0.");
  MS_CHECK_GE(static_cast<int>(tensor_shape.size()), axis + 1, nullptr);
  if (tensor_shape.at(axis) % num_splits != 0 && tensor_shape.at(axis) / num_splits != 0) {
    MS_LOG(ERROR) << "num_splits can't divide tensor's length at axis " << axis;
    return nullptr;
  }
  prim->set_output_num(num_splits);
  std::vector<int64_t> size_splits;
  if (tensor_shape[axis] / num_splits != 0) {
    for (int i = 0; i < num_splits; i++) {
      size_splits.push_back(tensor_shape[axis] / num_splits);
    }
  }
  prim->set_size_splits(size_splits);

  return prim.release();
}

TfliteNodeRegister g_tfliteSplitParser(tflite::BuiltinOperator_SPLIT, new TfliteSplitParser());
}  // namespace lite
}  // namespace mindspore
