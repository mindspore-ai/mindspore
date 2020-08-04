/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
namespace {
constexpr int kScatterNDInputNum = 3;
constexpr int kScatterNDOutputNum = 1;
constexpr int kScatterShapeIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;
}  // namespace

int ScatterND::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  if (inputs_.size() != kScatterNDInputNum) {
    MS_LOG(ERROR) << "inputs number is not equal to " << kScatterNDInputNum;
    return RET_ERROR;
  }
  if (outputs_.size() != kScatterNDOutputNum) {
    MS_LOG(ERROR) << "outputs number is not equal to " << kScatterNDInputNum;
    return RET_ERROR;
  }
  auto shape = inputs_.at(kScatterShapeIndex);
  if (shape == nullptr) {
    MS_LOG(ERROR) << "shape null pointer dereferencing.";
    return RET_ERROR;
  }
  auto indices = inputs_.at(kScatterIndicesIndex);
  if (indices == nullptr) {
    MS_LOG(ERROR) << "indices null pointer dereferencing.";
    return RET_ERROR;
  }
  auto update = inputs_.at(kScatterUpdateIndex);
  if (update == nullptr) {
    MS_LOG(ERROR) << "update null pointer dereferencing.";
    return RET_ERROR;
  }
  auto output = outputs_.front();
  auto shape_data = reinterpret_cast<int *>(shape->Data());
  std::vector<int> out_shape(shape_data, shape_data + shape->DataSize());
  output->set_shape(out_shape);
  output->set_data_type(update->data_type());
  output->SetFormat(update->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
