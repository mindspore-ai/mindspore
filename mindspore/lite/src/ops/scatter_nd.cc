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

#include "src/ops/scatter_nd.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

namespace {
constexpr int kScatterNDInputNum = 3;
constexpr int kScatterNDOutputNum = 1;
constexpr int kScatterShapeIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;
}  // namespace
int ScatterND::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
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
  if (output == nullptr) {
    MS_LOG(ERROR) << "output null pointer dereferencing.";
    return RET_ERROR;
  }
  output->set_data_type(update->data_type());
  output->set_format(update->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto shape_data = reinterpret_cast<int *>(shape->MutableData());
  std::vector<int> out_shape(shape_data, shape_data + shape->ElementsNum());
  output->set_shape(out_shape);
  return RET_OK;
}
#ifdef PRIMITIVE_WRITEABLE
#else
int ScatterND::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto val_offset = schema::CreateScatterND(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ScatterND, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif

}  // namespace lite
}  // namespace mindspore
