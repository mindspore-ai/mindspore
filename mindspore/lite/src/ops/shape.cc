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

#include "src/ops/shape.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

namespace {
constexpr int kShapeInputNum = 1;
constexpr int kShapeOutputNum = 1;
}  // namespace
int Shape::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != kShapeInputNum) {
    MS_LOG(ERROR) << "inputs to Shape operator should be 1, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (outputs_.size() != kShapeOutputNum) {
    MS_LOG(ERROR) << "outputs to Shape operator should be 1, but " << outputs_.size() << " is given.";
    return RET_ERROR;
  }
  auto in_tensor = inputs_.front();
  auto out_tensor = outputs_.front();
  out_tensor->set_data_type(kNumberTypeInt32);
  out_tensor->set_format(in_tensor->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  std::vector<int> out_shape;
  out_shape.push_back(static_cast<int>(in_tensor->shape().size()));
  out_tensor->set_shape(out_shape);
  return RET_OK;
}
#ifdef PRIMITIVE_WRITEABLE
#else
int Shape::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);

  auto val_offset = schema::CreateShape(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Shape, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *ShapeCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Shape>(primitive); }
Registry ShapeRegistry(schema::PrimitiveType_Shape, ShapeCreator);
#endif

}  // namespace lite
}  // namespace mindspore
