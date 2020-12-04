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

#include "src/ops/broadcast_to.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> BroadcastTo::GetDstShape() const { return this->primitive_->value.AsBroadcastTo()->dst_shape; }

void BroadcastTo::SetDstShape(const std::vector<int> &dst_shape) {
  this->primitive_->value.AsBroadcastTo()->dst_shape = dst_shape;
}

#else
int BroadcastTo::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BroadcastTo();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BroadcastTo return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> dst_shape;
  if (attr->dst_shape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->dst_shape()->size()); i++) {
      dst_shape.push_back(attr->dst_shape()->data()[i]);
    }
  }
  auto val_offset = schema::CreateBroadcastToDirect(*fbb, &dst_shape);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BroadcastTo, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> BroadcastTo::GetDstShape() const {
  auto fb_vector = this->primitive_->value_as_BroadcastTo()->dst_shape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

PrimitiveC *BroadcastToCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BroadcastTo>(primitive);
}
Registry BroadcastToRegistry(schema::PrimitiveType_BroadcastTo, BroadcastToCreator);
#endif

namespace {
constexpr int kBroadcastToInputNum = 1;
constexpr int kBroadcastToOnnxInputNum = 2;
constexpr int kBroadcastToOutputNum = 1;
}  // namespace

int BroadcastTo::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (inputs.size() != kBroadcastToInputNum && inputs.size() != kBroadcastToOnnxInputNum) {
    MS_LOG(ERROR) << "input size:" << inputs.size();
    return RET_PARAM_INVALID;
  }
  if (outputs.size() != kBroadcastToOutputNum) {
    MS_LOG(ERROR) << "output size:" << outputs.size();
    return RET_PARAM_INVALID;
  }

  auto input = inputs.at(0);
  outputs[0]->set_format(input->format());
  outputs[0]->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  std::vector<int32_t> dst_shape(GetDstShape());
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    if (dst_shape[i] == -1) {
      dst_shape[i] = inputs[0]->shape()[i];
    }
  }
  auto input_shape = input->shape();
  std::vector<int> shape(dst_shape.size());
  int input_shape_index = input_shape.size() - 1;
  if (input_shape.size() > dst_shape.size()) {
    MS_LOG(ERROR) << "input shape size " << input_shape.size() << " should <= broadcast to shape size "
                  << dst_shape.size() << "!";
    return RET_PARAM_INVALID;
  }

  for (int i = dst_shape.size() - 1; i >= 0; --i) {
    if (dst_shape[i] < 0) {
      MS_LOG(ERROR) << "shape[" << i << "] = " << dst_shape[i] << " ] should be > 0!";
      return RET_PARAM_INVALID;
    }
    if (input_shape_index >= 0) {
      auto dim = input_shape[input_shape_index];
      if (dim != dst_shape[i] && dim != 1) {
        MS_LOG(ERROR) << "Invalid broadcast shape!";
        return RET_PARAM_INVALID;
      }
    }
    shape[i] = dst_shape[i];
    --input_shape_index;
  }
  outputs[0]->set_shape(shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
