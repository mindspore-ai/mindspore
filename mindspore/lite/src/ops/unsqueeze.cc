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

#include "src/ops/unsqueeze.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Unsqueeze::GetAxis() const { return this->primitive_->value.AsUnsqueeze()->axis; }

void Unsqueeze::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsUnsqueeze()->axis = axis; }

#else
bool predicate(int n) { return n != 1; }
std::vector<int> Unsqueeze::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_Unsqueeze()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Unsqueeze::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Unsqueeze();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Unsqueeze return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateUnsqueezeDirect(*fbb, &axis);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Unsqueeze, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *UnsqueezeCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Unsqueeze>(primitive);
}
Registry UnsqueezeRegistry(schema::PrimitiveType_Unsqueeze, UnsqueezeCreator);

#endif

int Unsqueeze::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size is invalid";
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output size is invalid";
  }
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto dims = GetAxis();
  auto in_shape = input->shape();
  auto in_rank = in_shape.size();
  auto dim_rank = GetAxis().size();
  std::vector<int> out_shape;
  if (dim_rank == 0) {
    for (auto d : in_shape) {
      if (d != 1) {
        out_shape.push_back(d);
      }
    }
  } else {
    auto sz = in_rank + dim_rank;
    size_t in_itr = 0;
    size_t ax_itr = 0;
    for (size_t i = 0; i < sz; i++) {
      if (ax_itr < dim_rank && dims.at(ax_itr) == static_cast<int>(i)) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else if (ax_itr < dim_rank && dims.at(ax_itr) + sz == i) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else {
        out_shape.emplace_back(in_shape.at(in_itr));
        in_itr++;
      }
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
