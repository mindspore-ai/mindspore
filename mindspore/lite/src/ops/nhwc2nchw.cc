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

#include "src/ops/nhwc2nchw.h"
#include "src/common/common.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE
#else
int Nhwc2Nchw::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateNhwc2Nchw(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Nhwc2Nchw, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *Nhwc2NchwCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Nhwc2Nchw>(primitive);
}
Registry Nhwc2NchwRegistry(schema::PrimitiveType_Nhwc2Nchw, Nhwc2NchwCreator);
#endif

int Nhwc2Nchw::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_format(schema::Format::Format_NCHW);
  output->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  std::vector<int> nhwc_shape = input->shape();
  if (nhwc_shape.size() != 4) {
    output->set_shape(nhwc_shape);
  } else {
    std::vector<int> nchw_shape{nhwc_shape};
    nchw_shape[NCHW_N] = nhwc_shape[NHWC_N];
    nchw_shape[NCHW_C] = nhwc_shape[NHWC_C];
    nchw_shape[NCHW_H] = nhwc_shape[NHWC_H];
    nchw_shape[NCHW_W] = nhwc_shape[NHWC_W];
    output->set_shape(nchw_shape);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
