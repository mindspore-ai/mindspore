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

#include "src/ops/nchw2nhwc.h"
#include "src/common/common.h"

#include "src/ops/ops_register.h"
#include "nnacl/transpose.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
#else
int Nchw2Nhwc::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateNchw2Nhwc(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Nchw2Nhwc, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *Nchw2NhwcCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Nchw2Nhwc>(primitive);
}
Registry Nchw2NhwcRegistry(schema::PrimitiveType_Nchw2Nhwc, Nchw2NhwcCreator);
#endif

OpParameter *PopulateNchw2NhwcParameter(const mindspore::lite::PrimitiveC *primitive) {
  TransposeParameter *parameter = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(parameter, 0, sizeof(OpParameter));
  parameter->op_parameter_.type_ = primitive->Type();
  parameter->num_axes_ = 4;
  parameter->perm_[0] = 0;
  parameter->perm_[1] = 2;
  parameter->perm_[2] = 3;
  parameter->perm_[3] = 1;
  return reinterpret_cast<OpParameter *>(parameter);
}
Registry Nchw2NhwcParameterRegistry(schema::PrimitiveType_Nchw2Nhwc, PopulateNchw2NhwcParameter);

int Nchw2Nhwc::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->SetFormat(schema::Format::Format_NHWC);
  output->set_data_type(input->data_type());
  if (!GetInferFlag()) {
    return RET_OK;
  }
  std::vector<int> nchw_shape = input->shape();
  if (nchw_shape.size() != 4) {
    output->set_shape(nchw_shape);
  } else {
    std::vector<int> nhwc_shape{nchw_shape};
    nhwc_shape[NHWC_N] = nchw_shape[NCHW_N];
    nhwc_shape[NHWC_H] = nchw_shape[NCHW_H];
    nhwc_shape[NHWC_W] = nchw_shape[NCHW_W];
    nhwc_shape[NHWC_C] = nchw_shape[NCHW_C];
    output->set_shape(nhwc_shape);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
