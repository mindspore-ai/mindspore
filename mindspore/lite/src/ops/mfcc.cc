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

#include "src/ops/mfcc.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
float Mfcc::GetFreqUpperLimit() const { return this->primitive_->value.AsMfcc()->freqUpperLimit; }
float Mfcc::GetFreqLowerLimit() const { return this->primitive_->value.AsMfcc()->freqLowerLimit; }
int Mfcc::GetFilterBankChannelNum() const { return this->primitive_->value.AsMfcc()->filterBankChannelNum; }
int Mfcc::GetDctCoeffNum() const { return this->primitive_->value.AsMfcc()->dctCoeffNum; }

#else
float Mfcc::GetFreqUpperLimit() const { return this->primitive_->value_as_Mfcc()->freqUpperLimit(); }
float Mfcc::GetFreqLowerLimit() const { return this->primitive_->value_as_Mfcc()->freqLowerLimit(); }
int Mfcc::GetFilterBankChannelNum() const { return this->primitive_->value_as_Mfcc()->filterBankChannelNum(); }
int Mfcc::GetDctCoeffNum() const { return this->primitive_->value_as_Mfcc()->dctCoeffNum(); }
int Mfcc::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Mfcc();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Add return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateMfcc(*fbb, attr->freqUpperLimit(), attr->freqLowerLimit(),
                                       attr->filterBankChannelNum(), attr->dctCoeffNum());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Mfcc, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *MfccCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Mfcc>(primitive); }
Registry MfccRegistry(schema::PrimitiveType_Mfcc, MfccCreator);
#endif
int Mfcc::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != 3) {
    MS_LOG(ERROR) << "first input shape is error, which need to be 3 dimensions, but the dimension is "
                  << input_shape.size();
    return RET_ERROR;
  }
  if (inputs_[1]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "second input element num is error, which need only a value, but the number is "
                  << inputs_[1]->ElementsNum();
    return RET_ERROR;
  }
  std::vector<int> output_shape(3);
  output_shape[0] = input_shape[0];
  output_shape[1] = input_shape[1];
  output_shape[2] = GetDctCoeffNum();
  outputs_.front()->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
