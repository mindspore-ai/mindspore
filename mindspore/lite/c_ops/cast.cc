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

#include "c_ops/cast.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Cast::GetSrcT() const { return this->primitive->value.AsCast()->srcT; }
int Cast::GetDstT() const { return this->primitive->value.AsCast()->dstT; }

void Cast::SetSrcT(int src_t) { this->primitive->value.AsCast()->srcT = src_t; }
void Cast::SetDstT(int dst_t) { this->primitive->value.AsCast()->dstT = dst_t; }

#else

int Cast::GetSrcT() const { return this->primitive->value_as_Cast()->srcT(); }
int Cast::GetDstT() const { return this->primitive->value_as_Cast()->dstT(); }

void Cast::SetSrcT(int src_t) {}
void Cast::SetDstT(int dst_t) {}
#endif
int Cast::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
    return 1;
  }

  MS_ASSERT(cast_prim != nullptr);
  if (input->data_type() != GetSrcT()) {
    MS_LOG(ERROR) << "input dataType is error";
    return 1;
  }
  if (kSupportDataType.find(input->data_type()) == kSupportDataType.end()) {
    MS_LOG(ERROR) << "Unsupport input data type " << input->data_type();
    return 1;
  }
  if (GetDstT() != kNumberTypeFloat && GetDstT() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid output datatype " << GetDstT();
    return 1;
  }
  output->SetFormat(input->GetFormat());
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  return 0;
}
}  // namespace mindspore
