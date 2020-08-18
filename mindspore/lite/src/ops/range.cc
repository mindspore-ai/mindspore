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

#include "src/ops/range.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Range::GetDType() const { return this->primitive->value.AsRange()->dType; }
int Range::GetStart() const { return this->primitive->value.AsRange()->start; }
int Range::GetLimit() const { return this->primitive->value.AsRange()->limit; }
int Range::GetDelta() const { return this->primitive->value.AsRange()->delta; }

void Range::SetDType(int d_type) { this->primitive->value.AsRange()->dType = d_type; }
void Range::SetStart(int start) { this->primitive->value.AsRange()->start = start; }
void Range::SetLimit(int limit) { this->primitive->value.AsRange()->limit = limit; }
void Range::SetDelta(int delta) { this->primitive->value.AsRange()->delta = delta; }

#else

int Range::GetDType() const { return this->primitive->value_as_Range()->dType(); }
int Range::GetStart() const { return this->primitive->value_as_Range()->start(); }
int Range::GetLimit() const { return this->primitive->value_as_Range()->limit(); }
int Range::GetDelta() const { return this->primitive->value_as_Range()->delta(); }

void Range::SetDType(int d_type) {}
void Range::SetStart(int start) {}
void Range::SetLimit(int limit) {}
void Range::SetDelta(int delta) {}
#endif

int Range::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  auto range_prim = this->primitive->value_as_Range();
  MS_ASSERT(range_prim != nullptr);
  int shape_size = std::ceil(static_cast<float>(range_prim->limit() - range_prim->start()) / range_prim->delta());
  std::vector<int> in_shape(1);
  in_shape.push_back(shape_size);
  output->set_shape(in_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
