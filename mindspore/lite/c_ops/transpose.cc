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

#include "c_ops/transpose.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Transpose::GetPerm() const { return this->primitive->value.AsTranspose()->perm; }
bool Transpose::GetConjugate() const { return this->primitive->value.AsTranspose()->conjugate; }

void Transpose::SetPerm(const std::vector<int> &perm) { this->primitive->value.AsTranspose()->perm = perm; }
void Transpose::SetConjugate(bool conjugate) { this->primitive->value.AsTranspose()->conjugate = conjugate; }

#else

std::vector<int> Transpose::GetPerm() const {
  auto fb_vector = this->primitive->value_as_Transpose()->perm();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
bool Transpose::GetConjugate() const { return this->primitive->value_as_Transpose()->conjugate(); }

void Transpose::SetPerm(const std::vector<int> &perm) {}
void Transpose::SetConjugate(bool conjugate) {}
#endif
int Transpose::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  MS_ASSERT(inputs_.size() == kSingleNum);
  MS_ASSERT(outputs_.size() == kSingleNum);

  int conjugate = GetConjugate();
  if (conjugate) {
    MS_LOG(ERROR) << "Transpose conjugate is not support currently";
    return 1;
  }
  std::vector<int> perm;
  perm.insert(perm.begin(), GetPerm().begin(), GetPerm().end());

  std::vector<int> in_shape = input->shape();
  std::vector<int> out_shape;
  out_shape.resize(perm.size());
  for (int i = 0; i < perm.size(); ++i) {
    out_shape[i] = in_shape[perm[i]];
  }

  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace mindspore
