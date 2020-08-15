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

#include "c_ops/lrn.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
float Lrn::GetAlpha() const { return this->primitive->value.AsLrn()->alpha; }
float Lrn::GetBeta() const { return this->primitive->value.AsLrn()->beta; }
float Lrn::GetBias() const { return this->primitive->value.AsLrn()->bias; }
int Lrn::GetSize() const { return this->primitive->value.AsLrn()->size; }

void Lrn::SetAlpha(float alpha) { this->primitive->value.AsLrn()->alpha = alpha; }
void Lrn::SetBeta(float beta) { this->primitive->value.AsLrn()->beta = beta; }
void Lrn::SetBias(float bias) { this->primitive->value.AsLrn()->bias = bias; }
void Lrn::SetSize(int size) { this->primitive->value.AsLrn()->size = size; }

#else

float Lrn::GetAlpha() const { return this->primitive->value_as_Lrn()->alpha(); }
float Lrn::GetBeta() const { return this->primitive->value_as_Lrn()->beta(); }
float Lrn::GetBias() const { return this->primitive->value_as_Lrn()->bias(); }
int Lrn::GetSize() const { return this->primitive->value_as_Lrn()->size(); }

void Lrn::SetAlpha(float alpha) {}
void Lrn::SetBeta(float beta) {}
void Lrn::SetBias(float bias) {}
void Lrn::SetSize(int size) {}
#endif
}  // namespace mindspore
