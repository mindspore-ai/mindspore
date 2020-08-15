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

#include "c_ops/local_response_normalization.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int LocalResponseNormalization::GetDepthRadius() const {
  return this->primitive->value.AsLocalResponseNormalization()->depth_radius;
}
float LocalResponseNormalization::GetBias() const {
  return this->primitive->value.AsLocalResponseNormalization()->bias;
}
float LocalResponseNormalization::GetAlpha() const {
  return this->primitive->value.AsLocalResponseNormalization()->alpha;
}
float LocalResponseNormalization::GetBeta() const {
  return this->primitive->value.AsLocalResponseNormalization()->beta;
}

void LocalResponseNormalization::SetDepthRadius(int depth_radius) {
  this->primitive->value.AsLocalResponseNormalization()->depth_radius = depth_radius;
}
void LocalResponseNormalization::SetBias(float bias) {
  this->primitive->value.AsLocalResponseNormalization()->bias = bias;
}
void LocalResponseNormalization::SetAlpha(float alpha) {
  this->primitive->value.AsLocalResponseNormalization()->alpha = alpha;
}
void LocalResponseNormalization::SetBeta(float beta) {
  this->primitive->value.AsLocalResponseNormalization()->beta = beta;
}

#else

int LocalResponseNormalization::GetDepthRadius() const {
  return this->primitive->value_as_LocalResponseNormalization()->depth_radius();
}
float LocalResponseNormalization::GetBias() const {
  return this->primitive->value_as_LocalResponseNormalization()->bias();
}
float LocalResponseNormalization::GetAlpha() const {
  return this->primitive->value_as_LocalResponseNormalization()->alpha();
}
float LocalResponseNormalization::GetBeta() const {
  return this->primitive->value_as_LocalResponseNormalization()->beta();
}

void LocalResponseNormalization::SetDepthRadius(int depth_radius) {}
void LocalResponseNormalization::SetBias(float bias) {}
void LocalResponseNormalization::SetAlpha(float alpha) {}
void LocalResponseNormalization::SetBeta(float beta) {}
#endif
}  // namespace mindspore
