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

#include "c_ops/fused_batchnorm.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
float FusedBatchNorm::GetEpsilon() const { return this->primitive->value.AsFusedBatchNorm()->epsilon; }
float FusedBatchNorm::GetMomentum() const { return this->primitive->value.AsFusedBatchNorm()->momentum; }
int FusedBatchNorm::GetSpatial() const { return this->primitive->value.AsFusedBatchNorm()->spatial; }

void FusedBatchNorm::SetEpsilon(float epsilon) { this->primitive->value.AsFusedBatchNorm()->epsilon = epsilon; }
void FusedBatchNorm::SetMomentum(float momentum) { this->primitive->value.AsFusedBatchNorm()->momentum = momentum; }
void FusedBatchNorm::SetSpatial(int spatial) { this->primitive->value.AsFusedBatchNorm()->spatial = spatial; }

#else

float FusedBatchNorm::GetEpsilon() const { return this->primitive->value_as_FusedBatchNorm()->epsilon(); }
float FusedBatchNorm::GetMomentum() const { return this->primitive->value_as_FusedBatchNorm()->momentum(); }
int FusedBatchNorm::GetSpatial() const { return this->primitive->value_as_FusedBatchNorm()->spatial(); }

void FusedBatchNorm::SetEpsilon(float epsilon) {}
void FusedBatchNorm::SetMomentum(float momentum) {}
void FusedBatchNorm::SetSpatial(int spatial) {}
#endif
}  // namespace mindspore
