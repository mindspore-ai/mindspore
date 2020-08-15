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

#include "c_ops/elu.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
float Elu::GetAlpha() const { return this->primitive->value.AsElu()->alpha; }

void Elu::SetAlpha(float alpha) { this->primitive->value.AsElu()->alpha = alpha; }

#else

float Elu::GetAlpha() const { return this->primitive->value_as_Elu()->alpha(); }

void Elu::SetAlpha(float alpha) {}
#endif
}  // namespace mindspore
