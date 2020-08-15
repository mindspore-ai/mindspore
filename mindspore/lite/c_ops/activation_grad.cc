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

#include "c_ops/activation_grad.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int ActivationGrad::GetType() const { return this->primitive->value.AsActivationGrad()->type; }

void ActivationGrad::SetType(int type) {
  this->primitive->value.AsActivationGrad()->type = (schema::ActivationGradType)type;
}

#else

int ActivationGrad::GetType() const { return this->primitive->value_as_ActivationGrad()->type(); }

void ActivationGrad::SetType(int type) {}
#endif
}  // namespace mindspore
