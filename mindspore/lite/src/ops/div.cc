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

#include "src/ops/div.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Div::GetActivationType() const { return this->primitive->value.AsDiv()->activationType; }

void Div::SetActivationType(int activation_type) {
  this->primitive->value.AsDiv()->activationType = (schema::ActivationType)activation_type;
}

#else

int Div::GetActivationType() const { return this->primitive->value_as_Div()->activationType(); }

void Div::SetActivationType(int activation_type) {}
#endif
}  // namespace lite
}  // namespace mindspore
