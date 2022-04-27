/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/expression/ops/depend.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
DependM::DependM() : Node() {
  auto param = calloc(1, sizeof(OpParameter));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate parameter";
    return;
  }
  SetOpParam(param);
  set_primitive(schema::PrimitiveType_Depend);
  set_name(UniqueName("Depend"));
}
namespace NN {
Node *Depend() {
  auto d = new (std::nothrow) DependM();
  if (d == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate depend object";
    return nullptr;
  }
  return d;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
