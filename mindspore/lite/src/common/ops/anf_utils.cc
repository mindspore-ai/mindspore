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

#include "src/common/ops/anf_utils.h"
#ifdef PRIMITIVE_WRITEABLE
namespace mindspore {
namespace lite {
std::unique_ptr<schema::PrimitiveT> GetPrimitiveT(const AnfNodePtr &node) {
  auto prim = GetValueNode<std::shared_ptr<Primitive>>(node);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "primitive is nullptr";
    return nullptr;
  }

  if (prim->name().empty()) {
    MS_LOG(ERROR) << "the name of primitive is null";
    return nullptr;
  }

  MS_LOG(DEBUG) << "export prim: " << prim->name();
  auto creator = MSOpsRegistry::GetInstance()->GetPrimitiveCreator(prim->name());
  if (creator != nullptr) {
    return creator(prim);
  } else {
    MS_LOG(WARNING) << "can not find MSOpsRegistry for op: " << prim->name();
    return nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore
#endif
