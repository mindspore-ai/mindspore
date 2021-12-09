/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/resize_checker.h"
#include <vector>
#include <string>
#include "include/registry/converter_context.h"

namespace mindspore {
namespace dpico {
bool ResizeChecker::Check(CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = GetValueNode<PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  if (primitive->GetAttr(ops::kFmkType) != nullptr) {
    auto fmk_type = static_cast<converter::FmkType>(GetValue<int>(primitive->GetAttr(ops::kFmkType)));
    if (fmk_type == converter::kFmkTypeCaffe) {
      return true;
    }
  }
  MS_LOG(WARNING) << "resize op only supports caffe for now by dpico. " << op->fullname_with_scope();
  return false;
}

OpCheckerRegistrar g_ResizeChecker("Resize", new ResizeChecker());
}  // namespace dpico
}  // namespace mindspore
