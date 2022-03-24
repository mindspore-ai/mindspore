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

#include <string>
#include <vector>
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "checker/batchnorm_checker.h"

namespace mindspore {
namespace dpico {
bool BatchNormChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(WARNING) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive_c is nullptr." << op->fullname_with_scope();
    return false;
  }
  if (primitive->GetAttr(kUseGlobalStats) != nullptr) {
    auto use_global_stats = api::GetValue<bool>(primitive->GetAttr(kUseGlobalStats));
    if (!use_global_stats) {
      MS_LOG(WARNING) << "use global stats attr is false, which is not supported by dpico. "
                      << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_BatchNormChecker("BatchNorm", new BatchNormChecker());
OpCheckerRegistrar g_FusedBatchNormChecker("FusedBatchNorm", new BatchNormChecker());
}  // namespace dpico
}  // namespace mindspore
