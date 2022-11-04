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

#include "checker/squeeze_checker.h"
#include <vector>
#include <string>
#include "common/op_attr.h"
#include "common/anf_util.h"

namespace mindspore {
namespace dpico {
bool SqueezeChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format) {
  std::vector<ShapeVector> output_shapes;
  if (GetBoolAttr(op, dpico::kInferDone)) {
    if (GetOutputShapesFromCNode(op, &output_shapes) != RET_OK) {
      MS_LOG(ERROR) << "get node shape failed. " << op->fullname_with_scope();
      return false;
    }
    if (output_shapes.size() != 1) {
      MS_LOG(ERROR) << "squeeze should have single output, but in fact it has " << output_shapes.size() << " outputs.";
      return false;
    }
    auto output_shape = output_shapes.at(0);
    return output_shape.size() > 1;
  }
  return true;
}

OpCheckerRegistrar g_SqueezeChecker("Squeeze", new SqueezeChecker());
}  // namespace dpico
}  // namespace mindspore
