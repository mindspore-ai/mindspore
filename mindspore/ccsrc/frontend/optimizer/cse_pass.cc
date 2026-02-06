/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/cse_pass.h"

#include "include/utils/pipeline/cse.h"
#include "include/frontend/optimizer/optimizer.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {

CSEPass::CSEPass(bool report_changes) : CSE(), report_changes_(report_changes) {}

bool CSEPass::operator()(const FuncGraphPtr &root, const OptimizerPtr &optimizer) {
  bool chg = Cse(root, optimizer->resource()->manager());
  return chg && report_changes_;
}

}  // namespace opt
}  // namespace mindspore
