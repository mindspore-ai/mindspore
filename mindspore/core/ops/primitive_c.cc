/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/primitive_c.h"

#include "abstract/ops/primitive_infer_map.h"
#include "include/robin_hood.h"
#include "ir/value.h"
#include "mindapi/src/helper.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
OpPrimCRegister &OpPrimCRegister::GetInstance() {
  static OpPrimCRegister instance;
  return instance;
}

const HashMap<std::string, OpPrimCDefineFunc> &OpPrimCRegister::GetPrimCMap() const { return op_primc_fns_; }
void OpPrimCRegister::SetPrimCMap(const std::string &kname, const OpPrimCDefineFunc &fn) { op_primc_fns_[kname] = fn; }
}  // namespace ops
}  // namespace mindspore
