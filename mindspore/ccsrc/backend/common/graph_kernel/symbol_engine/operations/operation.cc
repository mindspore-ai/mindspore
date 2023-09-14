/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/hash_set.h"

namespace mindspore::graphkernel::symbol {
namespace ops {
void Operation::Build() {
  MS_EXCEPTION_IF_CHECK_FAIL(output_ == nullptr, "The operation is built.");
  MS_LOG(DEBUG) << ">>> Building operation " << ToString();
  output_ = Eval();
  MS_EXCEPTION_IF_NULL(output_);
  if (!output_->CanUpdate()) {
    need_eval_ = false;
  }
  MS_LOG(DEBUG) << "<<< Build result of [" << name() << "]: " << output_->ToString() << ". need_eval=" << need_eval();
  is_building_ = false;
}
}  // namespace ops
}  // namespace mindspore::graphkernel::symbol
