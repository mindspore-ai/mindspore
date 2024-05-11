/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "ops/ops_func_impl/embedding_table_import.h"

#include <memory>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/ops/op_infer.h"
#include "ops/nn_ops.h"

namespace mindspore::ops {
int32_t EmbeddingTableImportFuncImpl::SpecifiedCheckValidation(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  return OP_CHECK_SUCCESS;
}
}  // namespace mindspore::ops
