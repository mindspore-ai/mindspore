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
#include "ops/ops_func_impl/embedding_table_export.h"

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/ops/op_infer.h"
#include "ops/op_utils.h"
#include "ops/nn_ops.h"

namespace mindspore::ops {
int32_t EmbeddingTableExportFuncImpl::SpecifiedCheckValidation(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto export_mode_opt = GetScalarValue<std::string>(primitive->GetAttr("export_mode"));
  if (MS_UNLIKELY(!export_mode_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto export_mode = export_mode_opt.value();
  const std::set<std::string> valid_modes{"all", "old", "new", "sepcifiednew"};
  if (MS_UNLIKELY(valid_modes.find(export_mode) == valid_modes.end())) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", does not support Non-bin file.";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace mindspore::ops
