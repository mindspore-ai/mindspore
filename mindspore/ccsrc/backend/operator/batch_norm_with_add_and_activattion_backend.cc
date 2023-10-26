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

#include "abstract/dshape.h"
#include "backend/operator/ops_backend_def.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/batch_norm.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore::ops {
class MIND_API BatchNormWithAddAndActivationFuncImpl : public BatchNormFuncImpl {
 public:
  std::set<int64_t> GetValueDependArgIndices() const override { return {6, 7, 8, 9}; }

 protected:
  size_t GetAttrPosZero() const override { return 6; }
};

auto gBatchNormWithAddAndActivationFuncImpl = BatchNormWithAddAndActivationFuncImpl();
OpDef gBatchNormWithAddAndActivation = {
  .name_ = kNameBatchNormWithAddAndActivation,
  .args_ =
    {
      {.arg_name_ = "x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "scale", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "bias", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "mean", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "var", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "z", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "is_training", .arg_dtype_ = DT_BOOL, .as_init_arg_ = 1},
      {.arg_name_ = "epsilon", .arg_dtype_ = DT_FLOAT, .as_init_arg_ = 1},
      {.arg_name_ = "momentum", .arg_dtype_ = DT_FLOAT, .as_init_arg_ = 1},
      {.arg_name_ = "data_format", .arg_dtype_ = DT_INT, .as_init_arg_ = 1},
    },
  .returns_ =
    {
      {.arg_name_ = "output_x", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "batch_mean", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "batch_variance", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "reserve_space_1", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
      {.arg_name_ = "reserve_space_2", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0},
    },
  .indexes_ =
    {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
      {"mean", 3},
      {"variance", 4},
      {"is_training", 5},
      {"epsilon", 6},
      {"momentum", 7},
      {"data_format", 8},
    },
  .func_impl_ = gBatchNormWithAddAndActivationFuncImpl,
};
REGISTER_PRIMITIVE_OP_DEF(kNameBatchNormWithAddAndActivation, &gBatchNormWithAddAndActivation);
}  // namespace mindspore::ops
