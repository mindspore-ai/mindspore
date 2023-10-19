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

#include "ops/op_def.h"
#include "ops/ops_func_impl/baddbmm.h"

namespace mindspore::ops {
BaddbmmFuncImpl gBaddbmmFuncImpl;
OpDef gBaddbmm = {
  .name_ = "Baddbmm",
  .args_ =
    {
      {.arg_name_ = "input", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0, .arg_handler_ = "", .cast_dtype_ = {}},
      {.arg_name_ = "batch1", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0, .arg_handler_ = "", .cast_dtype_ = {}},
      {.arg_name_ = "batch2", .arg_dtype_ = DT_TENSOR, .as_init_arg_ = 0, .arg_handler_ = "", .cast_dtype_ = {}},
      {.arg_name_ = "beta", .arg_dtype_ = DT_NUMBER, .as_init_arg_ = 1, .arg_handler_ = "", .cast_dtype_ = {}},
      {.arg_name_ = "alpha", .arg_dtype_ = DT_NUMBER, .as_init_arg_ = 1, .arg_handler_ = "", .cast_dtype_ = {}},
    },
  .returns_ =
    {
      {.arg_name_ = "output", .arg_dtype_ = DT_TENSOR},
    },
  .indexes_ =
    {
      {"input", 0},
      {"batch1", 1},
      {"batch2", 2},
      {"beta", 3},
      {"alpha", 4},
    },
  .func_impl_ = &gBaddbmmFuncImpl,
};

std::unordered_map<std::string, OpDefPtr> gOpDefTable = {{"Baddbmm", &gBaddbmm}};
}  // namespace mindspore::ops
