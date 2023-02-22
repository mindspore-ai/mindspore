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

#include "ops/pack.h"
#include "ops/stack_comm.h"
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class AGStackInfer;

void Pack::set_axis(const int64_t &axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Pack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Pack::Init(const int64_t &axis) { this->set_axis(axis); }

MIND_API_OPERATOR_IMPL(Pack, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Pack, prim::kPrimPack, AGStackInfer, false);
}  // namespace ops
}  // namespace mindspore
