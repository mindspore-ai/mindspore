/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/addext_cast_rms_norm_cast_fusion.h"
#include <memory>
#include <vector>
#include "utils/ms_context.h"
#include "include/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
std::vector<std::string> AddExtCastRmsNormCastFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret = {prim::kPrimRmsNorm->name(), prim::kPrimCast->name(), prim::kPrimAddExt->name()};
  return ret;
}

const BaseRef AddExtCastRmsNormCastFusion::DefinePattern() const {
  VarPtr index0 = std::make_shared<CondVar>(IsConstant);
  VarPtr x0 = std::make_shared<Var>();
  VarPtr x1 = std::make_shared<Var>();
  VarPtr value = std::make_shared<Var>();
  VectorRef add_cast = VectorRef({prim::kPrimCast, VectorRef({prim::kPrimAddExt, x1_, x2_, value}), x0});
  VectorRef add_cast_rms_norm = VectorRef({prim::kPrimRmsNorm, add_cast, gamma_, eps_});
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_cast_rms_norm, index0});
  VectorRef add_cast_rms_norm_cast = VectorRef({prim::kPrimCast, tuple_get_item_0, x1});
  return add_cast_rms_norm_cast;
}
}  // namespace opt
}  // namespace mindspore
