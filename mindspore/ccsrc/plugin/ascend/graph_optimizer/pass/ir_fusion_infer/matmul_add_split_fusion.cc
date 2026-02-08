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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/matmul_add_split_fusion.h"
#include <memory>
#include <vector>
#include "backend/common/pass/common/gllo_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "include/backend/common/pass_manager/helper.h"

namespace mindspore {
namespace opt {
std::vector<std::string> MatmulAddSplitFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret = {prim::kPrimMatMul->name(), prim::kPrimSplitWithSize->name(), prim::kPrimAdd->name()};
  return ret;
}

const BaseRef MatmulAddSplitFusion::DefinePattern() const {
  auto matmul_ref = GetMatmulPattern();
  auto add_input = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(add_input != nullptr, {});
  auto is_add = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimAdd>);
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto add_ref = VectorRef({is_add, matmul_ref, add_input});
  auto split_with_size_ref = GetSplitWithSizePattern(add_ref);
  return split_with_size_ref;
}
}  // namespace opt
}  // namespace mindspore
