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
#include "plugin/device/ascend/optimizer/ge/fused_cast_add.h"
#include <memory>
#include <vector>
#include <string>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t selected_cast_input_index = 1;
constexpr size_t replaced_add_input_index = 2;
constexpr auto kCast16 = "cast_16";
constexpr auto kAdd32 = "add_fp32";
constexpr auto kVar16 = "bf16_fp16_var";
constexpr auto kAddWithoutCast = "add_without_cast";
constexpr auto kVar32 = "fp32_var";
constexpr auto kVar = "generic_var";
}  // namespace

bool FusedCastAdd::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto ms_context = MsContext::GetInstance();
  return ms_context->get_param<bool>(MS_CTX_ENABLE_FUSED_CAST_ADD_OPT);
}

AnfNodePtr SelectAddInput(const PatternMap &m, const AnfNodePtr &default_node) {
  MS_EXCEPTION_IF_NULL(default_node);
  const auto &cast_node = m.Get(kCast16);
  const auto &add_node = m.Get(kAdd32);
  MS_EXCEPTION_IF_NULL(cast_node);
  MS_EXCEPTION_IF_NULL(add_node);
  auto cast_cnode = cast_node->cast<CNodePtr>();
  auto add_cnode = add_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cast_cnode);
  MS_EXCEPTION_IF_NULL(add_cnode);
  const auto selected_input = cast_cnode->input(selected_cast_input_index);
  add_cnode->set_input(replaced_add_input_index, selected_input);
  return add_cnode;
}

bool checking_type_16(const BaseRef &node) {
  if (!utils::isa<AnfNodePtr>(node)) {
    return false;
  }
  AnfNodePtr anf_node = utils::cast<AnfNodePtr>(node);
  MS_EXCEPTION_IF_NULL(anf_node);
  TypePtr data_type;
  if (anf_node->isa<Parameter>()) {
    ParameterPtr para_node = anf_node->cast<ParameterPtr>();
    data_type = para_node->Type();
  } else if (anf_node->isa<CNode>()) {
    CNodePtr cnode = anf_node->cast<CNodePtr>();
    data_type = cnode->Type();
  } else {
    return false;
  }
  MS_EXCEPTION_IF_NULL(data_type);
  if (data_type->isa<TensorType>()) {
    data_type = data_type->cast<TensorTypePtr>()->element();
  }
  auto type_id = data_type->type_id();
  if (type_id == kNumberTypeFloat16 || type_id == kNumberTypeBFloat16) {
    return true;
  }
  return false;
}

bool checking_type_32(const BaseRef &node) {
  if (!utils::isa<AnfNodePtr>(node)) {
    return false;
  }
  AnfNodePtr anf_node = utils::cast<AnfNodePtr>(node);
  MS_EXCEPTION_IF_NULL(anf_node);
  TypePtr data_type;
  if (anf_node->isa<Parameter>()) {
    ParameterPtr para_node = anf_node->cast<ParameterPtr>();
    data_type = para_node->Type();
  } else if (anf_node->isa<CNode>()) {
    CNodePtr cnode = anf_node->cast<CNodePtr>();
    data_type = cnode->Type();
  } else {
    return false;
  }
  MS_EXCEPTION_IF_NULL(data_type);
  if (data_type->isa<TensorType>()) {
    data_type = data_type->cast<TensorTypePtr>()->element();
  }
  auto type_id = data_type->type_id();
  if (type_id == kNumberTypeFloat32) {
    return true;
  }
  return false;
}
void FusedCastAdd::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern)
    .AddVar(kVar16, checking_type_16)
    .AddVar(kVar32, checking_type_32)
    .AddVar(kVar)
    .AddCNode(kCast16, {prim::kPrimCast, kVar16, kVar})
    .AddCNode(kAdd32, {prim::kPrimAdd, kVar32, kCast16});
}

void FusedCastAdd::DefineDstPattern(DstPattern *dst_pattern) {
  (void)(*dst_pattern).AddCNode(kAddWithoutCast, {prim::kPrimAdd, kVar32, kVar16}, SelectAddInput);
}
}  // namespace opt
}  // namespace mindspore
