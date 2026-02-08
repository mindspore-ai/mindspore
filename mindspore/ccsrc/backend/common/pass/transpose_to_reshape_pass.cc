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
#include "backend/common/pass/transpose_to_reshape_pass.h"
#include <vector>
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr auto kInput = "input";
constexpr auto kPermute = "permute";
constexpr auto kShape = "shape";
constexpr auto kTranspose = "transpose";
constexpr auto kReshape = "reshape";

AnfNodePtr BuildShape(const PatternMap &m) {
  auto transpose = m.Get(kTranspose);
  MS_EXCEPTION_IF_NULL(transpose);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(transpose, 0);
  auto func_graph = transpose->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto permute = m.Get(kPermute);
  MS_EXCEPTION_IF_NULL(permute);
  if (permute->isa<ValueNode>()) {
    kernel_graph->RemoveValueNodeFromGraph(permute->cast<ValueNodePtr>());
  }
  auto shape_value = CreateShapeValueNode(func_graph, output_shape);
  return shape_value;
}

AnfNodePtr BuildReshape(const PatternMap &m, const AnfNodePtr &reshape) {
  MS_EXCEPTION_IF_NULL(reshape);
  auto transpose = m.Get(kTranspose);
  MS_EXCEPTION_IF_NULL(transpose);
  reshape->set_abstract(transpose->abstract());
  reshape->set_scope(transpose->scope());
  return reshape;
}
}  // namespace

bool TransposeToReshapePass::CheckMatchedDAG(const PatternMap &m, const FuncGraphPtr &,
                                             const AnfNodePtr &transpose) const {
  MS_EXCEPTION_IF_NULL(transpose);
  if (common::AnfAlgo::IsDynamicShape(transpose)) {
    return false;
  }

  // 1. get input shape and permute list
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(transpose, 0);
  auto permute_node = m.Get(kPermute);
  auto permute_vnode = permute_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(permute_vnode);
  auto permute_value = permute_vnode->value();
  auto permute = GetValue<std::vector<int64_t>>(permute_value);
  if (input_shape.size() != permute.size()) {
    MS_LOG(EXCEPTION) << "Expect same dimension size of input shape and permute input of node "
                      << transpose->fullname_with_scope() << ", but got input shape dim size [" << input_shape.size()
                      << "], permute dim size [" << permute.size() << "].";
  }
  // 2. remove shape==1 dimension in permute list
  std::transform(permute.cbegin(), permute.cend(), permute.begin(),
                 [&input_shape](int64_t dim) { return dim < 0 ? dim + SizeToLong(input_shape.size()) : dim; });
  permute.erase(std::remove_if(permute.begin(), permute.end(),
                               [&input_shape](int64_t dim) { return input_shape.at(LongToSize(dim)) == 1; }),
                permute.end());
  // 3. check if rest dimension in permute list is ascending
  for (size_t i = 1; i < permute.size(); ++i) {
    if (permute[i - 1] >= permute[i]) {
      return false;
    }
  }
  return true;
}

void TransposeToReshapePass::DefineSrcPattern(SrcPattern *src_pattern) {
  (void)(*src_pattern).AddVar(kInput).AddVar(kPermute).AddCNode(kTranspose, {prim::kPrimTranspose, kInput, kPermute});
}

void TransposeToReshapePass::DefineDstPattern(DstPattern *dst_pattern) {
  (void)(*dst_pattern)
    .AddValueNode(kShape, BuildShape)
    .AddCNode(kReshape, {prim::kPrimReshape, kInput, kShape}, BuildReshape);
}
}  // namespace opt
}  // namespace mindspore
