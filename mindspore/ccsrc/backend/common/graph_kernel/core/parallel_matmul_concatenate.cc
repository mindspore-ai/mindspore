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

#include "backend/common/graph_kernel/core/parallel_matmul_concatenate.h"
#include "base/base.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
MMAttr GetMatMulTransposeAttr(const CNodePtr &matmul) {
  auto mm_attrs = common::AnfAlgo::GetCNodePrimitive(matmul)->attrs();
  if (mm_attrs.count(kTransposeA) == 0 || mm_attrs.count(kTransposeB) == 0) {
    MS_LOG(WARNING) << "Can not find attr 'transpose_a' or 'transpose_b' in node " << matmul->fullname_with_scope();
    return std::make_pair(false, false);
  }
  auto trans_a = GetValue<bool>(mm_attrs[kTransposeA]);
  auto trans_b = GetValue<bool>(mm_attrs[kTransposeB]);
  return std::make_pair(trans_a, trans_b);
}

CNodePtr NewMatMulNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &matmul_inputs, const CNodePtr &orig_matmul,
                       ShapeVector new_out_shape) {
  auto matmul = func_graph->NewCNode(matmul_inputs);
  func_graph->AddNode(matmul);
  MS_EXCEPTION_IF_NULL(matmul);
  MS_EXCEPTION_IF_NULL(matmul_inputs[1]);
  auto orig_cnode = matmul_inputs[1]->cast<CNodePtr>();
  if (orig_cnode != nullptr && orig_cnode->HasAttr(kOutputsFormat)) {
    auto input_format = GetValue<std::vector<std::string>>(orig_cnode->GetAttr(kOutputsFormat))[0];
    std::vector<std::string> outputs_formats(AnfUtils::GetOutputTensorNum(matmul), input_format);
    matmul->AddAttr(kOutputsFormat, MakeValue(outputs_formats));
  }
  auto [trans_a, trans_b] = GetMatMulTransposeAttr(orig_matmul);
  matmul->AddAttr(kTransposeA, MakeValue(trans_a));
  matmul->AddAttr(kTransposeB, MakeValue(trans_b));
  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(matmul_inputs[1], 0)};
  std::vector<ShapeVector> shapes = {new_out_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, matmul.get());
  matmul->set_kernel_info(std::make_shared<device::KernelInfo>());
  return matmul;
}

BMNK GetBatchMNK(const CNodePtr &matmul) {
  int64_t b = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  auto shape_a = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, kIndex0);
  auto shape_b = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, kIndex1);
  auto [trans_a, trans_b] = GetMatMulTransposeAttr(matmul);
  if (shape_a.size() == kDim3 && shape_b.size() == kDim3 && shape_a[kIndex0] == shape_b[kIndex0]) {
    b = shape_a[kIndex0];
    (void)shape_a.erase(shape_a.begin());
    (void)shape_b.erase(shape_b.begin());
  } else {
    b = 1;
  }
  m = trans_a ? shape_a[kIndex1] : shape_a[kIndex0];
  k = trans_a ? shape_a[kIndex0] : shape_a[kIndex1];
  n = trans_b ? shape_b[kIndex0] : shape_b[kIndex1];
  return std::tuple(b, m, n, k);
}
}  // namespace

ConcatenatePlan ParallelMatMulConcatenater::Analyse(const Group &branches) const {
  ConcatenatePlan target_op_res;
  Branch b0 = branches[kIndex0];
  AnfNodePtr shared_input = b0.GetRootData();
  target_op_res.in_shape = Callback::Instance()->GetOutputInferShape(shared_input, kIndex0);
  auto matmul = b0.GetTargetOp()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul);
  bool is_a_shared = false;
  for (size_t i = 1; i < matmul->size(); ++i) {
    auto in = matmul->input(i);
    if (in == shared_input) {
      is_a_shared = i == kIndex1;
      break;
    }
  }

  auto [trans_a, trans_b] = GetMatMulTransposeAttr(matmul);
  int64_t b = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  std::tie(b, m, n, k) = GetBatchMNK(matmul);
  if (is_a_shared) {
    auto shape_b = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, kIndex1);
    size_t rank_b = shape_b.size();
    auto n_idx = trans_b ? rank_b - kIndex2 : rank_b - kIndex1;
    target_op_res.concat_in_idx = SizeToInt(n_idx);
    target_op_res.split_out_idx = SizeToInt(rank_b - kIndex1);
    int64_t new_n = n * SizeToLong(branches.size());
    if (rank_b == kDim3) {
      target_op_res.out_shape = ShapeVector({b, m, new_n});
    } else {
      target_op_res.out_shape = ShapeVector({m, new_n});
    }
  } else {
    auto shape_a = common::AnfAlgo::GetPrevNodeOutputInferShape(matmul, kIndex0);
    size_t rank_a = shape_a.size();
    auto m_idx = trans_a ? rank_a - kIndex1 : rank_a - kIndex2;
    target_op_res.concat_in_idx = SizeToInt(m_idx);
    target_op_res.split_out_idx = SizeToInt(rank_a - kIndex2);
    auto new_m = m * SizeToLong(branches.size());
    if (rank_a == kDim3) {
      target_op_res.out_shape = ShapeVector({b, new_m, n});
    } else {
      target_op_res.out_shape = ShapeVector({new_m, n});
    }
  }
  return target_op_res;
}

bool ParallelMatMulConcatenater::CanOpsBeCombined(const AnfNodePtr a, const AnfNodePtr b) {
  auto matmul1 = a->cast<CNodePtr>();
  auto matmul2 = b->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul1);
  MS_EXCEPTION_IF_NULL(matmul2);
  auto [trans_a1, trans_b1] = GetMatMulTransposeAttr(matmul1);
  auto [trans_a2, trans_b2] = GetMatMulTransposeAttr(matmul2);
  return trans_a1 == trans_a2 && trans_b1 == trans_b2;
}

bool ParallelMatMulConcatenater::IsSupportedOp(const AnfNodePtr n) {
  if (n == nullptr || n->cast<CNodePtr>() == nullptr) {
    return false;
  }
  auto prim = GetCNodePrimitive(n);
  if (prim == nullptr || unsupported_ops_.count(prim->name())) {
    return false;
  }
  return true;
}

AnfNodePtr ParallelMatMulConcatenater::MakeCombinedOp(const Group &branches) {
  Branch b1 = branches[0];
  AnfNodePtr shared_input = b1.GetRootData();
  auto matmul_op = b1.GetTargetOp()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(matmul_op);
  auto plan = Analyse(branches);
  plans_.push_back(plan);
  auto overall_inputs = ReloadInputs(branches, b1.target_op_pos, shared_input);
  auto matmul = NewMatMulNode(main_graph_, overall_inputs, matmul_op, plan.out_shape);
  MS_EXCEPTION_IF_CHECK_FAIL(AutoUpdateInfo(matmul), "AutoUpdateInfo fail");
  return matmul;
}

bool ParallelMatMulConcatenater::IsArgCompatible(const AnfNodePtr a, const AnfNodePtr b) { return true; }

AnfNodePtr ConcatParallelMatMul(AnfNodePtr root, uint64_t min_num_branches, const std::string &layout,
                                const FuncGraphPtr &func_graph) {
  if (layout == kOpFormat_NCHW) {
    auto res = ParallelMatMulConcatenater(min_num_branches, layout).Combine(root, func_graph);
    return res;
  }
  MS_LOG(WARNING) << "Not supported combine for layout " << layout;
  return root;
}
}  // namespace mindspore::graphkernel
