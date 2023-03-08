/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/insert_pad.h"
#include <string>
#include <tuple>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace graphkernel {
namespace {
using vec = std::vector<int64_t>;
constexpr size_t MAX_PER_DIM_SHAPE = 4096;
constexpr int64_t MAX_ALL_SHAPE = static_cast<int64_t>(3e10);

// M,N pad 32, K pad 16
const auto GetPadShape = [](size_t K, size_t M, size_t N) {
  size_t pad_K = ((K - 1) / 16 + 1) * 16;
  size_t pad_M = ((M - 1) / 32 + 1) * 32;
  size_t pad_N = ((N - 1) / 32 + 1) * 32;
  return std::tuple(pad_K, pad_M, pad_N);
};

using KMNLIST = std::tuple<size_t, size_t, size_t, size_t, size_t, size_t>;
KMNLIST GetTransShape(bool trans_a, bool trans_b, const vec &shape_a, const vec &shape_b, vec *pad_a, vec *pad_b) {
  size_t k, m, n, pad_k, pad_m, pad_n;
  size_t size = shape_a.size();
  k = LongToSize(trans_a ? shape_a[size - kIndex2] : shape_a[size - kIndex1]);
  m = LongToSize(trans_a ? shape_a[size - kIndex1] : shape_a[size - kIndex2]);
  n = LongToSize(trans_b ? shape_b[size - kIndex2] : shape_b[size - kIndex1]);
  std::tie(pad_k, pad_m, pad_n) = GetPadShape(k, m, n);
  if (trans_a) {
    pad_a->push_back(pad_k);
    pad_a->push_back(pad_m);
  } else {
    pad_a->push_back(pad_m);
    pad_a->push_back(pad_k);
  }
  if (trans_b) {
    pad_b->push_back(pad_n);
    pad_b->push_back(pad_k);
  } else {
    pad_b->push_back(pad_k);
    pad_b->push_back(pad_n);
  }
  return std::tuple(k, m, n, pad_k, pad_m, pad_n);
}

bool IsAkgMatMul(size_t K, size_t M, size_t N) {
  if (K > MAX_PER_DIM_SHAPE ||
      (static_cast<int64_t>(M) * static_cast<int64_t>(N) * static_cast<int64_t>(K)) >= MAX_ALL_SHAPE) {
    return false;
  }
  return true;
}

// Return ture if (K, M, N) need pad
std::tuple<bool, bool, bool> NeedPad(const CNodePtr &matmul, vec *pad_shape_a, vec *pad_shape_b, vec *unpad_shape,
                                     vec *tail_shape_a, vec *tail_shape_b, vec *tail_shape_unpad) {
  auto mm_attrs = common::AnfAlgo::GetCNodePrimitive(matmul)->attrs();
  if (mm_attrs.count("transpose_a") == 0 || mm_attrs.count("transpose_b") == 0) {
    MS_LOG(ERROR) << "Can not find attr 'transpose_a' or 'transpose_b' in node " << matmul->fullname_with_scope();
    return std::tuple(false, false, false);
  }
  auto tran_a = GetValue<bool>(mm_attrs["transpose_a"]);
  auto tran_b = GetValue<bool>(mm_attrs["transpose_b"]);
  auto shape_a = AnfAlgo::GetInputDeviceShape(matmul, 0);
  auto shape_b = AnfAlgo::GetInputDeviceShape(matmul, 1);
  auto size_a = shape_a.size();
  for (size_t dim = 0; dim < size_a - 2; ++dim) {
    pad_shape_a->push_back(shape_a[dim]);
    pad_shape_b->push_back(shape_a[dim]);
    unpad_shape->push_back(shape_a[dim]);
    tail_shape_a->push_back(0);
    tail_shape_b->push_back(0);
    tail_shape_unpad->push_back(0);
  }

  size_t K, M, N, pad_K, pad_M, pad_N;
  std::tie(K, M, N, pad_K, pad_M, pad_N) = GetTransShape(tran_a, tran_b, shape_a, shape_b, pad_shape_a, pad_shape_b);
  // Donot Pad for cublas operator
  if (!IsAkgMatMul(K, M, N)) {
    SetNodeAttrSafely("Akg", MakeValue(false), matmul);
    return std::tuple(false, false, false);
  }
  SetNodeAttrSafely("Akg", MakeValue(true), matmul);
  unpad_shape->push_back(M);
  unpad_shape->push_back(N);
  tail_shape_unpad->push_back(pad_M - M);
  tail_shape_unpad->push_back(pad_N - N);
  tail_shape_a->push_back(pad_shape_a->at(size_a - 2) - shape_a[size_a - 2]);
  tail_shape_a->push_back(pad_shape_a->at(size_a - 1) - shape_a[size_a - 1]);
  tail_shape_b->push_back(pad_shape_b->at(size_a - 2) - shape_b[size_a - 2]);
  tail_shape_b->push_back(pad_shape_b->at(size_a - 1) - shape_b[size_a - 1]);
  return std::tuple(pad_K != K, pad_M != M, pad_N != N);
}

// Insert pad for A if left is true, insert pad for B if left is false
void InsertPad(const CNodePtr &matmul, const FuncGraphPtr &func_graph, bool left, const vec &pad_shape,
               const vec &tail_shape) {
  size_t input_index = left ? 1 : 2;
  AnfNodePtrList pad_inp = {NewValueNode(std::make_shared<Primitive>("PadAkg")), matmul->input(input_index)};
  auto pad_cnode = func_graph->NewCNode(pad_inp);
  func_graph->AddNode(pad_cnode);

  ShapeVector tail;
  (void)tail.insert(tail.cbegin(), tail_shape.cbegin(), tail_shape.cend());
  ShapeVector head(tail_shape.size(), 0);

  SetNodeAttrSafely("head", MakeValue(head), pad_cnode);
  SetNodeAttrSafely("tail", MakeValue(tail), pad_cnode);
  SetNodeAttrSafely("pad_val", MakeValue(std::make_shared<Int32Imm>(0)), pad_cnode);
  std::vector<TypeId> pad_type = {common::AnfAlgo::GetPrevNodeOutputInferDataType(matmul, 0)};

  ShapeVector abs_shape;
  (void)abs_shape.insert(abs_shape.cbegin(), pad_shape.cbegin(), pad_shape.cend());
  auto abs_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(abs_shape));
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(pad_type[0]), abs_shape_ptr);
  pad_cnode->set_abstract(abstract);

  pad_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(matmul);
  std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(matmul);
  std::vector<std::string> pad_inp_formats = {input_formats.front()};
  std::vector<TypeId> pad_inp_types = {input_types.front()};
  std::vector<std::string> pad_output_formats = {input_formats.front()};
  std::vector<TypeId> output_types = {input_types.front()};
  auto graph_sel_info = BuildSelectKernelBuildInfo(pad_inp_formats, pad_inp_types, pad_output_formats, output_types);
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, pad_cnode.get());

  matmul->set_input(input_index, pad_cnode);
}

// unpad_shape is [batch, M, N], tail_shape is [0, pad_M - M, pad_N - N]
void InsertUnpad(const CNodePtr &matmul, const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                 const vec &unpad_shape, const vec &tail_shape) {
  AnfNodePtrList unpad_inp = {NewValueNode(std::make_shared<Primitive>("UnPadAkg")), matmul};
  auto unpad_cnode = func_graph->NewCNode(unpad_inp);
  func_graph->AddNode(unpad_cnode);
  ShapeVector tail;
  (void)tail.insert(tail.cbegin(), tail_shape.cbegin(), tail_shape.cend());
  SetNodeAttrSafely("tail", MakeValue(tail), unpad_cnode);
  std::vector<TypeId> unpad_type = {common::AnfAlgo::GetOutputInferDataType(matmul, 0)};

  ShapeVector abs_shape;
  (void)abs_shape.insert(abs_shape.cbegin(), unpad_shape.cbegin(), unpad_shape.cend());
  auto abs_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(abs_shape));
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(unpad_type[0]), abs_shape_ptr);
  unpad_cnode->set_abstract(abstract);

  unpad_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::vector<std::string> unpad_input_format = {AnfAlgo::GetOutputFormat(matmul, 0)};
  std::vector<TypeId> unpad_input_type = AnfAlgo::GetAllOutputDeviceTypes(matmul);
  std::vector<std::string> unpad_output_format = {unpad_input_format.front()};
  std::vector<TypeId> unpad_output_type = {unpad_input_type.front()};
  auto graph_sel_info =
    BuildSelectKernelBuildInfo(unpad_input_format, unpad_input_type, unpad_output_format, unpad_output_type);
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, unpad_cnode.get());

  (void)mng->Replace(matmul, unpad_cnode);
}

// Update matmul's Abatract and BuildInfo as M or N is changed
void UpdateMatmulInfo(const AnfNodePtr &matmul_node, const vec &unpad_shape, const vec &tail_shape) {
  ShapeVector abs_shape;
  for (size_t i = 0; i < unpad_shape.size(); ++i) {
    abs_shape.push_back(unpad_shape[i] + tail_shape[i]);
  }
  auto abs_shape_ptr = std::make_shared<abstract::Shape>(abstract::Shape(abs_shape));
  TypeId abs_type = common::AnfAlgo::GetOutputInferDataType(matmul_node, 0);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(abs_type), abs_shape_ptr);
  matmul_node->set_abstract(abstract);

  std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(matmul_node);
  std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(matmul_node);
  std::vector<std::string> output_formats = AnfAlgo::GetAllOutputFormats(matmul_node);
  std::vector<TypeId> output_types = AnfAlgo::GetAllOutputDeviceTypes(matmul_node);
  auto graph_sel_info = BuildSelectKernelBuildInfo(input_formats, input_types, output_formats, output_types,
                                                   AnfAlgo::GetProcessor(matmul_node));
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, matmul_node.get());
}

bool InsertPadUnpad(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (const auto &n : todos) {
    if (!common::AnfAlgo::CheckPrimitiveType(n, prim::kPrimMatMul)) {
      continue;
    }
    auto mm_cnode = n->cast<CNodePtr>();
    vec pad_shape_a, pad_shape_b, tail_shape_a, tail_shape_b, tail_shape_unpad, unpad_shape;
    bool pad_K{false}, pad_M{false}, pad_N{false};
    std::tie(pad_K, pad_M, pad_N) =
      NeedPad(mm_cnode, &pad_shape_a, &pad_shape_b, &unpad_shape, &tail_shape_a, &tail_shape_b, &tail_shape_unpad);
    if (!pad_K && !pad_M && !pad_N) {
      continue;
    }
    if (pad_K || pad_M) {
      InsertPad(mm_cnode, func_graph, true, pad_shape_a, tail_shape_a);
    }
    if (pad_K || pad_N) {
      InsertPad(mm_cnode, func_graph, false, pad_shape_b, tail_shape_b);
    }
    if (pad_M || pad_N) {
      UpdateMatmulInfo(mm_cnode, unpad_shape, tail_shape_unpad);
      InsertUnpad(mm_cnode, func_graph, mng, unpad_shape, tail_shape_unpad);
    }
    changed = true;
  }
  return changed;
}
}  // namespace

/* MatMul
 *
 *   C = MatMul(A, B)
 *   ------>
 *   A_pad = PadAkg(A)
 *   B_pad = PadAkg(B)
 *   C_pad = MatMul(A_pad, B_pad)
 *   C = UnPadAkg(C_pad)
 *
 */
bool InsertPadOps::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto graph_kernel_fg = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(graph_kernel_fg);
    changed = InsertPadUnpad(graph_kernel_fg) || changed;
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace graphkernel
}  // namespace mindspore
