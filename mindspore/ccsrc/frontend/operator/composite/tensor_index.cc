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

#include "frontend/operator/composite/tensor_index.h"
#include <algorithm>
#include <vector>

#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "ir/anf.h"
#include "frontend/optimizer/opt.h"
#include "ops/op_name.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
constexpr int64_t kZeroAnfValue = 0;

static inline bool IsAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildValue() == kValueAny;
}

// Parse slice to start, stop, step
AnfNodePtrList TensorIndex::ParseSlice(const AnfNodePtr &index_node, const abstract::AbstractSlicePtr &abs_slice_ptr,
                                       std::vector<int64_t> *init_by_one) {
  auto slice_info_abs = {abs_slice_ptr->start(), abs_slice_ptr->stop(), abs_slice_ptr->step()};
  const std::vector<string> &slice_str = {kSliceStart, kSliceStop, kSliceStep};
  AnfNodePtrList slice_nodes;
  (void)std::transform(
    slice_info_abs.begin(), slice_info_abs.end(), slice_str.begin(), std::back_inserter(slice_nodes),
    [this, &index_node](const AbstractBasePtr &slice_abs, const string &str) -> AnfNodePtr {
      if (IsAnyValue(slice_abs)) {
        return res_graph_->NewCNode({NewValueNode(prim::kPrimSliceGetItem), index_node, NewValueNode(str)});
      }
      if (slice_abs->isa<abstract::AbstractNone>()) {
        return NewValueNode(kZeroAnfValue);
      }
      if (slice_abs->isa<abstract::AbstractScalar>()) {
        if (slice_abs->BuildType()->type_id() != kNumberTypeInt64) {
          MS_EXCEPTION(TypeError) << "The type of input of the MakeSlice operator must be int64 bot got "
                                  << slice_abs->ToString();
        }
      }
      return NewValueNode(GetValue<int64_t>(slice_abs->BuildValue()));
    });

  constexpr size_t kStart = 0;
  constexpr size_t kStop = 1;
  constexpr size_t kStep = 2;
  *init_by_one = {0, 0, 0};
  if (abs_slice_ptr->start()->isa<abstract::AbstractNone>()) {
    (*init_by_one)[kStart] = 1;
  }
  if (abs_slice_ptr->stop()->isa<abstract::AbstractNone>()) {
    (*init_by_one)[kStop] = 1;
  }
  if (abs_slice_ptr->step()->isa<abstract::AbstractNone>()) {
    (*init_by_one)[kStep] = 1;
  }

  return AnfNodePtrList{slice_nodes[kStart], slice_nodes[kStop], slice_nodes[kStep]};
}

void TensorIndexGetitem::GetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                        const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice_ptr) {
  auto normalize_slice_prim = std::make_shared<Primitive>(kPrimNormalizeSlice->name());

  AnfNodePtrList normalize_slice_inputs{NewValueNode(normalize_slice_prim), data_node};
  std::vector<int64_t> init_by_none;
  auto normalize_slice_info_nodes = ParseSlice(index_node, abs_slice_ptr, &init_by_none);
  normalize_slice_prim->SetAttrs({{kAttrTupleIndexAxis, MakeValue(static_cast<int64_t>(0))},
                                  {kAttrTupleIndexTypes, MakeValue({})},
                                  {kAttrExpandDimsMask, MakeValue(static_cast<int64_t>(0))},
                                  {kAttrInitByNone, MakeValue(init_by_none)}});
  normalize_slice_inputs.insert(normalize_slice_inputs.end(), normalize_slice_info_nodes.begin(),
                                normalize_slice_info_nodes.end());
  auto normalize_slice_node = res_graph_->NewCNode(normalize_slice_inputs);

  AnfNodePtrList slice_nodes;
  // slice_info:{start, stop, step}
  const size_t slice_info_size = 3;
  for (size_t i = 0; i < slice_info_size; i++) {
    (void)slice_nodes.emplace_back(
      res_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), normalize_slice_node, NewValueNode(SizeToLong(i))}));
  }
  auto prim = std::make_shared<Primitive>(kPrimStridedSlice->name());
  const std::vector<std::string> &input_names = {"x", "begin", "end", "strides"};
  const std::vector<std::string> &output_names = {"output"};
  prim->SetAttrs({{ops::kBeginMask, MakeValue(kZeroAnfValue)},
                  {ops::kEndMask, MakeValue(kZeroAnfValue)},
                  {ops::kEllipsisMask, MakeValue(kZeroAnfValue)},
                  {ops::kNewAxisMask, MakeValue(kZeroAnfValue)},
                  {ops::kShrinkAxisMask, MakeValue(kZeroAnfValue)},
                  {kAttrInputNames, MakeValue(input_names)},
                  {kAttrOutputNames, MakeValue(output_names)}});
  ValueNodePtr strided_slice_vnode = NewValueNode(prim);
  (void)slice_nodes.insert(slice_nodes.begin(), {strided_slice_vnode, data_node});
  res_graph_->set_output(res_graph_->NewCNode(slice_nodes));
}

FuncGraphPtr TensorIndexGetitem::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 2;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The TensorIndexGetitem operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("TensorIndexGetitem");
  AnfNodePtr data_node = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();
  if (args_abs_list[1]->isa<abstract::AbstractSlice>()) {
    GetItemBySlice(data_node, index_node, args_abs_list[0], dyn_cast<abstract::AbstractSlice>(args_abs_list[1]));
  }
  return res_graph_;
}

void TensorIndexSetitem::SetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                        const AnfNodePtr &value_node, const AbstractBasePtr &data,
                                        const abstract::AbstractSlicePtr &abs_slice_ptr, const AbstractBasePtr &value) {
  AnfNodePtr value_shape_node;
  AnfNodePtrList output_nodes{NewValueNode(kPrimMakeTuple)};
  {
    std::vector<int64_t> init_by_none;
    auto stride_slice_input = ParseSlice(index_node, abs_slice_ptr, &init_by_none);
    auto slice_to_indices_prim = std::make_shared<Primitive>(kPrimSliceToIndices->name());
    slice_to_indices_prim->SetAttrs({{kAttrTupleIndexAxis, MakeValue(static_cast<int64_t>(0))},
                                     {kAttrTupleIndexTypes, MakeValue({})},
                                     {kAttrExpandDimsMask, MakeValue(static_cast<int64_t>(0))},
                                     {kAttrInitByNone, MakeValue(init_by_none)}});

    AnfNodePtrList slice_to_indices{NewValueNode(slice_to_indices_prim), data_node};
    const size_t slice_to_indices_output_size = 5;
    slice_to_indices.insert(slice_to_indices.end(), stride_slice_input.begin(), stride_slice_input.end());
    auto slice_to_indices_node = res_graph_->NewCNode(slice_to_indices);
    for (size_t i = 0; i < slice_to_indices_output_size; i++) {
      output_nodes.emplace_back(
        res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), slice_to_indices_node, NewValueNode(SizeToLong(i))}));
    }
  }
  auto new_value_node = value_node;
  auto type_id = dyn_cast<abstract::AbstractTensor>(data)->element()->BuildType();
  if (value->isa<abstract::AbstractTensor>()) {
    auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
    ValueNodePtr cast_vnode = NewValueNode(cast);
    new_value_node = res_graph_->NewCNode({cast_vnode, value_node, NewValueNode(type_id)});
  } else if (value->isa<abstract::AbstractScalar>()) {
    new_value_node = res_graph_->NewCNode(
      {NewValueNode(prim::kPrimFill), NewValueNode(type_id), NewValueNode(ShapeVector()), value_node});
  } else if (value->isa<abstract::AbstractSequence>()) {
    auto sequence_to_tensor =
      prim::GetPythonOps("sequence_to_tensor", "mindspore.ops.composite.multitype_ops._compile_utils");
    ValueNodePtr sequence_to_tensor_node = NewValueNode(sequence_to_tensor);
    new_value_node = res_graph_->NewCNode({sequence_to_tensor_node, value_node, NewValueNode(type_id)});
  }
  output_nodes.emplace_back(new_value_node);
  res_graph_->set_output(res_graph_->NewCNode(output_nodes));
}

FuncGraphPtr TensorIndexSetitem::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 3;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The TensorIndexSetitem operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("TensorIndexSetitem");
  AnfNodePtr data_node = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();
  AnfNodePtr value_node = res_graph_->add_parameter();

  if (args_abs_list[1]->isa<abstract::AbstractSlice>()) {
    SetItemBySlice(data_node, index_node, value_node, args_abs_list[0],
                   dyn_cast<abstract::AbstractSlice>(args_abs_list[kIndex1]), args_abs_list[kIndex2]);
  }

  return res_graph_;
}
}  // namespace prim
}  // namespace mindspore
