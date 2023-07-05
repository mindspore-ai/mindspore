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
constexpr int64_t kOneAnfValue = 1;

static abstract::AbstractTuplePtr VectorToTuple(const std::vector<int64_t> &nums) {
  abstract::AbstractBasePtrList elems;
  (void)std::transform(nums.begin(), nums.end(), std::back_inserter(elems),
                       [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });
  return std::make_shared<abstract::AbstractTuple>(elems);
}

static inline bool IsAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildValue() == kValueAny;
}

IndexHandleLevel TensorIndex::PreHandleIndex(const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice) {
  ShapeMap shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data->BuildShape());
  auto input_shape = shape_map[kShape];
  data_shape_ = input_shape;
  if (data_shape_.empty()) {
    MS_EXCEPTION(TypeError) << "Cannot iterate over a scalar tensor.";
  }
  if (!IsDynamic(data_shape_) && !IsAnyValue(abs_slice->start()) && !IsAnyValue(abs_slice->stop()) &&
      !IsAnyValue(abs_slice->step())) {
    return IndexHandleLevel::kHandleByConstFold;
  }
  MS_LOG(DEBUG) << "The slice index is dynamic.";
  return IndexHandleLevel::kHandleByFunc;
}

// Handle slice by cpu ops kPrimNormalizeSlice
AnfNodePtrList TensorIndex::NormalizeSlice(const AbstractBasePtrList &slice_info_abs, const AnfNodePtr &shape_node,
                                           const AnfNodePtr &index_node) {
  std::vector<string> slice_str = {kSliceStart, kSliceStop, kSliceStep};
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
      Slice::CheckSliceType(slice_abs);
      return NewValueNode(GetValue<int64_t>(slice_abs->BuildValue()));
    });

  std::vector<int64_t> init_by_none_node_args = {0, 0, 0};

  for (size_t i = 0; i < slice_info_abs.size(); i++) {
    if (slice_info_abs[i]->isa<abstract::AbstractNone>()) {
      init_by_none_node_args[i] = kOneAnfValue;
    }
  }
  const tensor::TensorPtr init_by_none_tensor = std::make_shared<tensor::Tensor>(
    kNumberTypeInt64, ShapeVector{3}, init_by_none_node_args.data(), sizeof(int64_t) * init_by_none_node_args.size());
  auto init_by_none_node = NewValueNode(init_by_none_tensor->ToAbstract()->BuildValue());
  constexpr size_t kStart = 0;
  constexpr size_t kStop = 1;
  constexpr size_t kStep = 2;
  return AnfNodePtrList{shape_node, init_by_none_node, slice_nodes[kStart], slice_nodes[kStop], slice_nodes[kStep]};
}

AnfNodePtr TensorIndex::NormalizeSliceInfo(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                           const IndexHandleLevel &index_handle_level,
                                           const abstract::AbstractSlicePtr &abs_slice_ptr, bool *empty,
                                           bool slice_to_indices) {
  auto start_abs = abs_slice_ptr->start();
  auto stop_abs = abs_slice_ptr->stop();
  auto step_abs = abs_slice_ptr->step();
  if (index_handle_level == IndexHandleLevel::kHandleByConstFold) {
    std::shared_ptr<Slice> slice_ptr =
      std::make_shared<Slice>(start_abs, stop_abs, step_abs, data_shape_[0], slice_to_indices);
    if (slice_ptr->is_empty_slice()) {
      *empty = true;
      return data_node;
    }
    abstract::AbstractTuplePtr slice_info_abs =
      VectorToTuple({slice_ptr->start(), slice_ptr->stop(), slice_ptr->step()});
    return NewValueNode(slice_info_abs->BuildValue());
  }
  AnfNodePtr shape_node = NewCNode({NewValueNode(prim::kPrimShape), data_node}, res_graph_);
  AnfNodePtrList stride_slice{NewValueNode(prim::kPrimNormalizeSlice)};
  auto stride_slice_input = NormalizeSlice({start_abs, stop_abs, step_abs}, shape_node, index_node);
  (void)stride_slice.insert(stride_slice.end(), stride_slice_input.begin(), stride_slice_input.end());
  return res_graph_->NewCNode(stride_slice);
}

void TensorIndexGetitem::GetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                        const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice_ptr) {
  IndexHandleLevel index_handle_level = PreHandleIndex(data, abs_slice_ptr);
  bool is_empty_slice = false;
  AnfNodePtr normalized_slice_node =
    NormalizeSliceInfo(data_node, index_node, index_handle_level, abs_slice_ptr, &is_empty_slice, false);
  if (is_empty_slice) {
    ShapeVector empty_slice = data_shape_;
    empty_slice[0] = 0;
    const tensor::TensorPtr &slice_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, empty_slice);
    return res_graph_->set_output(NewValueNode(slice_tensor->ToAbstract()->BuildValue()));
  }
  AnfNodePtrList slice_nodes;
  // slice_info:{start, stop, step}
  const size_t slice_info_size = 3;
  if (normalized_slice_node->isa<ValueNode>()) {
    size_t data_dim = data_shape_.size();
    std::vector<int64_t> start_strides(data_dim, 0);
    std::vector<int64_t> stop_strides = data_shape_;
    std::vector<int64_t> step_strides(data_dim, 1);
    auto slice_info = GetValue<std::vector<int64_t>>(GetValueNode(normalized_slice_node));
    start_strides[0] = slice_info[kIndex0];
    stop_strides[0] = slice_info[kIndex1];
    step_strides[0] = slice_info[kIndex2];
    auto slice_infos = std::vector<AbstractBasePtr>{VectorToTuple(start_strides), VectorToTuple(stop_strides),
                                                    VectorToTuple(step_strides)};
    (void)std::transform(slice_infos.begin(), slice_infos.end(), std::back_inserter(slice_nodes),
                         [](const AbstractBasePtr &slice_info) { return NewValueNode(slice_info->BuildValue()); });
  } else {
    for (size_t i = 0; i < slice_info_size; i++) {
      (void)slice_nodes.emplace_back(res_graph_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), normalized_slice_node, NewValueNode(SizeToLong(i))}));
    }
  }

  ValueNodePtr strided_slice_vnode = NewValueNode(prim::kPrimStridedSlice);
  auto prim = GetValueNode<PrimitivePtr>(strided_slice_vnode);
  const std::vector<std::string> &input_names = {"x", "begin", "end", "strides"};
  const std::vector<std::string> &output_names = {"output"};
  (void)prim->SetAttrs({{ops::kBeginMask, MakeValue(kZeroAnfValue)},
                        {ops::kEndMask, MakeValue(kZeroAnfValue)},
                        {ops::kEllipsisMask, MakeValue(kZeroAnfValue)},
                        {ops::kNewAxisMask, MakeValue(kZeroAnfValue)},
                        {ops::kShrinkAxisMask, MakeValue(kZeroAnfValue)},
                        {kAttrInputNames, MakeValue(input_names)},
                        {kAttrOutputNames, MakeValue(output_names)}});
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
  IndexHandleLevel index_handle_level = PreHandleIndex(data, abs_slice_ptr);
  AnfNodePtr value_shape_node;
  AnfNodePtrList output_nodes{NewValueNode(kPrimMakeTuple)};
  if (index_handle_level == IndexHandleLevel::kHandleByConstFold) {
    bool is_empty_slice = false;
    AnfNodePtr normalized_slice_node =
      NormalizeSliceInfo(data_node, index_node, index_handle_level, abs_slice_ptr, &is_empty_slice, true);
    if (is_empty_slice) {
      auto stub_outputs = AnfNodePtrList(6, NewValueNode(SizeToLong(0)));
      (void)output_nodes.insert(output_nodes.end(), stub_outputs.begin(), stub_outputs.end());
      return res_graph_->set_output(res_graph_->NewCNode(output_nodes));
    }
    auto slice_info = GetValue<std::vector<int64_t>>(GetValueNode(normalized_slice_node));
    int64_t start = slice_info[kIndex0];
    int64_t stop = slice_info[kIndex1];
    int64_t step = slice_info[kIndex2];
    std::vector<int64_t> indices;
    if (step > 0) {
      for (int64_t i = start; i < stop; i += step) {
        (void)indices.emplace_back(i);
      }
    } else {
      for (int64_t i = start; i > stop; i += step) {
        (void)indices.emplace_back(i);
      }
    }
    ShapeVector indices_shp({static_cast<int64_t>(indices.size()), 1});
    auto shp_buf_size = sizeof(int64_t) * indices.size();
    auto indices_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, indices_shp, indices.data(), shp_buf_size);
    (void)output_nodes.emplace_back(NewValueNode(indices_tensor->ToAbstract()->BuildValue()));
    auto value_shape = data_shape_;
    value_shape[0] = SizeToLong(indices.size());
    value_shape_node = NewValueNode(std::vector<int64_t>(value_shape));
    (void)output_nodes.emplace_back(value_shape_node);
    (void)output_nodes.emplace_back(NewValueNode(start));
    (void)output_nodes.emplace_back(NewValueNode(stop));
    (void)output_nodes.emplace_back(NewValueNode(step));
  } else {
    auto start_abs = abs_slice_ptr->start();
    auto stop_abs = abs_slice_ptr->stop();
    auto step_abs = abs_slice_ptr->step();
    AnfNodePtr shape_node = NewCNode({NewValueNode(prim::kPrimShape), data_node}, res_graph_);
    AnfNodePtrList slice_to_indices{NewValueNode(prim::kPrimSliceToIndices)};
    auto stride_slice_input = NormalizeSlice({start_abs, stop_abs, step_abs}, shape_node, index_node);
    const size_t slice_to_indices_output_size = 5;
    (void)slice_to_indices.insert(slice_to_indices.end(), stride_slice_input.begin(), stride_slice_input.end());
    auto slice_to_indices_node = res_graph_->NewCNode(slice_to_indices);
    for (size_t i = 0; i < slice_to_indices_output_size; i++) {
      (void)output_nodes.emplace_back(
        res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), slice_to_indices_node, NewValueNode(SizeToLong(i))}));
    }
  }
  auto new_value_node = value_node;
  auto type_id = dyn_cast<abstract::AbstractTensor>(data)->element()->BuildType();
  if (value->isa<abstract::AbstractTensor>()) {
    auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
    ValueNodePtr cast_node = NewValueNode(cast);
    new_value_node = res_graph_->NewCNode({cast_node, value_node, NewValueNode(type_id)});
  } else if (value->isa<abstract::AbstractScalar>()) {
    new_value_node = res_graph_->NewCNode(
      {NewValueNode(prim::kPrimFill), NewValueNode(type_id), NewValueNode(ShapeVector()), value_node});
  } else if (value->isa<abstract::AbstractSequence>()) {
    auto sequence_to_tensor =
      prim::GetPythonOps("sequence_to_tensor", "mindspore.ops.composite.multitype_ops._compile_utils");
    ValueNodePtr sequence_to_tensor_node = NewValueNode(sequence_to_tensor);
    new_value_node = res_graph_->NewCNode({sequence_to_tensor_node, value_node, NewValueNode(type_id)});
  }
  (void)output_nodes.emplace_back(new_value_node);
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
