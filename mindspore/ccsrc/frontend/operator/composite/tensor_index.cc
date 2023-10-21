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
#include <tuple>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "frontend/operator/cc_implementations.h"
#include "ir/anf.h"
#include "frontend/optimizer/opt.h"
#include "ops/op_name.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
constexpr int64_t kZeroAnfValue = 0;
constexpr size_t kMaxDimNums = 8;

static inline bool IsAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildValue() == kValueAny;
}

IndexHandleLevel TensorIndex::PreHandleIndex(const AbstractBasePtr &data, const abstract::AbstractTuplePtr &tuple_abs) {
  ShapeMap shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data->BuildShape());
  auto input_shape = shape_map[kShape];
  data_shape_ = input_shape;
  if (!IsDynamic(data_shape_) && !IsAnyValue(tuple_abs)) {
    MS_LOG(DEBUG) << "The tuple index is constant.";
    return IndexHandleLevel::kHandleByConstFold;
  }
  if (tuple_abs->size() >= kMaxDimNums) {
    MS_EXCEPTION(IndexError) << "The size of tuple index must in the range of [0, 8] if tensor shape is dynamic.";
  }
  MS_LOG(DEBUG) << "The tuple index is dynamic.";
  return IndexHandleLevel::kHandleByFunc;
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

static ValueNodePtr MakeStridedSliceNode(int64_t shrink_axis) {
  auto prim = std::make_shared<Primitive>(kPrimStridedSlice->name());
  const std::vector<std::string> &input_names = {"x", "begin", "end", "strides"};
  const std::vector<std::string> &output_names = {"output"};
  prim->SetAttrs({{ops::kBeginMask, MakeValue(kZeroAnfValue)},
                  {ops::kEndMask, MakeValue(kZeroAnfValue)},
                  {ops::kEllipsisMask, MakeValue(kZeroAnfValue)},
                  {ops::kNewAxisMask, MakeValue(kZeroAnfValue)},
                  {ops::kShrinkAxisMask, MakeValue(shrink_axis)},
                  {kAttrInputNames, MakeValue(input_names)},
                  {kAttrOutputNames, MakeValue(output_names)}});
  return NewValueNode(prim);
}

static ValueNodePtr MakeSliceToIndicesNode(size_t index_axis, const std::vector<int64_t> &tuple_index_types,
                                           int64_t expand_dims_mask, const std::vector<int64_t> &init_by_none) {
  auto slice_to_indices_prim = std::make_shared<Primitive>(kPrimSliceToIndices->name());
  slice_to_indices_prim->SetAttrs({{kAttrTupleIndexAxis, MakeValue(SizeToLong(index_axis))},
                                   {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                                   {kAttrExpandDimsMask, MakeValue(expand_dims_mask)},
                                   {kAttrInitByNone, MakeValue(init_by_none)}});
  return NewValueNode(slice_to_indices_prim);
}

static ValueNodePtr MakeGatherNode() {
  auto prim = std::make_shared<Primitive>(kPrimGather->name());
  const std::vector<std::string> &input_names = {"params", "indices", "axis"};
  const std::vector<std::string> &output_names = {"output"};
  prim->SetAttrs({{"batch_dims", MakeValue(kZeroAnfValue)},
                  {kAttrInputNames, MakeValue(input_names)},
                  {kAttrOutputNames, MakeValue(output_names)}});
  return NewValueNode(prim);
}

static ValueNodePtr MakeReshapeNode() {
  auto reshape_prim = std::make_shared<Primitive>(kPrimReshape->name());
  const std::vector<std::string> &reshape_input_names = {"tensor", "shape"};
  const std::vector<std::string> &reshape_output_names = {"output"};
  reshape_prim->SetAttrs(
    {{kAttrInputNames, MakeValue(reshape_input_names)}, {kAttrOutputNames, MakeValue(reshape_output_names)}});
  return NewValueNode(reshape_prim);
}

static ValueNodePtr MakeExpandDimsNode() {
  auto expand_dim_prims = std::make_shared<Primitive>(kPrimExpandDims->name());
  const std::vector<std::string> &input_names = {"x", "axis"};
  const std::vector<std::string> &output_names = {"output"};
  expand_dim_prims->SetAttrs({{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
  return NewValueNode(expand_dim_prims);
}

static ValueNodePtr MakeRemoveExpandedDimsNode(bool has_true, bool has_sequence,
                                               const std::vector<AnfNodePtr> &indices_out_list,
                                               int64_t expand_dims_mask,
                                               const std::vector<int64_t> &new_tuple_index_types) {
  auto remove_expanded_dims_prim = std::make_shared<Primitive>(kPrimRemoveExpandedDims->name());
  remove_expanded_dims_prim->SetAttrs({
    {kAttrHasTrue, MakeValue(has_true)},
    {kAttrHasSequence, MakeValue(has_sequence)},
    {kAttrEmptyIndicesOut, MakeValue(indices_out_list.size() == 1)},
    {kAttrExpandDimsCnt, MakeValue(SizeToLong(std::bitset<8>(expand_dims_mask).count()))},
    {kAttrTupleIndexTypes, MakeValue(new_tuple_index_types)},
  });
  return NewValueNode(remove_expanded_dims_prim);
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
  (void)normalize_slice_inputs.insert(normalize_slice_inputs.end(), normalize_slice_info_nodes.begin(),
                                      normalize_slice_info_nodes.end());
  auto normalize_slice_node = res_graph_->NewCNode(normalize_slice_inputs);

  AnfNodePtrList slice_nodes;
  // slice_info:{start, stop, step}
  const size_t slice_info_size = 3;
  for (size_t i = 0; i < slice_info_size; i++) {
    (void)slice_nodes.emplace_back(
      res_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), normalize_slice_node, NewValueNode(SizeToLong(i))}));
  }
  auto strided_slice_vnode = MakeStridedSliceNode(0);
  (void)slice_nodes.insert(slice_nodes.begin(), {strided_slice_vnode, data_node});
  res_graph_->set_output(res_graph_->NewCNode(slice_nodes));
}

FuncGraphPtr TensorIndexGetitem::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t max_args_size = 3;
  if (arg_length > max_args_size) {
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
  if (args_abs_list[1]->isa<abstract::AbstractTuple>()) {
    (void)res_graph_->add_parameter();
    GetItemByTuple(data_node, index_node, args_abs_list[0], dyn_cast<abstract::AbstractTuple>(args_abs_list[1]),
                   args_abs_list[2]);
  }
  return res_graph_;
}

template <typename T>
static bool CheckTypeIsInstance(const T &type, const std::vector<T> &target_types) {
  return std::any_of(target_types.begin(), target_types.end(),
                     [&type](const auto &target_type) { return target_type == type; });
}

static std::vector<int64_t> GetTupleIndexType(const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                              const ShapeVector &data_shape, bool *has_ellipsis,
                                              size_t *ellipsis_position, size_t *not_ellipsis_position_cnt,
                                              std::bitset<kMaxDimNums> *expand_dims_mask) {
  std::vector<int64_t> tuple_index_types;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    const auto index_type_id = index_abs->BuildType()->type_id();
    if (CheckTypeIsInstance<TypeId>(
          index_type_id, {kNumberTypeInt, kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64,
                          kObjectTypeList, kObjectTypeTuple, kNumberTypeBool})) {
      (void)tuple_index_types.emplace_back(kObjectTypeTensorType);
      *not_ellipsis_position_cnt += 1;
    } else if (index_type_id == kObjectTypeSlice) {
      (void)tuple_index_types.emplace_back(kObjectTypeSlice);
      *not_ellipsis_position_cnt += 1;
    } else if (index_type_id == kMetaTypeNone) {
      (void)tuple_index_types.emplace_back(kObjectTypeSlice);
      *not_ellipsis_position_cnt += 1;
    } else if (index_type_id == kMetaTypeEllipsis) {
      if (*has_ellipsis) {
        MS_EXCEPTION(IndexError) << "An index can only have a single ellipsis('...')";
      }
      *ellipsis_position = i;
      *has_ellipsis = true;
      (void)tuple_index_types.emplace_back(kMetaTypeEllipsis);
    } else if (index_type_id == kObjectTypeTensorType) {
      (void)tuple_index_types.emplace_back(kObjectTypeTensorType);
      *not_ellipsis_position_cnt += 1;
    }
    if (CheckTypeIsInstance<TypeId>(index_type_id, {kNumberTypeBool, kMetaTypeNone})) {
      (*expand_dims_mask)[i] = 1;
    }
  }
  auto ori_size = tuple_index_types.size();
  for (size_t i = ori_size; i < 8; i++) {
    (void)tuple_index_types.emplace_back(kTypeUnknown);
  }
  size_t expand_dims_cnt = expand_dims_mask->count();
  if (!IsDynamicRank(data_shape)) {
    if (*not_ellipsis_position_cnt - expand_dims_cnt > data_shape.size()) {
      MS_EXCEPTION(IndexError) << "Tuple index " << tuple_abs_ptr->ToString() << " out rang of tensor shape"
                               << data_shape;
    }
  }
  return tuple_index_types;
}

AnfNodePtr TensorIndex::IntIndexToTensor(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                         const AbstractBasePtr &int_index_abs,
                                         const std::vector<int64_t> &tuple_index_types, size_t dim_index,
                                         int64_t expand_dims_mask) {
  AnfNodePtr new_index_node = index_node;
  auto prim = std::make_shared<Primitive>(kPrimNormalizeTupleIndex->name());
  prim->SetAttrs({{kAttrOriginIndexType, MakeValue(kIntIndex)},
                  {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                  {kAttrTupleIndexAxis, MakeValue(SizeToLong(dim_index))},
                  {kAttrExpandDimsMask, MakeValue(expand_dims_mask)}});
  auto normalize_index_node = NewValueNode(prim);
  if (int_index_abs->BuildType()->type_id() != kNumberTypeInt64) {
    new_index_node = res_graph_->NewCNode({NewValueNode(kPrimScalarCast), index_node, NewValueNode(kInt64)});
  }
  auto new_int_index_node = res_graph_->NewCNode({normalize_index_node, data_node, new_index_node});
  return new_int_index_node;
}

AnfNodePtr TensorIndex::SequenceIndexToTensor(const AnfNodePtr &data_node, const AnfNodePtr &sequence_index_node,
                                              const std::vector<int64_t> &tuple_index_types,
                                              const AbstractBasePtr &sequence_index_abs, size_t dim_index,
                                              int64_t expand_dims_mask, bool *empty_sequence) {
  auto prim = std::make_shared<Primitive>(kPrimNormalizeTupleIndex->name());
  AnfNodePtr new_index_node = sequence_index_node;
  *empty_sequence = dyn_cast<abstract::AbstractSequence>(sequence_index_abs)->empty();
  if (*empty_sequence) {
    return NewValueNode(SizeToLong(0));
  }
  auto list_index_val_abs = sequence_index_abs->cast<abstract::AbstractSequencePtr>();
  // Handle bool list index
  const AbstractBasePtrList &list_index_val_ele = list_index_val_abs->elements();
  if (std::all_of(list_index_val_ele.begin(), list_index_val_ele.end(),
                  [](const AbstractBasePtr &x) { return x->BuildType()->type_id() == kNumberTypeBool; })) {
    *empty_sequence = std::all_of(list_index_val_ele.begin(), list_index_val_ele.end(), [](const AbstractBasePtr &x) {
      if (IsAnyValue(x)) {
        MS_EXCEPTION(IndexError) << "Bool index in list must be const value.";
      }
      return GetValue<bool>(x->BuildValue()) == False;
    });
    if (*empty_sequence) {
      return NewValueNode(SizeToLong(0));
    }
    prim->SetAttrs({{kAttrOriginIndexType, MakeValue(kBoolSequenceIndex)},
                    {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                    {kAttrTupleIndexAxis, MakeValue(SizeToLong(dim_index))},
                    {kAttrExpandDimsMask, MakeValue(expand_dims_mask)}});
  } else if (std::all_of(list_index_val_ele.begin(), list_index_val_ele.end(), [](const AbstractBasePtr &x) {
               return CheckTypeIsInstance<TypeId>(
                 x->BuildType()->type_id(),
                 {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool});
             })) {
    AnfNodePtrList new_sequence_index_node_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    std::transform(list_index_val_ele.begin(), list_index_val_ele.end(),
                   std::back_inserter(new_sequence_index_node_inputs), [this](const AbstractBasePtr &x) -> AnfNodePtr {
                     auto ele_type_id = x->BuildType()->type_id();
                     if (ele_type_id != kNumberTypeInt64) {
                       return res_graph_->NewCNode(
                         {NewValueNode(kPrimScalarCast), NewValueNode(x->BuildValue()), NewValueNode(kInt64)});
                     }
                     return NewValueNode(x->BuildValue());
                   });
    new_index_node = res_graph_->NewCNode(new_sequence_index_node_inputs);
    prim->SetAttrs({{kAttrOriginIndexType, MakeValue(kTensorIndexSequenceIndex)},
                    {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                    {kAttrTupleIndexAxis, MakeValue(SizeToLong(dim_index))},
                    {kAttrExpandDimsMask, MakeValue(expand_dims_mask)}});
  } else {
    auto sequence_to_index =
      prim::GetPythonOps("sequence_to_index", "mindspore.ops.composite.multitype_ops._constexpr_utils");
    ValueNodePtr sequence_to_index_vnode = NewValueNode(sequence_to_index);
    return res_graph_->NewCNode({sequence_to_index_vnode, sequence_index_node, NewValueNode(data_shape_[dim_index])});
  }

  auto normalize_index_node = NewValueNode(prim);
  return res_graph_->NewCNode({normalize_index_node, data_node, new_index_node});
}

static size_t NormalizeDimIndex(const size_t data_dims, size_t dim_index,
                                const std::vector<int64_t> &tuple_index_types) {
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  bool has_ellipsis = false;
  for (size_t i = 0; i < 8; i++) {
    if (tuple_index_types[i] == kMetaTypeEllipsis) {
      has_ellipsis = true;
      ellipse_position = i;
    } else if (tuple_index_types[i] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }
  size_t output = 0;
  size_t ellipse_occupy_dims = data_dims - not_ellipse_occupy_dims;
  if (!has_ellipsis || dim_index < ellipse_position) {
    return dim_index;
  }
  output = ellipse_occupy_dims + dim_index - 1;
  return output;
}

AnfNodePtr TensorIndex::SliceIndexToTensor(const AnfNodePtr &data_node, const std::vector<int64_t> &tuple_index_types,
                                           const AnfNodePtr &slice_index_node,
                                           const abstract::AbstractSlicePtr &slice_abs, const size_t dim_index,
                                           const IndexHandleLevel index_handle_level, int64_t expand_dims_mask) {
  std::vector<int64_t> init_by_none;
  auto normalize_slice_info = ParseSlice(slice_index_node, slice_abs, &init_by_none);
  auto slice_to_indices_vnode = MakeSliceToIndicesNode(dim_index, tuple_index_types, expand_dims_mask, init_by_none);
  AnfNodePtrList slice_to_indices_inputs{slice_to_indices_vnode, data_node};
  (void)slice_to_indices_inputs.insert(slice_to_indices_inputs.end(), normalize_slice_info.begin(),
                                       normalize_slice_info.end());
  AnfNodePtr new_normalized_slice_index_node = res_graph_->NewCNode(slice_to_indices_inputs);
  return res_graph_->NewCNode(
    {NewValueNode(kPrimTupleGetItem), new_normalized_slice_index_node, NewValueNode(SizeToLong(0))});
}

AnfNodePtr TensorIndex::NoneIndexToTensor(const AnfNodePtr &data_node, const std::vector<int64_t> &tuple_index_types,
                                          const AnfNodePtr &none_index_node, const size_t dim_index) {
  auto prim = std::make_shared<Primitive>(kPrimNormalizeTupleIndex->name());
  prim->SetAttrs({{kAttrOriginIndexType, MakeValue(kNoneIndex)},
                  {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                  {kAttrTupleIndexAxis, MakeValue(SizeToLong(dim_index))}});
  auto normalize_index_node = NewValueNode(prim);
  auto new_normalized_slice_node = res_graph_->NewCNode({normalize_index_node, data_node, none_index_node});
  return new_normalized_slice_node;
}

AnfNodePtr TensorIndex::EllipsisIndexToTensor(const AnfNodePtr &data_node,
                                              const std::vector<int64_t> &tuple_index_types,
                                              const AnfNodePtr &none_index_node, const size_t dim_index) {
  auto prim = std::make_shared<Primitive>(kPrimNormalizeTupleIndex->name());
  prim->SetAttrs({{kAttrOriginIndexType, MakeValue(kEllipsisIndex)},
                  {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                  {kAttrTupleIndexAxis, MakeValue(SizeToLong(dim_index))}});
  auto normalize_index_node = NewValueNode(prim);
  auto new_normalized_slice_node = res_graph_->NewCNode({normalize_index_node, data_node, none_index_node});
  return new_normalized_slice_node;
}

std::vector<AnfNodePtr> TensorIndex::NormalizeTupleIndex(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                                         const std::vector<int64_t> &tuple_index_types,
                                                         const IndexHandleLevel index_handle_level,
                                                         const bool has_ellipsis,
                                                         const abstract::AbstractTuplePtr &tuple_abs_ptr) {
  std::vector<AnfNodePtr> normalized_tensors;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    auto new_index_node =
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), index_node, NewValueNode(SizeToLong(i))});
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    AnfNodePtr shape_node = NewCNode({NewValueNode(prim::kPrimShape), data_node}, res_graph_);
    if (CheckTypeIsInstance<TypeId>(index_type_id,
                                    {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64})) {
      auto new_index_item = IntIndexToTensor(data_node, new_index_node, index_abs, tuple_index_types, i, 0);
      (void)normalized_tensors.emplace_back(new_index_item);
    } else if (CheckTypeIsInstance<TypeId>(index_type_id, {kObjectTypeList, kObjectTypeTuple})) {
      bool empty_sequence = true;
      auto new_index_item =
        SequenceIndexToTensor(data_node, new_index_node, tuple_index_types, index_abs, i, 0, &empty_sequence);
      if (empty_sequence) {
        MS_EXCEPTION(IndexError) << "The sequence element(tuple/list) in tuple index can't be empty.";
      }
      (void)normalized_tensors.emplace_back(new_index_item);
    } else if (index_type_id == kObjectTypeTensorType) {
      auto tensor_abs = dyn_cast<abstract::AbstractTensor>(index_abs);
      if (!CheckTypeIsInstance<TypeId>(
            tensor_abs->element()->BuildType()->type_id(),
            {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool})) {
        MS_EXCEPTION(IndexError) << "The tensor element in tuple index must be int or bool type, but got"
                                 << tensor_abs->element()->BuildType();
      }
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      ValueNodePtr cast_vnode = NewValueNode(cast);
      auto new_index_item = res_graph_->NewCNode({cast_vnode, new_index_node, NewValueNode(kInt64)});
      (void)normalized_tensors.emplace_back(new_index_item);
    } else if (index_type_id == kObjectTypeSlice) {
      auto new_index_item = SliceIndexToTensor(data_node, tuple_index_types, new_index_node,
                                               dyn_cast<abstract::AbstractSlice>(index_abs), i, index_handle_level, 0);
      (void)normalized_tensors.emplace_back(new_index_item);
    } else if (index_type_id == kMetaTypeNone) {
      auto new_index_item = NoneIndexToTensor(data_node, tuple_index_types, NewValueNode(static_cast<int64_t>(0)), i);
      (void)normalized_tensors.emplace_back(new_index_item);
    } else if (index_type_id == kNumberTypeBool) {
      if (IsAnyValue(index_abs) || GetValue<bool>(index_abs->BuildValue()) == false) {
        MS_EXCEPTION(IndexError) << "Bool element of tuple index must be 'True', but got 'False'.";
      }
      (void)normalized_tensors.emplace_back(NewValueNode(std::make_shared<tensor::Tensor>(std::vector<int64_t>{0})));
    } else if (index_type_id == kMetaTypeEllipsis) {
      continue;
    } else {
      MS_EXCEPTION(TypeError)
        << "For 'tensor_setitem_by_tuple', the types only support 'Slice', 'Ellipsis', 'None', 'Tensor', "
           "'int', 'List', 'Tuple', 'bool', but got "
        << index_abs->BuildType()->ToString();
    }
  }
  // Normalize ellipse index in tuple by transfer_ellipse_slice because the nums of ellipse occupy dims unknown.
  if (has_ellipsis) {
    size_t index_size = IsDynamicRank(data_shape_) ? kMaxDimNums : data_shape_.size();
    index_size = index_size - normalized_tensors.size();
    for (size_t i = 0; i < index_size; i++) {
      auto new_index_item =
        EllipsisIndexToTensor(data_node, tuple_index_types, NewValueNode(static_cast<int64_t>(0)), i);
      (void)normalized_tensors.emplace_back(new_index_item);
    }
  }
  return normalized_tensors;
}

std::tuple<AnfNodePtr, AnfNodePtr, AnfNodePtr> TensorIndexGetitem::NormalizeStrideInfoFromTuple(
  const AnfNodePtr &data_node, const AnfNodePtr &index_node, const AbstractBasePtr &index_abs,
  const std::vector<int64_t> &tuple_index_types, size_t tuple_index) {
  std::vector<int64_t> init_by_none;
  auto normalize_slice_info_nodes = ParseSlice(index_node, dyn_cast<abstract::AbstractSlice>(index_abs), &init_by_none);
  auto normalize_slice_prim = std::make_shared<Primitive>(kPrimNormalizeSlice->name());
  normalize_slice_prim->SetAttrs({{kAttrTupleIndexAxis, MakeValue(SizeToLong(tuple_index))},
                                  {kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
                                  {kAttrExpandDimsMask, MakeValue(static_cast<int64_t>(0))},
                                  {kAttrInitByNone, MakeValue(init_by_none)}});
  AnfNodePtrList normalize_slice_inputs{NewValueNode(normalize_slice_prim), data_node};
  (void)normalize_slice_inputs.insert(normalize_slice_inputs.end(), normalize_slice_info_nodes.begin(),
                                      normalize_slice_info_nodes.end());
  AnfNodePtrList slice_nodes;
  // slice_info:{start, stop, step}
  auto normalize_slice_node = res_graph_->NewCNode(normalize_slice_inputs);
  auto begin_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalize_slice_node, NewValueNode(SizeToLong(kIndex0))});

  auto end_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalize_slice_node, NewValueNode(SizeToLong(kIndex1))});

  auto step_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalize_slice_node, NewValueNode(SizeToLong(kIndex2))});
  return {begin_stride, end_stride, step_stride};
}

void TensorIndexGetitem::ConstGetStrideInfoFromTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                                     const std::vector<int64_t> &tuple_index_types, bool has_ellipsis,
                                                     const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                                     size_t not_ellipsis_position_cnt, size_t ellipsis_position) {
  AnfNodePtrList begin_strides{NewValueNode(kPrimMakeTuple)};
  AnfNodePtrList end_strides{NewValueNode(kPrimMakeTuple)};
  AnfNodePtrList step_strides{NewValueNode(kPrimMakeTuple)};
  int64_t shrink_axis = 0;
  size_t index_count = 0;
  size_t ellipsis_count = 0;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    auto new_index_node =
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), index_node, NewValueNode(SizeToLong(i))});
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    if (index_type_id == kMetaTypeNone) {
      (void)begin_strides.emplace_back(NewValueNode(static_cast<int64_t>(0)));
      (void)end_strides.emplace_back(NewValueNode(static_cast<int64_t>(1)));
      (void)step_strides.emplace_back(NewValueNode(static_cast<int64_t>(1)));
      index_count += 1;
    } else if (index_type_id == kObjectTypeTensorType) {
      auto tensor_abs = index_abs->BuildValue()->cast<mindspore::tensor::TensorPtr>();
      int64_t start = 0;
      if (tensor_abs->data_type() == kNumberTypeInt64) {
        start = *reinterpret_cast<int64_t *>(tensor_abs->data_c());
      } else if (tensor_abs->data_type() == kNumberTypeInt32) {
        start = *reinterpret_cast<int32_t *>(tensor_abs->data_c());
      } else {
        MS_EXCEPTION(IndexError) << "Basic index in tuple must be int64/int32";
      }
      (void)begin_strides.emplace_back(NewValueNode(start));
      (void)end_strides.emplace_back(NewValueNode(start + 1));
      (void)step_strides.emplace_back(NewValueNode(static_cast<int64_t>(1)));
      shrink_axis += 1 << index_count;
      index_count += 1;
    } else if (index_type_id == kNumberTypeInt64) {
      int64_t start = GetValue<int64_t>(index_abs->BuildValue());
      (void)begin_strides.emplace_back(NewValueNode(start));
      (void)end_strides.emplace_back(NewValueNode(start + 1));
      (void)step_strides.emplace_back(NewValueNode(static_cast<int64_t>(1)));
      shrink_axis += 1 << index_count;
      index_count += 1;
    } else if (index_type_id == kObjectTypeSlice) {
      auto [begin_stride, end_stride, step_stride] =
        NormalizeStrideInfoFromTuple(data_node, new_index_node, index_abs, tuple_index_types, i);
      (void)begin_strides.emplace_back(begin_stride);
      (void)end_strides.emplace_back(end_stride);
      (void)step_strides.emplace_back(step_stride);
      index_count += 1;
    } else if (index_type_id == kMetaTypeEllipsis) {
      if (ellipsis_count >= 1) {
        MS_EXCEPTION(ValueError) << "An index can have only one ellipsis (...)";
      }
      ellipsis_count += 1;
      size_t ellipsis_range_size = data_shape_.size() - not_ellipsis_position_cnt;
      for (size_t j = 0; j < ellipsis_range_size; j++) {
        (void)begin_strides.emplace_back(NewValueNode(static_cast<int64_t>(0)));
        if (ellipsis_position + j >= data_shape_.size()) {
          MS_EXCEPTION(IndexError) << "Index size out of data dims.";
        }
        (void)end_strides.emplace_back(NewValueNode(data_shape_[ellipsis_position + j]));
        (void)step_strides.emplace_back(NewValueNode(static_cast<int64_t>(1)));
      }
      index_count += ellipsis_range_size;
    }
  }

  AnfNodePtr begin_stride = res_graph_->NewCNode(begin_strides);
  AnfNodePtr end_stride = res_graph_->NewCNode(end_strides);
  AnfNodePtr step_stride = res_graph_->NewCNode(step_strides);
  auto strided_slice_vnode = MakeStridedSliceNode(shrink_axis);

  auto slice_node = res_graph_->NewCNode({strided_slice_vnode, data_node, begin_stride, end_stride, step_stride});
  res_graph_->set_output(slice_node);
}

AnfNodePtrList TensorIndexGetitem::EllipsisIndexToSlice(const std::vector<int64_t> &tuple_index_types,
                                                        const AnfNodePtr &data_node, const AnfNodePtr &begin_stride,
                                                        const AnfNodePtr &end_stride, const AnfNodePtr &step_stride) {
  auto prim = std::make_shared<Primitive>(kPrimEllipsisToSlice->name());
  prim->set_attr(kAttrTupleIndexTypes, MakeValue(tuple_index_types));
  auto ellipse_index_to_slice_node = NewValueNode(prim);
  AnfNodePtr normalized_ellipsis_node =
    res_graph_->NewCNode({ellipse_index_to_slice_node, data_node,
                          res_graph_->NewCNode({NewValueNode(kPrimMakeTuple), begin_stride, end_stride, step_stride})});
  auto new_begin_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalized_ellipsis_node, NewValueNode(SizeToLong(kIndex0))});
  auto new_end_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalized_ellipsis_node, NewValueNode(SizeToLong(kIndex1))});
  auto new_step_stride = res_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), normalized_ellipsis_node, NewValueNode(SizeToLong(kIndex2))});
  return {new_begin_stride, new_end_stride, new_step_stride};
}

void TensorIndexGetitem::GetStrideInfoFromTuple(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                                const std::vector<int64_t> &tuple_index_types,
                                                const IndexHandleLevel index_handle_level, bool has_ellipsis,
                                                const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                                size_t not_ellipsis_position_cnt, size_t ellipsis_position) {
  if (index_handle_level == IndexHandleLevel::kHandleByConstFold) {
    ConstGetStrideInfoFromTuple(data_node, index_node, tuple_index_types, has_ellipsis, tuple_abs_ptr,
                                not_ellipsis_position_cnt, ellipsis_position);
    return;
  }
  AnfNodePtrList begin_strides{NewValueNode(kPrimMakeTuple)};
  AnfNodePtrList end_strides{NewValueNode(kPrimMakeTuple)};
  AnfNodePtrList step_strides{NewValueNode(kPrimMakeTuple)};
  auto one_tensor = std::make_shared<tensor::Tensor>(std::vector<int64_t>({1}));
  auto one_tensor_node = NewValueNode(one_tensor->ToAbstract()->BuildValue());
  int64_t shrink_axis = 0;
  size_t index_count = 0;
  bool has_int = false;
  size_t ellipsis_count = 0;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    auto new_index_node =
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), index_node, NewValueNode(SizeToLong(i))});
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    if (index_type_id == kMetaTypeNone) {
      auto zero_tensor = std::make_shared<tensor::Tensor>(std::vector<int64_t>({0}));
      auto zero_tensor_node = NewValueNode(zero_tensor->ToAbstract()->BuildValue());
      (void)begin_strides.emplace_back(zero_tensor_node);
      (void)end_strides.emplace_back(one_tensor_node);
      (void)step_strides.emplace_back(one_tensor_node);
      index_count += 1;
    } else if (index_type_id == kObjectTypeTensorType) {
      new_index_node = res_graph_->NewCNode({MakeExpandDimsNode(), new_index_node, NewValueNode(0)});
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      ValueNodePtr cast_vnode = NewValueNode(cast);
      new_index_node = res_graph_->NewCNode({cast_vnode, new_index_node, NewValueNode(kInt64)});
      (void)begin_strides.emplace_back(new_index_node);
      (void)end_strides.emplace_back(res_graph_->NewCNode({NewValueNode(kPrimAdd), new_index_node, one_tensor_node}));
      (void)step_strides.emplace_back(one_tensor_node);
      shrink_axis += 1 << index_count;
      index_count += 1;
      has_int = true;
    } else if (index_type_id == kNumberTypeInt64) {
      auto new_index_item = IntIndexToTensor(data_node, new_index_node, index_abs, tuple_index_types, i, 0);
      new_index_item = res_graph_->NewCNode({MakeExpandDimsNode(), new_index_item, NewValueNode(0)});
      (void)begin_strides.emplace_back(new_index_item);
      (void)end_strides.emplace_back(res_graph_->NewCNode({NewValueNode(kPrimAdd), new_index_item, one_tensor_node}));
      (void)step_strides.emplace_back(one_tensor_node);
      shrink_axis += 1 << index_count;
      index_count += 1;
      has_int = true;
    } else if (index_type_id == kObjectTypeSlice) {
      auto [begin_stride, end_stride, step_stride] =
        NormalizeStrideInfoFromTuple(data_node, new_index_node, index_abs, tuple_index_types, i);
      auto scalar_to_tensor = NewValueNode(kPrimScalarToTensor);
      if (!IsDynamic(data_shape_) && !IsAnyValue(index_abs)) {
        begin_stride = res_graph_->NewCNode({scalar_to_tensor, begin_stride, NewValueNode(MakeValue(kInt64))});
        begin_stride = res_graph_->NewCNode({MakeExpandDimsNode(), begin_stride, NewValueNode(0)});
        end_stride = res_graph_->NewCNode({scalar_to_tensor, end_stride, NewValueNode(MakeValue(kInt64))});
        end_stride = res_graph_->NewCNode({MakeExpandDimsNode(), end_stride, NewValueNode(0)});
        step_stride = res_graph_->NewCNode({scalar_to_tensor, step_stride, NewValueNode(MakeValue(kInt64))});
        step_stride = res_graph_->NewCNode({MakeExpandDimsNode(), step_stride, NewValueNode(0)});
      }
      (void)begin_strides.emplace_back(begin_stride);
      (void)end_strides.emplace_back(end_stride);
      (void)step_strides.emplace_back(step_stride);
      index_count += 1;
    } else if (index_type_id == kMetaTypeEllipsis) {
      if (ellipsis_count >= 1) {
        MS_EXCEPTION(ValueError) << "An index can have only one ellipsis (...)";
      }
      index_count += data_shape_.size() - not_ellipsis_position_cnt;
      ellipsis_count += 1;
    }
  }
  auto concat_prim = std::make_shared<Primitive>(kPrimConcat->name());
  concat_prim->set_attr(ops::kAxis, MakeValue(static_cast<int64_t>(0)));
  AnfNodePtr begin_stride = res_graph_->NewCNode({NewValueNode(concat_prim), res_graph_->NewCNode(begin_strides)});
  AnfNodePtr end_stride = res_graph_->NewCNode({NewValueNode(concat_prim), res_graph_->NewCNode(end_strides)});
  AnfNodePtr step_stride = res_graph_->NewCNode({NewValueNode(concat_prim), res_graph_->NewCNode(step_strides)});
  if (IsDynamic(data_shape_) && has_int && has_ellipsis) {
    shrink_axis = kZeroAnfValue;
  }
  auto strided_slice_vnode = MakeStridedSliceNode(shrink_axis);
  if (has_ellipsis) {
    auto new_slice_info = EllipsisIndexToSlice(tuple_index_types, data_node, begin_stride, end_stride, step_stride);
    begin_stride = new_slice_info[kIndex0];
    end_stride = new_slice_info[kIndex1];
    step_stride = new_slice_info[kIndex2];
  }
  auto slice_node = res_graph_->NewCNode({strided_slice_vnode, data_node, begin_stride, end_stride, step_stride});

  if (IsDynamic(data_shape_) && has_int && has_ellipsis) {
    auto get_shape_prim = std::make_shared<Primitive>(kPrimGetSqueezeSliceShape->name());
    get_shape_prim->set_attr(kAttrTupleIndexTypes, MakeValue(tuple_index_types));
    auto get_shape_node = NewValueNode(get_shape_prim);
    auto slice_shape_node = res_graph_->NewCNode({get_shape_node, slice_node});
    slice_node = res_graph_->NewCNode({MakeReshapeNode(), slice_node, slice_shape_node});
  }
  res_graph_->set_output(slice_node);
}

void TensorIndex::RemakeTupleIndex(bool has_ellipsis, const std::vector<int64_t> &tuple_index_types,
                                   const AnfNodePtr &data_node, const std::vector<AnfNodePtr> &new_normalized_tensors,
                                   size_t not_ellipsis_position_cnt, size_t ellipsis_position) {
  if (IsDynamicRank(data_shape_) && has_ellipsis) {
    auto prim = std::make_shared<Primitive>(kPrimRemakeTupleIndex->name());
    prim->set_attr(kAttrTupleIndexTypes, MakeValue(tuple_index_types));
    std::vector<AnfNodePtr> remake_tuple_inputs = {NewValueNode(prim)};
    (void)remake_tuple_inputs.emplace_back(data_node);
    (void)remake_tuple_inputs.insert(remake_tuple_inputs.end(), new_normalized_tensors.begin(),
                                     new_normalized_tensors.end());
    AnfNodePtr indices_node = NewCNode(remake_tuple_inputs, res_graph_);
    auto gather_nd_node = NewCNode({NewValueNode(kPrimGatherNd), data_node, indices_node}, res_graph_);
    res_graph_->set_output(gather_nd_node);
  } else {
    std::vector<AnfNodePtr> remake_tuple_inputs(new_normalized_tensors.begin(),
                                                new_normalized_tensors.begin() + SizeToLong(not_ellipsis_position_cnt));
    (void)remake_tuple_inputs.insert(remake_tuple_inputs.begin() + SizeToLong(ellipsis_position),
                                     new_normalized_tensors.begin() + SizeToLong(not_ellipsis_position_cnt),
                                     new_normalized_tensors.end());
    (void)remake_tuple_inputs.insert(remake_tuple_inputs.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto remake_tuple = NewCNode(remake_tuple_inputs, res_graph_);
    auto concat_prim = std::make_shared<Primitive>(kPrimConcat->name());
    concat_prim->set_attr(ops::kAxis, MakeValue(static_cast<int64_t>(-1)));
    AnfNodePtr indices_node = res_graph_->NewCNode({NewValueNode(concat_prim), remake_tuple});
    auto gather_nd_node = NewCNode({NewValueNode(kPrimGatherNd), data_node, indices_node}, res_graph_);
    res_graph_->set_output(gather_nd_node);
  }
}

void TensorIndexGetitem::GetItemByTuple(const AnfNodePtr &input_data_node, const AnfNodePtr &index_node,
                                        const AbstractBasePtr &data, const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                        const AbstractBasePtr &all_empty_tensor_index) {
  if (tuple_abs_ptr->empty()) {
    res_graph_->set_output(input_data_node);
  }
  IndexHandleLevel index_handle_level = PreHandleIndex(data, tuple_abs_ptr);
  // Get type of each index in tuple.
  bool has_ellipsis = false;
  size_t ellipsis_position = 0;
  size_t not_ellipsis_position_cnt = 0;
  std::bitset<kMaxDimNums> expand_dims_mask;
  auto tuple_index_types = GetTupleIndexType(tuple_abs_ptr, data_shape_, &has_ellipsis, &ellipsis_position,
                                             &not_ellipsis_position_cnt, &expand_dims_mask);
  size_t expand_dims_cnt = expand_dims_mask.count();
  // Expand dims if there are bool/None index
  auto data_node = ExpandDimsByTupleIndex(input_data_node, tuple_abs_ptr, tuple_index_types, expand_dims_cnt);
  auto data_dim = data_shape_.size();
  constexpr int min_data_dim = 1;
  constexpr int max_data_dim = 8;
  if (data_dim < min_data_dim || data_dim > max_data_dim) {
    MS_EXCEPTION(ValueError) << "The input data's dim must in the range of [" << min_data_dim << ", " << max_data_dim
                             << "], but got '" << data_dim << "'.";
  }
  const auto &indices_abs = tuple_abs_ptr->elements();
  if (std::all_of(indices_abs.begin(), indices_abs.end(), [](AbstractBasePtr index_abs) {
        if (index_abs->BuildType()->type_id() == kObjectTypeTensorType) {
          auto index_shape = index_abs->BuildShape()->cast<abstract::ShapePtr>()->shape();
          return index_shape.empty();
        }
        return CheckTypeIsInstance<TypeId>(index_abs->BuildType()->type_id(),
                                           {kNumberTypeInt64, kObjectTypeSlice, kMetaTypeEllipsis, kMetaTypeNone});
      })) {
    GetStrideInfoFromTuple(data_node, index_node, tuple_index_types, index_handle_level, has_ellipsis, tuple_abs_ptr,
                           not_ellipsis_position_cnt, ellipsis_position);
    return;
  }
  MS_LOG(DEBUG) << "Tuple index types in TensorIndexing is: " << tuple_index_types;
  auto normalized_tensors =
    NormalizeTupleIndex(data_node, index_node, tuple_index_types, index_handle_level, has_ellipsis, tuple_abs_ptr);

  mindspore::HashMap<std::string, ValuePtr> attrs(
    {{kAttrTupleIndexTypes, MakeValue(tuple_index_types)}, {kAttrExpandDimsCnt, MakeValue(SizeToLong(0))}});
  auto tuple_index_info_node = GetTupleIndexInfo(data_node, NewValueNode(SizeToLong(0)), normalized_tensors, attrs);
  auto broad_cast_shape_node = tuple_index_info_node[kIndex0];
  auto new_index_shape_node = tuple_index_info_node[kIndex1];
  auto final_shape_node = tuple_index_info_node[kIndex2];
  auto tensor_index_transfer =
    prim::GetPythonOps("_tuple_index_transfer", "mindspore.ops.composite.multitype_ops._compile_utils");
  auto broadcast_to = prim::GetPythonOps("broadcast_to", "mindspore.ops.function.array_func");
  ValueNodePtr tensor_index_transfer_node = NewValueNode(tensor_index_transfer);
  ValueNodePtr broadcast_to_node = NewValueNode(broadcast_to);
  size_t slice_index_count = 0;
  std::vector<AnfNodePtr> new_normalized_tensors{};
  auto new_tuple_index_types = tuple_index_types;
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (new_tuple_index_types[i] == kMetaTypeEllipsis) {
      (void)new_tuple_index_types.erase(new_tuple_index_types.begin() + i);
      (void)new_tuple_index_types.emplace_back(kMetaTypeEllipsis);
      break;
    }
  }
  for (size_t i = 0; i < normalized_tensors.size(); i++) {
    AnfNodePtr new_tensor_index = normalized_tensors[i];
    if (new_tuple_index_types[i] == kObjectTypeTensorType) {
      new_tensor_index =
        NewCNode({tensor_index_transfer_node, broad_cast_shape_node, final_shape_node, new_index_shape_node,
                  new_tensor_index, NewValueNode(all_empty_tensor_index->BuildValue())},
                 res_graph_);
    } else {
      auto new_slice_shape_node = tuple_index_info_node[kIndex5 + slice_index_count];
      new_tensor_index = NewCNode({MakeReshapeNode(), new_tensor_index, new_slice_shape_node}, res_graph_);
      new_tensor_index = NewCNode({broadcast_to_node, new_tensor_index, final_shape_node}, res_graph_);
      slice_index_count += 1;
    }
    if (!IsDynamicRank(data_shape_) || !has_ellipsis) {
      new_tensor_index =
        res_graph_->NewCNode({MakeExpandDimsNode(), new_tensor_index, NewValueNode(static_cast<int64_t>(-1))});
    }
    (void)new_normalized_tensors.emplace_back(new_tensor_index);
  }
  RemakeTupleIndex(has_ellipsis, tuple_index_types, data_node, new_normalized_tensors, not_ellipsis_position_cnt,
                   ellipsis_position);
}

void TensorIndexSetitem::SetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                        const AnfNodePtr &value_node, const AbstractBasePtr &data,
                                        const abstract::AbstractSlicePtr &abs_slice_ptr, const AbstractBasePtr &value) {
  AnfNodePtr value_shape_node;
  AnfNodePtrList output_nodes{NewValueNode(kPrimMakeTuple)};
  std::vector<int64_t> init_by_none;
  auto stride_slice_input = ParseSlice(index_node, abs_slice_ptr, &init_by_none);
  AnfNodePtrList slice_to_indices{MakeSliceToIndicesNode(0, {}, 0, init_by_none), data_node};
  const size_t slice_to_indices_output_size = 5;
  (void)slice_to_indices.insert(slice_to_indices.end(), stride_slice_input.begin(), stride_slice_input.end());
  auto slice_to_indices_node = res_graph_->NewCNode(slice_to_indices);
  for (size_t i = 0; i < slice_to_indices_output_size; i++) {
    (void)output_nodes.emplace_back(
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), slice_to_indices_node, NewValueNode(SizeToLong(i))}));
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
  (void)output_nodes.emplace_back(new_value_node);
  res_graph_->set_output(res_graph_->NewCNode(output_nodes));
}

AnfNodePtr PreSetitemByTuple::FormatIndex(const abstract::AbstractBasePtr &index_abs, const AnfNodePtr &data_node,
                                          const AnfNodePtr &index_node, size_t cur_dim,
                                          const std::vector<int64_t> &tuple_index_types, int64_t expand_dims_mask,
                                          bool *empty_sequence) {
  AnfNodePtr new_index_node = index_node;
  const TypeId index_type_id = index_abs->BuildType()->type_id();
  if (CheckTypeIsInstance<TypeId>(index_type_id,
                                  {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64})) {
    new_index_node =
      IntIndexToTensor(data_node, new_index_node, index_abs, tuple_index_types, cur_dim, expand_dims_mask);
  } else if (CheckTypeIsInstance<TypeId>(index_type_id, {kObjectTypeList, kObjectTypeTuple})) {
    new_index_node = SequenceIndexToTensor(data_node, new_index_node, tuple_index_types, index_abs, cur_dim,
                                           expand_dims_mask, empty_sequence);
  } else if (index_type_id == kObjectTypeTensorType) {
    auto tensor_abs = dyn_cast<abstract::AbstractTensor>(index_abs);
    if (CheckTypeIsInstance<TypeId>(tensor_abs->element()->BuildType()->type_id(),
                                    {kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeInt64})) {
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      ValueNodePtr cast_vnode = NewValueNode(cast);
      new_index_node = res_graph_->NewCNode({cast_vnode, index_node, NewValueNode(kInt64)});
      if (!IsDynamic(data_shape_)) {
        size_t normalize_dim_index = NormalizeDimIndex(data_shape_.size(), cur_dim, tuple_index_types);
        // Equal to python: idx = F.select(idx < 0, idx + data_shape[cur_dim], idx)
        auto less = prim::GetPythonOps("less", "mindspore.ops.functional");
        ValueNodePtr less_vnode = NewValueNode(less);
        auto less_cnode = res_graph_->NewCNode({less_vnode, new_index_node, NewValueNode(static_cast<int64_t>(0))});
        auto add = prim::GetPythonOps("add", "mindspore.ops.functional");
        ValueNodePtr add_vnode = NewValueNode(add);
        auto add_cnode =
          res_graph_->NewCNode({add_vnode, new_index_node, NewValueNode(data_shape_[normalize_dim_index])});
        auto select = prim::GetPythonOps("select", "mindspore.ops.functional");
        ValueNodePtr select_vnode = NewValueNode(select);
        new_index_node = res_graph_->NewCNode({select_vnode, less_cnode, add_cnode, new_index_node});
      }
    } else if (tensor_abs->element()->BuildType()->type_id() != kNumberTypeBool) {
      MS_EXCEPTION(IndexError) << "The tensor element in tuple index must be int or bool type, but got"
                               << tensor_abs->element()->BuildType();
    }
  }
  return new_index_node;
}

std::vector<CNodePtr> TensorIndex::GetTupleIndexInfo(const AnfNodePtr &data_node, const AnfNodePtr &fancy_position_node,
                                                     const std::vector<AnfNodePtr> &normalized_tensors,
                                                     const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto get_tuple_index_info_prim = std::make_shared<Primitive>(kPrimGetTupleIndexInfo->name());
  get_tuple_index_info_prim->SetAttrs(attrs);
  auto get_tuple_index_info_node = NewValueNode(get_tuple_index_info_prim);
  AnfNodePtrList get_tuple_index_info_inputs{get_tuple_index_info_node, data_node, fancy_position_node};
  (void)get_tuple_index_info_inputs.insert(get_tuple_index_info_inputs.end(), normalized_tensors.begin(),
                                           normalized_tensors.end());
  for (size_t i = normalized_tensors.size(); i < 8; i++) {
    (void)get_tuple_index_info_inputs.emplace_back(NewValueNode(std::vector<int64_t>{1}));
  }
  auto tuple_index_info_node = NewCNode(get_tuple_index_info_inputs, res_graph_);
  // {broadcast_shape, new_index_shape_node, final_shape_node, fancy_position, zero_dim_tensor, new_slice_shape_nodes*8}
  const size_t tuple_index_info_nums = 13;
  std::vector<CNodePtr> output_nodes;
  for (size_t i = 0; i <= tuple_index_info_nums; i++) {
    auto index_info_node =
      res_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), tuple_index_info_node, NewValueNode(SizeToLong(i))});
    (void)output_nodes.emplace_back(index_info_node);
  }
  return output_nodes;
}

void PreSetitemByTuple::RemoveExpandedDims(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                           const AnfNodePtr &value_node, const std::vector<int64_t> &tuple_index_types,
                                           const IndexHandleLevel index_handle_level, const bool has_ellipsis,
                                           const abstract::AbstractTuplePtr &tuple_abs_ptr, int64_t expand_dims_mask) {
  std::vector<AnfNodePtr> indices_out_list{NewValueNode(kPrimMakeTuple)};
  std::vector<AnfNodePtr> normalized_tensors;
  auto sub_tensor = std::make_shared<tensor::Tensor>(SizeToLong(1));
  auto sub_tensor_node = NewValueNode(sub_tensor->ToAbstract()->BuildValue());
  bool has_true = false;
  AnfNodePtr has_false_node = NewValueNode(std::make_shared<tensor::Tensor>(static_cast<int64_t>(0)));
  auto add = prim::GetPythonOps("add", "mindspore.ops.function.math_func");
  ValueNodePtr add_vnode = NewValueNode(add);
  bool has_sequence = false;
  std::vector<int64_t> new_tuple_index_types = tuple_index_types;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    AnfNodePtr new_index_node =
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), index_node, NewValueNode(SizeToLong(i))});
    AnfNodePtr index_item = NewValueNode(index_abs->BuildValue());
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    bool empty_sequence = false;
    new_index_node =
      FormatIndex(index_abs, data_node, new_index_node, i, tuple_index_types, expand_dims_mask, &empty_sequence);
    has_false_node =
      res_graph_->NewCNode({add_vnode, has_false_node, NewValueNode(std::make_shared<tensor::Tensor>(empty_sequence))});
    if (index_type_id == kMetaTypeNone) {
      (void)normalized_tensors.emplace_back(sub_tensor_node);
      new_tuple_index_types[i] = kMetaTypeNone;
    } else if (index_type_id == kObjectTypeSlice) {
      (void)normalized_tensors.emplace_back(sub_tensor_node);
      (void)indices_out_list.emplace_back(new_index_node);
      auto abs_slice_ptr = dyn_cast<abstract::AbstractSlice>(index_abs);
      std::vector<int64_t> init_by_none;
      auto stride_slice_input = ParseSlice(new_index_node, abs_slice_ptr, &init_by_none);
      auto slice_to_indices_prim = std::make_shared<Primitive>(kPrimSliceToIndices->name());
      AnfNodePtrList slice_to_indices{MakeSliceToIndicesNode(i, tuple_index_types, expand_dims_mask, init_by_none),
                                      data_node};
      (void)slice_to_indices.insert(slice_to_indices.end(), stride_slice_input.begin(), stride_slice_input.end());
      auto slice_to_indices_node = res_graph_->NewCNode(slice_to_indices);
      auto empty_slice_node = res_graph_->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), slice_to_indices_node, NewValueNode(SizeToLong(kIndex5))});
      has_false_node = res_graph_->NewCNode({add_vnode, has_false_node, empty_slice_node});
    } else if (index_type_id == kObjectTypeTensorType) {
      (void)normalized_tensors.emplace_back(new_index_node);
      (void)indices_out_list.emplace_back(new_index_node);
      auto tensor_shape = index_abs->BuildShape()->cast<abstract::ShapePtr>()->shape();
      if (IsDynamicRank(tensor_shape)) {
        MS_EXCEPTION(IndexError) << "Tensor index in tuple can not be dynamic rank.";
      }
      if (tensor_shape.size() > 0) {
        has_sequence = true;
      }
    } else if (index_type_id == kNumberTypeInt64) {
      (void)normalized_tensors.emplace_back(new_index_node);
      (void)indices_out_list.emplace_back(new_index_node);
    } else if (index_type_id == kNumberTypeBool) {
      (void)normalized_tensors.emplace_back(sub_tensor_node);
      has_true = has_true || GetValue<bool>(GetValueNode(index_item));
      has_false_node = res_graph_->NewCNode({add_vnode, has_false_node,
                                             NewValueNode(std::make_shared<tensor::Tensor>(
                                               static_cast<int64_t>(!GetValue<bool>(GetValueNode(index_item)))))});
    } else if (index_type_id == kObjectTypeList || index_type_id == kObjectTypeTuple) {
      (void)normalized_tensors.emplace_back(new_index_node);
      (void)indices_out_list.emplace_back(new_index_node);
      auto sequence_abs = dyn_cast<abstract::AbstractSequence>(index_abs);
      if (sequence_abs->size() > 0) {
        has_sequence = true;
      }
    } else if (index_type_id == kMetaTypeEllipsis) {
      (void)indices_out_list.emplace_back(new_index_node);
    } else {
      MS_EXCEPTION(IndexError) << "invalid index type";
    }
  }
  mindspore::HashMap<std::string, ValuePtr> attrs(
    {{kAttrTupleIndexTypes, MakeValue(tuple_index_types)},
     {kAttrTupleIndexInfoType, MakeValue(kPreSetitemByTuple)},
     {kAttrExpandDimsCnt, MakeValue(SizeToLong(std::bitset<8>(expand_dims_mask).count()))}});
  auto tuple_index_infos = GetTupleIndexInfo(data_node, NewValueNode(SizeToLong(0)), normalized_tensors, attrs);
  auto broadcast_shape = tuple_index_infos[kIndex0];
  auto fancy_position = tuple_index_infos[kIndex3];

  auto indices_out = res_graph_->NewCNode(indices_out_list);
  auto rem_not_expanded_dims_node = res_graph_->NewCNode(
    {MakeRemoveExpandedDimsNode(has_true, has_sequence, indices_out_list, expand_dims_mask, new_tuple_index_types),
     data_node, value_node, has_false_node, broadcast_shape, fancy_position});
  auto indices_out_type =
    res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), rem_not_expanded_dims_node, NewValueNode(SizeToLong(0))});
  auto value_shape =
    res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), rem_not_expanded_dims_node, NewValueNode(SizeToLong(1))});
  auto idx_advanced =
    res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), rem_not_expanded_dims_node, NewValueNode(SizeToLong(2))});
  auto output = res_graph_->NewCNode(
    {NewValueNode(kPrimMakeTuple), indices_out_type, indices_out, value_shape, idx_advanced, broadcast_shape});
  res_graph_->set_output(output);
}

void TensorIndexSetitem::SetItemByTuple(const AnfNodePtr &input_data_node, const AnfNodePtr &index_node,
                                        const AnfNodePtr &value_node, const AnfNodePtr &fancy_position_node,
                                        const AbstractBasePtr &data, const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                        const AbstractBasePtr &value, const AbstractBasePtr &all_empty_tensor_flag) {
  auto data_node = input_data_node;
  IndexHandleLevel index_handle_level = PreHandleIndex(data, tuple_abs_ptr);

  // Get type of each index in tuple.
  bool has_ellipsis = false;
  size_t ellipsis_position = 0;
  size_t not_ellipsis_position_cnt = 0;

  std::bitset<kMaxDimNums> expand_dims_mask;
  auto tuple_index_types = GetTupleIndexType(tuple_abs_ptr, data_shape_, &has_ellipsis, &ellipsis_position,
                                             &not_ellipsis_position_cnt, &expand_dims_mask);

  // Get ellipse_occupy_dims_cnt
  auto normalized_tensors =
    NormalizeTupleIndex(data_node, index_node, tuple_index_types, index_handle_level, has_ellipsis, tuple_abs_ptr);
  mindspore::HashMap<std::string, ValuePtr> attrs(
    {{kAttrTupleIndexTypes, MakeValue(tuple_index_types)}, {kAttrExpandDimsCnt, MakeValue(SizeToLong(0))}});
  if (std::all_of(tuple_index_types.begin(), tuple_index_types.end(), [](const auto &index_type) {
        return index_type == kObjectTypeTensorType || index_type == kTypeUnknown;
      })) {
    attrs.insert(attrs.end(), {kAttrTupleIndexInfoType, MakeValue(kSetitemByTupleWithTensor)});
  } else {
    attrs.insert(attrs.end(), {kAttrTupleIndexInfoType, MakeValue(kSetitemByTuple)});
  }
  auto tuple_index_info_node = GetTupleIndexInfo(data_node, fancy_position_node, normalized_tensors, attrs);
  auto broad_cast_shape_node = tuple_index_info_node[kIndex0];
  auto new_index_shape_node = tuple_index_info_node[kIndex1];
  auto final_shape_node = tuple_index_info_node[kIndex2];
  auto tensor_index_transfer =
    prim::GetPythonOps("_tuple_index_transfer", "mindspore.ops.composite.multitype_ops._compile_utils");
  auto broadcast_to = prim::GetPythonOps("broadcast_to", "mindspore.ops.function.array_func");
  ValueNodePtr tensor_index_transfer_node = NewValueNode(tensor_index_transfer);

  ValueNodePtr broadcast_to_node = NewValueNode(broadcast_to);
  size_t slice_index_count = 0;
  std::vector<AnfNodePtr> new_normalized_tensors{};
  auto new_tuple_index_types = tuple_index_types;
  for (size_t i = 0; i < tuple_index_types.size(); i++) {
    if (new_tuple_index_types[i] == kMetaTypeEllipsis) {
      (void)new_tuple_index_types.erase(new_tuple_index_types.begin() + i);
      (void)new_tuple_index_types.emplace_back(kMetaTypeEllipsis);
      break;
    }
  }
  for (size_t i = 0; i < normalized_tensors.size(); i++) {
    AnfNodePtr new_tensor_index = normalized_tensors[i];
    if (new_tuple_index_types[i] == kObjectTypeTensorType) {
      new_tensor_index =
        NewCNode({tensor_index_transfer_node, broad_cast_shape_node, final_shape_node, new_index_shape_node,
                  new_tensor_index, NewValueNode(all_empty_tensor_flag->BuildValue())},
                 res_graph_);
    } else {
      auto new_slice_shape_node = tuple_index_info_node[kIndex5 + slice_index_count];
      new_tensor_index = NewCNode({MakeReshapeNode(), new_tensor_index, new_slice_shape_node}, res_graph_);
      new_tensor_index = NewCNode({broadcast_to_node, new_tensor_index, final_shape_node}, res_graph_);
      slice_index_count += 1;
    }
    if (!IsDynamicRank(data_shape_) || !has_ellipsis) {
      new_tensor_index =
        res_graph_->NewCNode({MakeExpandDimsNode(), new_tensor_index, NewValueNode(static_cast<int64_t>(-1))});
    }
    (void)new_normalized_tensors.emplace_back(new_tensor_index);
  }
  if (IsDynamicRank(data_shape_) && has_ellipsis) {
    auto prim = std::make_shared<Primitive>(kPrimRemakeTupleIndex->name());
    prim->set_attr(kAttrTupleIndexTypes, MakeValue(tuple_index_types));
    std::vector<AnfNodePtr> remake_tuple_inputs = {NewValueNode(prim)};
    (void)remake_tuple_inputs.emplace_back(data_node);
    (void)remake_tuple_inputs.insert(remake_tuple_inputs.end(), new_normalized_tensors.begin(),
                                     new_normalized_tensors.end());
    AnfNodePtr indices_node = NewCNode(remake_tuple_inputs, res_graph_);
    res_graph_->set_output(indices_node);
  } else {
    std::vector<AnfNodePtr> remake_tuple_inputs(new_normalized_tensors.begin(),
                                                new_normalized_tensors.begin() + SizeToLong(not_ellipsis_position_cnt));
    (void)remake_tuple_inputs.insert(remake_tuple_inputs.begin() + SizeToLong(ellipsis_position),
                                     new_normalized_tensors.begin() + SizeToLong(not_ellipsis_position_cnt),
                                     new_normalized_tensors.end());

    (void)remake_tuple_inputs.insert(remake_tuple_inputs.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto remake_tuple = NewCNode(remake_tuple_inputs, res_graph_);

    auto concat_prim = std::make_shared<Primitive>(kPrimConcat->name());
    concat_prim->set_attr(ops::kAxis, MakeValue(static_cast<int64_t>(-1)));
    AnfNodePtr indices_node = res_graph_->NewCNode({NewValueNode(concat_prim), remake_tuple});
    res_graph_->set_output(indices_node);
  }
  MS_LOG(DEBUG) << "Tuple index types in TensorIndexing is: " << tuple_index_types;
}

FuncGraphPtr TensorIndexSetitem::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t max_args_size = 5;
  if (arg_length > max_args_size) {
    MS_LOG(EXCEPTION) << "The TensorIndexSetitem operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("TensorIndexSetitem");
  AnfNodePtr data_node = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();
  AnfNodePtr value_node = res_graph_->add_parameter();

  if (args_abs_list[1]->isa<abstract::AbstractSlice>()) {
    (void)SetItemBySlice(data_node, index_node, value_node, args_abs_list[0],
                         dyn_cast<abstract::AbstractSlice>(args_abs_list[kIndex1]), args_abs_list[kIndex2]);
  }

  if (args_abs_list[1]->isa<abstract::AbstractTuple>()) {
    AnfNodePtr fancy_position_node = res_graph_->add_parameter();
    (void)res_graph_->add_parameter();
    (void)SetItemByTuple(data_node, index_node, value_node, fancy_position_node, args_abs_list[0],
                         dyn_cast<abstract::AbstractTuple>(args_abs_list[kIndex1]), args_abs_list[kIndex2],
                         args_abs_list[kIndex4]);
  }
  return res_graph_;
}

AnfNodePtr TensorIndex::ExpandDimsByTupleIndex(const AnfNodePtr &input_data_node,
                                               const abstract::AbstractTuplePtr &tuple_abs_ptr,
                                               const std::vector<int64_t> &tuple_index_types, size_t expand_dims_cnt) {
  // Expand dims if there are bool/None index
  auto data_node = input_data_node;
  size_t data_dim = data_shape_.size() + expand_dims_cnt;
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    AnfNodePtr index_item;
    index_item = NewValueNode(index_abs->BuildValue());
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    if (index_type_id == kMetaTypeNone || index_type_id == kNumberTypeBool) {
      if (!IsDynamicRank(data_shape_)) {
        size_t normalize_dim_index = NormalizeDimIndex(data_dim, i, tuple_index_types);
        (void)data_shape_.insert(data_shape_.begin() + normalize_dim_index, 1);
      }
      auto normalize_dim_index_prim = std::make_shared<Primitive>(kPrimNormalizeDimIndex->name());
      normalize_dim_index_prim->set_attr(kAttrTupleIndexTypes, MakeValue(tuple_index_types));
      normalize_dim_index_prim->set_attr(kAttrExpandDimsCnt, MakeValue(SizeToLong(expand_dims_cnt)));
      normalize_dim_index_prim->set_attr(kAttrTupleIndexAxis, MakeValue(SizeToLong(i)));
      auto normalize_dim_index_node = NewValueNode(normalize_dim_index_prim);
      auto normalize_axis_cnode = res_graph_->NewCNode({normalize_dim_index_node, data_node});
      expand_dims_cnt -= 1;
      data_node = res_graph_->NewCNode({MakeExpandDimsNode(), data_node, normalize_axis_cnode});
    }
  }
  return data_node;
}

void HandleEmptySlice::HandleEmptySliceByTupleIndex(const AnfNodePtr &input_data_node, const AnfNodePtr &index_node,
                                                    const AbstractBasePtr &data,
                                                    const abstract::AbstractTuplePtr &tuple_abs_ptr) {
  //  auto data_node = input_data_node;
  if (tuple_abs_ptr->empty()) {
    res_graph_->set_output(input_data_node);
  }
  IndexHandleLevel index_handle_level = PreHandleIndex(data, tuple_abs_ptr);

  // Get type of each index in tuple.
  bool has_ellipsis = false;
  size_t ellipsis_position = 0;
  size_t not_ellipsis_position_cnt = 0;
  std::bitset<kMaxDimNums> expand_dims_mask;
  auto tuple_index_types = GetTupleIndexType(tuple_abs_ptr, data_shape_, &has_ellipsis, &ellipsis_position,
                                             &not_ellipsis_position_cnt, &expand_dims_mask);
  // Expand dims if there are bool/None index
  size_t expand_dims_cnt = expand_dims_mask.count();
  auto data_node = ExpandDimsByTupleIndex(input_data_node, tuple_abs_ptr, tuple_index_types, expand_dims_cnt);

  if (data_shape_.size() < 1 || data_shape_.size() > kMaxDimNums) {
    MS_EXCEPTION(ValueError) << "The input data's dim must in the range of [1, 8], but got '" << data_shape_.size()
                             << "'.";
  }
  MS_LOG(DEBUG) << "Tuple index types in TensorIndexing is: " << tuple_index_types;
  auto normalized_tensors =
    NormalizeTupleIndex(data_node, index_node, tuple_index_types, index_handle_level, has_ellipsis, tuple_abs_ptr);

  mindspore::HashMap<std::string, ValuePtr> attrs(
    {{kAttrTupleIndexTypes, MakeValue(tuple_index_types)}, {kAttrExpandDimsCnt, MakeValue(SizeToLong(0))}});
  auto tuple_index_info_node = GetTupleIndexInfo(data_node, NewValueNode(SizeToLong(0)), normalized_tensors, attrs);
  auto final_shape_node = tuple_index_info_node[kIndex2];
  auto zero_dim_tensor = tuple_index_info_node[kIndex4];
  res_graph_->set_output(res_graph_->NewCNode({NewValueNode(kPrimMakeTuple), final_shape_node, zero_dim_tensor}));
}

FuncGraphPtr HandleEmptySlice::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 2;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The HandleZeroTupleIndex operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("HandleZeroTupleIndex");
  AnfNodePtr data_node = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();

  if (args_abs_list[1]->isa<abstract::AbstractTuple>()) {
    (void)HandleEmptySliceByTupleIndex(data_node, index_node, args_abs_list[0],
                                       dyn_cast<abstract::AbstractTuple>(args_abs_list[1]));
  }
  return res_graph_;
}

FuncGraphPtr HandleScalarTensorIndex::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 2;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The HandleBoolTensor operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("HandleScalarTensorIndex");
  AnfNodePtr data = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();

  auto tuple_abs_ptr = dyn_cast<abstract::AbstractTuple>(args_abs_list[1]);
  IndexHandleLevel index_handle_level = PreHandleIndex(args_abs_list[0], tuple_abs_ptr);
  // Get type of each index in tuple.
  bool has_ellipsis = false;
  size_t ellipsis_position = 0;
  size_t not_ellipsis_position_cnt = 0;
  std::bitset<kMaxDimNums> expand_dims_mask;

  auto tuple_index_types = GetTupleIndexType(tuple_abs_ptr, data_shape_, &has_ellipsis, &ellipsis_position,
                                             &not_ellipsis_position_cnt, &expand_dims_mask);
  size_t expand_dims_cnt = expand_dims_mask.count();
  // Expand dims if there are bool/None index
  auto data_node = ExpandDimsByTupleIndex(data, tuple_abs_ptr, tuple_index_types, expand_dims_cnt);

  MS_LOG(DEBUG) << "Tuple index types in TensorIndexing is: " << tuple_index_types;
  auto normalized_tensors =
    NormalizeTupleIndex(data_node, index_node, tuple_index_types, index_handle_level, has_ellipsis, tuple_abs_ptr);

  mindspore::HashMap<std::string, ValuePtr> attrs(
    {{kAttrTupleIndexTypes, MakeValue(tuple_index_types)}, {kAttrExpandDimsCnt, MakeValue(SizeToLong(0))}});
  auto tuple_index_info_node = GetTupleIndexInfo(data_node, NewValueNode(SizeToLong(0)), normalized_tensors, attrs);
  auto broad_cast_shape_node = tuple_index_info_node[kIndex0];
  res_graph_->set_output(broad_cast_shape_node);
  return res_graph_;
}

FuncGraphPtr HandleBoolTensor::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 1;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The HandleBoolTensor operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("HandleBoolTensor");

  AnfNodePtr index_node = res_graph_->add_parameter();

  auto tuple_abs_ptr = dyn_cast<abstract::AbstractTuple>(args_abs_list[0]);

  std::vector<AnfNodePtr> indices_out_list{NewValueNode(kPrimMakeTuple)};
  std::vector<AnfNodePtr> indices_out_list_with_zero{NewValueNode(kPrimMakeTuple)};

  std::vector<AnfNodePtr> non_zero_shape_list{NewValueNode(kPrimMakeTuple)};
  tensor::TensorPtr zero_shape_tensor_index = std::make_shared<tensor::Tensor>(kNumberTypeInt32, ShapeVector({0}));
  for (size_t i = 0; i < tuple_abs_ptr->size(); i++) {
    const auto &index_abs = tuple_abs_ptr->elements()[i];
    AnfNodePtr new_index_node =
      res_graph_->NewCNode({NewValueNode(kPrimTupleGetItem), index_node, NewValueNode(SizeToLong(i))});
    AnfNodePtr index_item = NewValueNode(index_abs->BuildValue());
    const TypeId index_type_id = index_abs->BuildType()->type_id();
    if (index_type_id == kObjectTypeTensorType) {
      auto tensor_abs = dyn_cast<abstract::AbstractTensor>(index_abs);
      if (tensor_abs->element()->BuildType()->type_id() == kNumberTypeBool) {
        auto tensor_shape = index_abs->BuildShape()->cast<abstract::ShapePtr>()->shape();
        if (IsDynamicRank(tensor_shape)) {
          MS_EXCEPTION(IndexError) << "Tensor index in tuple can not be dynamic rank.";
        }
        new_index_node = res_graph_->NewCNode({NewValueNode(kPrimNonZero), new_index_node});
        (void)non_zero_shape_list.emplace_back(res_graph_->NewCNode({NewValueNode(kPrimTensorShape), new_index_node}));
        for (size_t j = 0; j < tensor_shape.size(); j++) {
          auto gather_index_tensor = std::make_shared<tensor::Tensor>(SizeToLong(j));
          auto gather_index_tensor_node = NewValueNode(gather_index_tensor->ToAbstract()->BuildValue());
          auto bool_tensor_index_node = res_graph_->NewCNode(
            {MakeGatherNode(), new_index_node, gather_index_tensor_node, NewValueNode(SizeToLong(1))});
          bool_tensor_index_node =
            res_graph_->NewCNode({MakeReshapeNode(), bool_tensor_index_node, NewValueNode(std::vector<int64_t>{-1})});
          (void)indices_out_list.emplace_back(bool_tensor_index_node);
          (void)indices_out_list_with_zero.emplace_back(NewValueNode(zero_shape_tensor_index));
        }
      } else {
        (void)indices_out_list.emplace_back(new_index_node);
        (void)indices_out_list_with_zero.emplace_back(new_index_node);
      }
    } else {
      (void)indices_out_list.emplace_back(new_index_node);
      (void)indices_out_list_with_zero.emplace_back(new_index_node);
    }
  }
  auto new_indices_node = res_graph_->NewCNode(indices_out_list);
  auto new_indices_with_zero_node = res_graph_->NewCNode(indices_out_list_with_zero);
  auto non_zero_shape_node = res_graph_->NewCNode(non_zero_shape_list);
  std::vector<AnfNodePtr> out_list{NewValueNode(kPrimMakeTuple), new_indices_node, new_indices_with_zero_node,
                                   non_zero_shape_node};
  res_graph_->set_output(res_graph_->NewCNode(out_list));
  return res_graph_;
}

FuncGraphPtr PreSetitemByTuple::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t min_args_size = 3;
  if (arg_length != min_args_size) {
    MS_LOG(EXCEPTION) << "The PreSetitemByTuple operator requires arguments, but got " << arg_length << ".";
  }
  res_graph_ = std::make_shared<FuncGraph>();
  res_graph_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  res_graph_->debug_info()->set_name("HandleZeroTupleIndex");
  AnfNodePtr data_node = res_graph_->add_parameter();
  AnfNodePtr index_node = res_graph_->add_parameter();
  AnfNodePtr value_node = res_graph_->add_parameter();
  auto data = args_abs_list[0];
  auto tuple_abs_ptr = dyn_cast<abstract::AbstractTuple>(args_abs_list[1]);

  IndexHandleLevel index_handle_level = PreHandleIndex(data, tuple_abs_ptr);
  // Get type of each index in tuple.
  bool has_ellipsis = false;
  size_t ellipsis_position = 0;
  size_t not_ellipsis_position_cnt = 0;
  std::bitset<kMaxDimNums> expand_dims_mask;
  auto tuple_index_types = GetTupleIndexType(tuple_abs_ptr, data_shape_, &has_ellipsis, &ellipsis_position,
                                             &not_ellipsis_position_cnt, &expand_dims_mask);

  // Get ellipse_occupy_dims_cnt
  MS_LOG(DEBUG) << "Tuple index types in TensorIndexing is: " << tuple_index_types;
  RemoveExpandedDims(data_node, index_node, value_node, tuple_index_types, index_handle_level, has_ellipsis,
                     tuple_abs_ptr, expand_dims_mask.to_ulong());
  return res_graph_;
}

}  // namespace prim
}  // namespace mindspore
