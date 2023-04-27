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

static AbstractBasePtr VectorToTuple(const std::vector<int64_t> &nums) {
  abstract::AbstractBasePtrList elems;
  std::transform(nums.begin(), nums.end(), std::back_inserter(elems),
                 [](int64_t num) { return std::make_shared<abstract::AbstractScalar>(num); });
  return std::make_shared<abstract::AbstractTuple>(elems);
}

static inline bool IsAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  return abs->BuildValue() == kValueAny;
}
IndexHandleLevel TensorIndexGetitem::PreHandleIndex(const AbstractBasePtr &data,
                                                    const abstract::AbstractSlicePtr &abs_slice) {
  ShapeMap shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data->BuildShape());
  auto input_shape = shape_map[kShape];
  data_shape_ = input_shape;
  if (data_shape_.empty()) {
    MS_EXCEPTION(TypeError) << "Cannot iterate over a scalar tensor.";
  }
  if (data_shape_[0] >= 0 && !IsAnyValue(abs_slice->start()) && !IsAnyValue(abs_slice->stop()) &&
      !IsAnyValue(abs_slice->step())) {
    return IndexHandleLevel::kHandleByConstFold;
  }
  MS_LOG(DEBUG) << "The slice index is dynamic.";
  return IndexHandleLevel::kHandleByFunc;
}

AnfNodePtrList TensorIndexGetitem::NormalizeSlice(const AbstractBasePtrList &slice_info_abs,
                                                  const AnfNodePtr &shape_node, const AnfNodePtr &index_node) {
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

  // Handle slice by cpu ops GetitemTensorIndexInfo
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
  const CNodePtr format_slice_node =
    res_graph_->NewCNode({NewValueNode(prim::kPrimNormalizeSlice), shape_node, init_by_none_node, slice_nodes[kStart],
                          slice_nodes[kStop], slice_nodes[kStep]});
  for (size_t i = 0; i < slice_nodes.size(); i++) {
    slice_nodes[i] =
      res_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), format_slice_node, NewValueNode(SizeToLong(i))});
  }
  return slice_nodes;
}

void TensorIndexGetitem::GetItemBySlice(const AnfNodePtr &data_node, const AnfNodePtr &index_node,
                                        const AbstractBasePtr &data, const abstract::AbstractSlicePtr &abs_slice_ptr) {
  IndexHandleLevel index_handle_level = PreHandleIndex(data, abs_slice_ptr);
  auto start_abs = abs_slice_ptr->start();
  auto stop_abs = abs_slice_ptr->stop();
  auto step_abs = abs_slice_ptr->step();
  std::vector<AnfNodePtr> strided_slice;
  if (index_handle_level == IndexHandleLevel::kHandleByConstFold) {
    std::shared_ptr<Slice> slice_ptr = std::make_shared<Slice>(start_abs, stop_abs, step_abs, data_shape_[0]);
    if (slice_ptr->start() == slice_ptr->stop()) {
      ShapeVector empty_slice = data_shape_;
      empty_slice[0] = 0;
      const tensor::TensorPtr slice_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, empty_slice);
      return res_graph_->set_output(NewValueNode(slice_tensor->ToAbstract()->BuildValue()));
    }
    size_t data_dim = data_shape_.size();
    std::vector<int64_t> start_strides(data_dim, 0);
    std::vector<int64_t> stop_strides = data_shape_;
    std::vector<int64_t> step_strides(data_dim, 1);
    start_strides[0] = slice_ptr->start();
    stop_strides[0] = slice_ptr->stop();
    step_strides[0] = slice_ptr->step();
    auto slice_infos = std::vector<AbstractBasePtr>{VectorToTuple(start_strides), VectorToTuple(stop_strides),
                                                    VectorToTuple(step_strides)};
    std::transform(slice_infos.begin(), slice_infos.end(), std::back_inserter(strided_slice),
                   [](const AbstractBasePtr &slice_info) { return NewValueNode(slice_info->BuildValue()); });
  } else {
    AnfNodePtr shape_node;
    if (data_shape_[0] < 0) {
      shape_node = NewCNode({NewValueNode(prim::kPrimShape), data_node}, res_graph_);
    } else {
      shape_node = NewValueNode(data_shape_[0]);
    }
    strided_slice = NormalizeSlice({start_abs, stop_abs, step_abs}, shape_node, index_node);
  }

  ValueNodePtr strided_slice_vnode = NewValueNode(prim::kPrimStridedSlice);
  auto prim = GetValueNode<PrimitivePtr>(strided_slice_vnode);
  const std::vector<std::string> &input_names = {"x", "begin", "end", "strides"};
  const std::vector<std::string> &output_names = {"output"};
  prim->SetAttrs({{ops::kBeginMask, MakeValue(kZeroAnfValue)},
                  {ops::kEndMask, MakeValue(kZeroAnfValue)},
                  {ops::kEllipsisMask, MakeValue(kZeroAnfValue)},
                  {ops::kNewAxisMask, MakeValue(kZeroAnfValue)},
                  {ops::kShrinkAxisMask, MakeValue(kZeroAnfValue)},
                  {kAttrInputNames, MakeValue(input_names)},
                  {kAttrOutputNames, MakeValue(output_names)}});
  (void)strided_slice.insert(strided_slice.begin(), {strided_slice_vnode, data_node});
  res_graph_->set_output(res_graph_->NewCNode(strided_slice));
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
    (void)GetItemBySlice(data_node, index_node, args_abs_list[0], dyn_cast<abstract::AbstractSlice>(args_abs_list[1]));
  }
  return res_graph_;
}
}  // namespace prim
}  // namespace mindspore
