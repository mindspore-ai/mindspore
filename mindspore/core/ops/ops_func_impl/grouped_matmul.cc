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
#include "ops/ops_func_impl/grouped_matmul.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"

namespace mindspore {
namespace ops {
/*
separated means the size of tensorlist not equal 1.
integrated means the size of tensorlist is 1.
split_item        inputs     weight      outputs
      0:      separated     separated    separated
      1:     integrated     b, k, n      separated
      2:      separated     separated    integrated
      3:     integrated     b, k, n      integrated
*/
namespace {
constexpr size_t listInputNum = 7;
constexpr size_t kGmmInputX = 0;
constexpr size_t kGmmInputWeight = 1;
// optional None
constexpr size_t kGmmInputGroupList = 7;
// attr
constexpr size_t kGmmInputSplitItem = 8;
constexpr size_t kGmmInputGroupType = 9;
// TensorShape
constexpr size_t gmmTensor2D = 2;
constexpr size_t gmmTensor3D = 3;
constexpr size_t gmmTensor6D = 6;
// split_item mode
constexpr size_t multiTensor = 0;
constexpr size_t singleTensor = 3;
}  // namespace

int64_t gGroupedMatmulSplitItem = 0;

void GroupedMatmulFuncImpl::CheckSplitItem(const std::string &op_name, const int64_t split_item) const {
  if (split_item != multiTensor && split_item != singleTensor) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the split_item only support 0 or 3, but got " << split_item;
  }
}

void GroupedMatmulFuncImpl::CheckGroupType(const std::string &op_name, const int64_t group_type) const {
  if (group_type != -1 && group_type != 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the group_type only support -1 or 0, but got " << group_type;
  }
}

void GroupedMatmulFuncImpl::CheckSplitItemAndGroupType(const std::string &op_name, const int64_t group_type,
                                                       const int64_t split_item) const {
  if (group_type == -1 && split_item != 0) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', group_type is -1 (not grouped), the split_item only support 0(multi tensor)"
                             << "but split_item got " << split_item;
  }
  if (group_type == 0 && split_item != singleTensor) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', group_type is 0 (group m-axis), the split_item only support 3(one tensor)"
                             << "but split_item got " << split_item;
  }
}

void GroupedMatmulFuncImpl::CheckInputType(const std::vector<AbstractBasePtr> &input_args, const std::string &op_name,
                                           const std::string &input_name, const size_t input_idx,
                                           const std::set<TypePtr> &check_list) const {
  // Optional input args must be TensorList. If optional, it is a TensorList which has only a empty Tensor.
  if (input_args[input_idx]->GetType()->isa<TypeNone>()) {
    MS_EXCEPTION(ShapeError) << "For " << op_name << ", the input {" << input_name
                             << "}, should be TensorList. but got "
                             << input_args[input_idx]->GetType()->isa<TypeNone>();
  }
  // Check Type
  abstract::AbstractTuple optional_list = *(input_args[input_idx]->cast<abstract::AbstractTuplePtr>());
  for (size_t i = 0; i < optional_list.size(); i++) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid(input_name, optional_list[i]->GetType(), check_list, op_name);
  }
}

BaseShapePtr GroupedMatmulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();

  // check split_item
  MS_EXCEPTION_IF_NULL(input_args[kGmmInputSplitItem]);
  auto split_type = input_args[kGmmInputSplitItem]->GetType();
  if (split_type->isa<TypeNone>()) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name << "', the split_item must be a int. Current split_item is None";
  }
  ValuePtr split_ptr = input_args[kGmmInputSplitItem]->GetValue();
  auto split_item = GetValue<int64_t>(split_ptr);
  CheckSplitItem(op_name, split_item);
  gGroupedMatmulSplitItem = split_item;

  // check group_type
  MS_EXCEPTION_IF_NULL(input_args[kGmmInputGroupType]);
  auto group_type_type = input_args[kGmmInputGroupType]->GetType();
  if (group_type_type->isa<TypeNone>()) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name << "', the group_type must be a int. Current group_type is None";
  }
  ValuePtr group_type_ptr = input_args[kGmmInputGroupType]->GetValue();
  auto group_type = GetValue<int64_t>(group_type_ptr);
  CheckGroupType(op_name, group_type);

  CheckSplitItemAndGroupType(op_name, group_type, split_item);

  // x_list
  auto x_ptr = input_args[kGmmInputX]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(x_ptr);
  abstract::AbstractTuple x_list = *x_ptr;

  // weight_list
  auto weight_ptr = input_args[kGmmInputWeight]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(weight_ptr);
  abstract::AbstractTuple weight_list = *weight_ptr;

  // for tensorlist(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < listInputNum; ++i) {
    if (input_args[i]->GetType()->isa<TypeNone>()) {
      dyn_input_sizes.push_back(0);
    } else {
      abstract::AbstractTuple list = *(input_args[i]->cast<abstract::AbstractTuplePtr>());
      dyn_input_sizes.push_back(list.size());
    }
  }
  primitive->set_attr("group_info", MakeValue(dyn_input_sizes));

  // calculate shape. split_item = 3, x[0](m, n) * w[0](e, n, k) = out(m, k)
  if (split_item == singleTensor) {
    if (x_list.size() == 1 && weight_list.size() == 1) {
      std::vector<int64_t> x_shape = x_list[0]->GetShape()->GetShapeVector();
      std::vector<int64_t> w_shape = weight_list[0]->GetShape()->GetShapeVector();
      if (x_shape.size() != gmmTensor2D) {
        MS_EXCEPTION(ShapeError) << "For '" << op_name
                                 << "', when split_item is 3, the x[0] must be 2D Tensor. But x[0] shape :" << x_shape;
      }
      if (w_shape.size() != gmmTensor3D) {
        MS_EXCEPTION(ShapeError) << "For '" << op_name
                                 << "', when split_item is 3, the w[0] must be 3D Tensor. But w[0] shape :" << w_shape;
      }
      if (x_shape[1] != w_shape[1]) {
        MS_EXCEPTION(ShapeError) << "For '" << op_name << "', x[0] shape should be (m, n), w[0] shape show be(e, n, k)."
                                 << "But x[0] shape: " << x_shape << ", w[0] shape: " << w_shape;
      }
      std::vector<BaseShapePtr> outshape_merge = {};
      std::vector<int64_t> res_shape = {x_shape[0], w_shape.back()};
      outshape_merge.emplace_back(std::make_shared<abstract::TensorShape>(res_shape));
      return std::make_shared<abstract::TupleShape>(outshape_merge);
    } else {
      MS_EXCEPTION(ShapeError) << "For '" << op_name << "', when split_item is 3. the size of x or weight only be 1."
                               << "But x size: " << x_list.size() << ", weight size: " << weight_list.size();
    }
  }

  if (x_list.size() != weight_list.size()) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name << "', when split_item is 0, x.size() == w.size()."
                             << "But x size: " << x_list.size() << ", weight size: " << weight_list.size();
  }

  std::vector<BaseShapePtr> outshape_list = {};
  for (size_t i = 0; i < x_list.size(); i++) {
    std::vector<int64_t> x_shape = x_list[i]->GetShape()->GetShapeVector();
    std::vector<int64_t> w_shape = weight_list[i]->GetShape()->GetShapeVector();
    if (x_shape.size() < gmmTensor2D || x_shape.size() > gmmTensor6D) {
      MS_EXCEPTION(ShapeError) << "For '" << op_name << "', when split_item is 0, the tensor in x must be 2-6D. But"
                               << i << "th tensor in x, shape is : " << x_shape;
    }
    if (w_shape.size() != gmmTensor2D) {
      MS_EXCEPTION(ShapeError) << "For '" << op_name << "', when split_item is 0, the tensor in x must be 2-6D. But"
                               << i << "th tensor in x, shape is : " << x_shape;
    }
    if (x_shape.back() != w_shape[0]) {
      MS_EXCEPTION(ShapeError) << "For '" << op_name
                               << "' The back in x[i] shape should be equal to the first in w[i] shape. But x[" << i
                               << "] shape : " << x_shape << ", w[" << i << "] shape : " << w_shape;
    }
    std::vector<int64_t> res_shape = x_shape;
    res_shape.back() = w_shape[1];  // x[a,b,c,m,n] * w[n,k] = out[a,b,c,m,k]
    outshape_list.emplace_back(std::make_shared<abstract::TensorShape>(res_shape));
  }

  return std::make_shared<abstract::TupleShape>(outshape_list);
}

TypePtr GroupedMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  const std::set<TypePtr> xw_input_type = {kFloat16, kBFloat16, kFloat32, kInt8};

  MS_EXCEPTION_IF_NULL(input_args[kGmmInputX]);
  CheckInputType(input_args, op_name, "x", kGmmInputX, xw_input_type);

  MS_EXCEPTION_IF_NULL(input_args[kGmmInputWeight]);
  CheckInputType(input_args, op_name, "weight", kGmmInputWeight, xw_input_type);

  // get split_item and check groups
  MS_EXCEPTION_IF_NULL(input_args[kGmmInputSplitItem]);
  auto split_type = input_args[kGmmInputSplitItem]->GetType();
  if (split_type->isa<TypeNone>()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the group_type must be a int. Current split_item is None";
  }
  ValuePtr split_ptr = input_args[kGmmInputSplitItem]->GetValue();
  auto split_item = GetValue<int64_t>(split_ptr);
  CheckSplitItem(op_name, split_item);
  if (split_item == singleTensor) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("grouplist", input_args[kGmmInputGroupList]->GetType(), {kInt64},
                                                     op_name);
  }

  // check group_list
  MS_EXCEPTION_IF_NULL(input_args[kGmmInputGroupList]);
  auto group_list_type = input_args[kGmmInputGroupList]->GetType();
  if (split_item == singleTensor && group_list_type->isa<TypeNone>()) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name
                             << "', the group_type must be a int when split_item equal 3. Current group_type is None";
  }
  auto group_list_shape_map =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGmmInputGroupList]->GetShape());
  auto group_list_shape = group_list_shape_map[kShape];
  if (split_item == singleTensor && group_list_shape.size() != 1) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name << "', the grouplist must be 1D Tensor when split_item equal 3."
                             << "Current groups_list shape is " << group_list_shape;
  }

  // check group_type
  MS_EXCEPTION_IF_NULL(input_args[kGmmInputGroupType]);
  auto group_type_type = input_args[kGmmInputGroupType]->GetType();
  if (split_item == singleTensor && group_type_type->isa<TypeNone>()) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name
                             << "', the group_type must be a int when split_item equal 3. Current group_type is None";
  }
  ValuePtr group_type_ptr = input_args[kGmmInputGroupType]->GetValue();
  auto group_type = GetValue<int64_t>(group_type_ptr);
  if (split_item == singleTensor && group_type != 0) {
    MS_EXCEPTION(ShapeError) << "For '" << op_name
                             << "', the group_type must be 0(split axis m) when split_item equal 3."
                             << "Current group_type is " << group_type;
  }

  // support split_item 0 or 3
  std::vector<TypePtr> type_tuple;
  abstract::AbstractTuple x_list = *(input_args[kGmmInputX]->cast<abstract::AbstractTuplePtr>());
  for (size_t i = 0; i < x_list.size(); i++) {
    type_tuple.emplace_back(x_list[i]->GetType()->Clone());
  }

  return std::make_shared<Tuple>(std::move(type_tuple));
}

// In compiler get grouplist(not none) for resize
std::set<int64_t> GroupedMatmulFuncImpl::GetValueDependArgIndices() const {
  if (gGroupedMatmulSplitItem == singleTensor) {
    return {kGmmInputGroupList};
  } else {
    return {};
  }
}
}  // namespace ops
}  // namespace mindspore
