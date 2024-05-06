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
#include "grouped_matmul.h"
#include <string>
#include <map>
#include <set>
#include <vector>
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
constexpr size_t listInputNum = 7;
constexpr size_t kInputX = 0;
constexpr size_t kInputWeight = 1;
constexpr size_t kInputBias = 2;
constexpr size_t kInputScale = 3;
constexpr size_t kInputOffset = 4;
constexpr size_t kInputAntiquantScale = 5;
constexpr size_t kInputAntiquantOffset = 6;
// optional None
constexpr size_t kInputGroupList = 7;
// attr
constexpr size_t kInputSplitItem = 8;
constexpr size_t kInputTransposeWeight = 10;

int64_t gGroupedMatmulSplitItem = 0;

void GroupedMatmulFuncImpl::CheckSplitItem(const std::string &op_name, const int64_t split_mode) const {
  if (split_mode != 0 && split_mode != 3) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the split_mode only support 0 or 3, but got " << split_mode;
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

  // split_item
  ValuePtr split_ptr = input_args[kInputSplitItem]->GetValue();
  auto split_mode = GetValue<int64_t>(split_ptr);
  CheckSplitItem(op_name, split_mode);
  gGroupedMatmulSplitItem = split_mode;

  // x_list
  auto x_ptr = input_args[kInputX]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(x_ptr);
  abstract::AbstractTuple x_list = *x_ptr;

  // weight_list
  auto weight_ptr = input_args[kInputWeight]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(weight_ptr);
  abstract::AbstractTuple weight_list = *weight_ptr;

  // transpose_weight
  ValuePtr transpose_weight_ptr = input_args[kInputTransposeWeight]->GetValue();
  bool transpose_weight = GetValue<bool>(transpose_weight_ptr);

  // for tensorlist(input arg) in backend split. (AscendConvertTupleInputToDynamicInput pass)
  std::vector<int64_t> dyn_input_sizes;
  for (size_t i = 0; i < listInputNum; ++i) {
    abstract::AbstractTuple list = *(input_args[i]->cast<abstract::AbstractTuplePtr>());
    MS_LOG(INFO) << "GroupedMatmulFuncImpl " << i << "'s input list size: " << list.size();
    dyn_input_sizes.push_back(list.size());
  }
  primitive->set_attr("group_info", MakeValue(dyn_input_sizes));

  // calculate shape
  std::vector<BaseShapePtr> outshape_list = {};

  int64_t weight_kdim = 0;
  for (size_t i = 0; i < x_list.size(); i++) {
    std::vector<int64_t> x_shape = x_list[i]->GetShape()->GetShapeVector();
    std::vector<int64_t> w_shape = weight_list[i]->GetShape()->GetShapeVector();
    if (split_mode == 0) {
      weight_kdim = (transpose_weight == false) ? w_shape[1] : w_shape[0];
    } else {
      weight_kdim = (transpose_weight == false) ? w_shape[2] : w_shape[1];
    }
    std::vector<int64_t> res_shape = {x_shape[0], weight_kdim};
    outshape_list.emplace_back(std::make_shared<abstract::TensorShape>(res_shape));
  }

  if (split_mode == 3) {
    std::vector<BaseShapePtr> outshape_merge = {};
    int64_t merge_mdim = 0;
    for (auto outptr : outshape_list) {
      merge_mdim += (outptr->GetShapeVector())[0];
    }
    int64_t merge_ndim = (outshape_list[0]->GetShapeVector())[1];
    std::vector<int64_t> merge_shape = {merge_mdim, merge_ndim};
    outshape_merge.emplace_back(std::make_shared<abstract::TensorShape>(merge_shape));
    return std::make_shared<abstract::TupleShape>(outshape_merge);
  }

  return std::make_shared<abstract::TupleShape>(outshape_list);
}

TypePtr GroupedMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  const std::set<TypePtr> fp16_bf16_type = {kFloat16, kBFloat16};
  const std::set<TypePtr> fp16_fp32_type = {kFloat16, kFloat32};

  MS_EXCEPTION_IF_NULL(input_args[kInputX]);
  CheckInputType(input_args, op_name, "x", kInputX, fp16_bf16_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputWeight]);
  CheckInputType(input_args, op_name, "weight", kInputWeight, fp16_bf16_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputBias]);
  CheckInputType(input_args, op_name, "bias(optional)", kInputBias, fp16_fp32_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputScale]);
  CheckInputType(input_args, op_name, "scale(optional)", kInputScale, {kUInt64});

  MS_EXCEPTION_IF_NULL(input_args[kInputOffset]);
  CheckInputType(input_args, op_name, "offset(optional)", kInputOffset, {kFloat32});

  MS_EXCEPTION_IF_NULL(input_args[kInputAntiquantScale]);
  CheckInputType(input_args, op_name, "kInputAntiquantScale(optional)", kInputAntiquantScale, fp16_fp32_type);

  MS_EXCEPTION_IF_NULL(input_args[kInputAntiquantOffset]);
  CheckInputType(input_args, op_name, "kInputAntiquantOffset(optional)", kInputAntiquantOffset, fp16_fp32_type);

  // get split_item and check groups
  ValuePtr split_ptr = input_args[kInputSplitItem]->GetValue();
  auto split_mode = GetValue<int64_t>(split_ptr);
  CheckSplitItem(op_name, split_mode);
  gGroupedMatmulSplitItem = split_mode;
  if (split_mode == 3) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("grouplist", input_args[kInputGroupList]->GetType(), {kInt64},
                                                     op_name);
  }

  // support split_mode 0 or 3
  std::vector<TypePtr> type_tuple;
  abstract::AbstractTuple x_list = *(input_args[kInputX]->cast<abstract::AbstractTuplePtr>());
  for (size_t i = 0; i < x_list.size(); i++) {
    type_tuple.emplace_back(x_list[i]->GetType()->Clone());
  }

  return std::make_shared<Tuple>(std::move(type_tuple));
}

// In compiler get grouplist(not none) for resize
std::set<int64_t> GroupedMatmulFuncImpl::GetValueDependArgIndices() const {
  if (gGroupedMatmulSplitItem == 3) {
    return {kInputGroupList};
  } else {
    return {};
  }
}
}  // namespace ops
}  // namespace mindspore
