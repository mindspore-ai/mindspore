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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/conv1d_padding.h"
#include <string>
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/array_ops.h"
#include "primitive/sequence_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/framework_ops.h"

namespace mindspore::prim {
namespace {
constexpr auto kConv1dPaddingNum3 = 3;
constexpr auto kConv1dPaddingNum2 = 2;
constexpr auto kConv1dPaddingSize1 = 1;
}  // namespace
inline void CheckConv1DPaddingPositiveVector(const string &arg_name, const ArrayValue<int64_t> &array,
                                             const string &prim_name, bool exclude_zeros) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array.IsValueUnknown(i)) {
      continue;
    }
    if (exclude_zeros) {
      if (MS_UNLIKELY(array[i] <= 0)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", '" << arg_name << "' must be positive, but it's "
                                 << array.ToString() << ".";
      }
    } else {
      if (MS_UNLIKELY(array[i] < 0)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", '" << arg_name << "' must be not negetive, but it's "
                                 << array.ToString() << ".";
      }
    }
  }
}

void CheckConv1DShape(const PrimitivePtr &primitive, const ShapeVector &input_shape_param,
                      const ShapeVector &weight_shape, const AbstractBasePtrList &input_args) {
  auto prim_name = primitive->name();
  auto group_opt = GetScalarValue<int64_t>(input_args[kIndex6]->GetValue());
  auto input_shape = input_shape_param;
  if (input_shape.size() == kIndex2) {
    input_shape.insert(input_shape.begin(), 1);
  }
  abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
  abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);
  if (group_opt.has_value()) {
    auto groups = group_opt.value();
    auto in_channels = input_shape[kIndex1];
    (void)CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1);
    (void)CheckAndConvertUtils::CheckInteger("out_channels", weight_shape[0], kGreaterEqual, groups);
    (void)CheckAndConvertUtils::CheckInteger("out_channels mod groups", weight_shape[0] % groups, kEqual, 0);
    if (in_channels / groups != weight_shape[kIndex1]) {
      MS_EXCEPTION(ValueError) << "The argument error. in_channels/groups must be equal weight[1], "
                               << "but in_channels/groups is " << in_channels / groups << ", and weight[1] is "
                               << weight_shape[kIndex1];
    }
  }
}

void CheckConv1DPaddingInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  auto prim_name = primitive->name();
  const auto &input_shape_ = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto input_shape = input_shape_;
  const auto &weight_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex3]->GetValue());
  auto padding_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());
  auto dilation_opt = GetArrayValue<int64_t>(input_args[kIndex5]->GetValue());
  const auto input_rank = input_shape.size();
  const auto weight_rank = weight_shape.size();
  if (!IsDynamicRank(input_shape) && input_rank != kIndex2 && input_rank != kIndex3) {
    MS_EXCEPTION(ValueError) << "Conv1d expects input to have a 2 or 3 dimensions, but got input with " << input_rank
                             << " dimension(s).";
  }
  if (!IsDynamicRank(weight_shape) && weight_rank != kIndex3) {
    MS_EXCEPTION(ValueError) << "Conv1d expects weight to have a 3 dimensions, but got weight with " << weight_rank
                             << " dimension(s).";
  }
  if (padding_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    if (padding_enum != PadMode::VALID && padding_enum != PadMode::SAME) {
      MS_EXCEPTION(ValueError) << "For primitive[Conv1DPadding], input padding string must be one of {'same', 'valid'}";
    }
  }
  if (padding_opt.has_value() && stride_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    const auto &stride = stride_opt.value();
    if (padding_enum == PadMode::SAME) {
      for (int i = 0; i < SizeToLong(stride.size()); i++) {
        if (!stride.IsValueUnknown(i) && stride[i] != 1) {
          MS_EXCEPTION(ValueError) << "For primitive[Conv1DPadding], stride must be 1 when padding is same.";
        }
      }
    }
  }
  if (stride_opt.has_value()) {
    const auto &stride = stride_opt.value();
    CheckConv1DPaddingPositiveVector("stride", stride, prim_name, true);
  }
  if (!dilation_opt.has_value()) {
    return;
  }
  auto dilation = dilation_opt.value();
  CheckConv1DPaddingPositiveVector("dilation", dilation, prim_name, true);

  if (!IsDynamic(input_shape) && !IsDynamic(weight_shape)) {
    CheckConv1DShape(primitive, input_shape, weight_shape, input_args);
  }
}

NodePtr Conv1DPaddingMetaImpl::ProcessDims(const NodePtr &input) {
  auto input_rank = Rank(input);
  auto is_3d = Equal(input_rank, Value(kConv1dPaddingNum3));

  auto process_2d_input_branch = [&]() { Return(Call(Prim(ExpandDims), input, Value(0))); };
  auto input_branch = [&]() { Return(input); };
  auto processed_input = If(is_3d, input_branch, process_2d_input_branch);
  auto needs_squeeze = Not(is_3d);
  return Tuple(processed_input, needs_squeeze);
}

NodePtr Conv1DPaddingMetaImpl::CalcPadding(const NodePtr &in_shape, const NodePtr &w_shape, const NodePtr &stride,
                                           const NodePtr &dilation) {
  auto padding_l = List();
  auto padding_r = List();
  auto is_symmetric_list = List();
  auto res_list = List(padding_l, padding_r, is_symmetric_list);
  auto calc_dim_padding = [&](const NodePtr &index, const NodePtr &i, const NodePtr &res_out) {
    auto in_size = GetItem(in_shape, i);
    auto kernel_size = GetItem(w_shape, i);
    auto stride_value = GetItem(stride, Value(0));
    auto dilation_value = GetItem(dilation, Value(0));

    auto total_pad = ScalarMul(dilation_value, ScalarSub(kernel_size, Value(1)));
    auto total_pad_need_calc_branch = [&]() {
      auto true_branch = [&]() {
        auto new_total_pad = ScalarSub(total_pad, Value(1));
        Return(new_total_pad);
      };
      auto false_branch = [&]() { Return(total_pad); };
      auto wiggle_room = ScalarSub(ScalarMod(in_size, stride_value), Value(1));
      auto expand_condition = Greater(wiggle_room, Value(0));
      auto res = If(expand_condition, true_branch, false_branch);
      Return(res);
    };
    auto total_pad_no_need_calc_branch = [&]() { Return(total_pad); };
    auto total_pad_need_calc_condition = And(Greater(stride_value, Value(kConv1dPaddingNum2)),
                                             Equal(ScalarMod(total_pad, Value(kConv1dPaddingNum2)), Value(1)));
    auto total_pad_new = If(total_pad_need_calc_condition, total_pad_need_calc_branch, total_pad_no_need_calc_branch);

    auto left = ScalarFloorDiv(total_pad_new, Value(kConv1dPaddingNum2));
    auto right = ScalarSub(total_pad_new, left);
    auto left_equal_right = Equal(left, right);

    auto true_branch = [&]() { Return(Value(1)); };
    auto false_branch = [&]() { Return(Value(0)); };
    auto left_equal_right_value = If(left_equal_right, true_branch, false_branch);
    auto padding_l = GetItem(res_out, Value(0));
    auto padding_r = GetItem(res_out, Value(1));
    auto is_symmetric_list = GetItem(res_out, Value(kConv1dPaddingNum2));
    padding_l = Call(Prim(ListAppend), padding_l, left);
    padding_r = Call(Prim(ListAppend), padding_r, right);
    is_symmetric_list = Call(Prim(ListAppend), is_symmetric_list, left_equal_right_value);
    Return(List(padding_l, padding_r, is_symmetric_list));
  };

  res_list = For(calc_dim_padding, Tuple(Value(kConv1dPaddingNum2)), res_list);
  padding_l = GetItem(res_list, Value(0));
  padding_r = GetItem(res_list, Value(1));
  is_symmetric_list = GetItem(res_list, Value(kConv1dPaddingNum2));
  auto is_symmetric = GetItem(is_symmetric_list, Value(0));

  auto true_branch = [&]() { Return(Value(1)); };
  auto false_branch = [&]() { Return(Value(0)); };
  auto is_symmetric_value = If(is_symmetric, true_branch, false_branch);

  auto out = Tuple(padding_l, padding_r, is_symmetric_value);
  return out;
}

BeginFunction(Conv1DPadding, input, weight, bias, stride, padding, dilation, groups) {
  // illegal padding
  auto invalid_padding = [&]() { Return(Raise("ValueError", "Padding must be either 'SAME' or 'VALID'")); };
  // main compute
  auto compute_conv = [&]() {
    // process dimension
    auto processed_input_needs_squeeze_is_batch_match = ProcessDims(input);
    auto processed_input = GetItem(processed_input_needs_squeeze_is_batch_match, Value(0));
    auto needs_squeeze = GetItem(processed_input_needs_squeeze_is_batch_match, Value(1));
    // expand parameters
    auto weight_shape = Shape(weight);
    // padding is same
    auto same_mode_branch = [&]() {
      auto pad_l_pad_r_is_sym = CalcPadding(Shape(processed_input), weight_shape, stride, dilation);
      auto pad_l = GetItem(pad_l_pad_r_is_sym, Value(0));
      auto pad_r = GetItem(pad_l_pad_r_is_sym, Value(1));
      auto is_sym = GetItem(pad_l_pad_r_is_sym, Value(kConv1dPaddingNum2));
      auto is_sym_branch = [&]() {
        auto pad_nd = Tuple(GetItem(pad_l, Value(0)), GetItem(pad_r, Value(0)));
        Return(Tuple(pad_l, pad_nd, Value(1)));
      };
      auto not_sym_branch = [&]() {
        auto pad_nd = List(Value(0), Value(0));
        auto calc_pad_nd = [&](const NodePtr &index, const NodePtr &item, const NodePtr &res) {
          auto delta_pad = ScalarSub(GetItem(pad_r, Value(0)), GetItem(pad_l, Value(0)));
          auto true_branch = [&]() { Return(Value(1)); };
          auto false_branch = [&]() { Return(Value(0)); };
          auto real_pad_idx = If(Greater(delta_pad, Value(0)), true_branch, false_branch);
          auto out = SetItem(res, real_pad_idx, delta_pad);
          Return(out);
        };
        pad_nd = For(calc_pad_nd, Tuple(Value(0)), pad_nd);
        Return(Tuple(pad_l, ListToTuple(pad_nd), Value(0)));
      };
      Return(If(Equal(is_sym, Value(1)), is_sym_branch, not_sym_branch));
    };
    // padding is VALID
    auto valid_mode_branch = [&]() { Return(Tuple(Tuple(Value(0)), Tuple(Value(0), Value(0)), Value(1))); };
    auto pad_params_and_is_symmetric = If(Equal(padding, Value(1)), same_mode_branch, valid_mode_branch);
    auto pad_params = GetItem(pad_params_and_is_symmetric, Value(0));
    auto pad_nd = GetItem(pad_params_and_is_symmetric, Value(1));
    auto is_symmetric = GetItem(pad_params_and_is_symmetric, Value(kConv1dPaddingNum2));
    // call convolution
    auto is_symmetric_branch = [&]() {
      auto output = Call(Prim(Convolution), processed_input, weight, bias, stride, pad_params, dilation, Value(false),
                         Tuple(Value(0)), groups);
      Return(output);
    };
    auto not_symmetric_branch = [&]() {
      auto input_padded = Call(Prim(ConstantPadND), processed_input, pad_nd, Value(0));
      auto output = Call(Prim(Convolution), input_padded, weight, bias, stride, pad_params, dilation, Value(false),
                         Tuple(Value(0)), groups);
      Return(output);
    };
    auto conv_result = If(Equal(is_symmetric, Value(1)), is_symmetric_branch, not_symmetric_branch);
    // process dimension
    auto is_needs_squeeze = [&]() { Return(Call(Prim(Squeeze), conv_result, Value(0))); };
    auto not_needs_squeeze = [&]() { Return(conv_result); };
    Return(If(needs_squeeze, is_needs_squeeze, not_needs_squeeze));
  };

  Return(If(Or(Equal(padding, Value(1)),                    // same
               Equal(padding, Value(kConv1dPaddingNum2))),  // valid
            compute_conv, invalid_padding));
}
EndFunction(Conv1DPadding)
}  // namespace mindspore::prim
