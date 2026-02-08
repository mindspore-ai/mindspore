/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <memory>
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"
#include "ir/value.h"
#include "infer/conv2d.h"
#include "primitive/conv_pool_op_name.h"
#include "primitive/nn_op_name.h"
#include "primitive/nn_optimizer_op_name.h"
#include "ops_utils/op_utils.h"
#include "include/op_enum.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "mindapi/base/types.h"

namespace mindspore::expander::bprop {
namespace {
const int kConstNumberTwo = 2;
}  // namespace

NodePtrList Dropout2DBpropExpander(BpropBuilder *ib) {
  auto keep_prob = GetValue<float>(ib->GetAttr("keep_prob"));
  auto x = ib->GetInput(i0);
  auto out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto mask = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  mask = ib->Cast(mask, kFloat32);
  if (keep_prob != 0) {
    dy = ib->Muls(dy, ib->ValueByType(1.0 / keep_prob, ib->GetDtype(dy)));
  }
  dy = ib->Mul(mask, dy);
  dy = ib->Cast(dy, ib->GetDtype(x));
  return {dy};
}

NodePtrList GeLUBpropExpander(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto dx = ib->GeLUGrad(dout, x, out);
  return {dx};
}

NodePtrList FastGeLUBpropExpander(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto dout = ib->GetInput(i2);
  auto dx = ib->Emit("FastGeLUGrad", {dout, x});
  return {dx};
}

NodePtr MeanExtGrad(BpropBuilder *ib, const NodePtr &input, const NodePtr &out, const NodePtr &dout) {
  auto input_dtype_id = ib->GetDtypeId(input);
  if (input_dtype_id == kNumberTypeComplex64 || input_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'MeanExt', gradient not support for complex type currently.";
  }

  auto grad = SumGrad(ib, input, ib->Value<std::vector<int64_t>>({-1, -2, -3}), dout, true);
  grad = ib->Cast(grad, ib->GetDtype(input));

  NodePtr div_shape_node;
  NodePtr dx;
  if (IsDynamic(ib->GetShape(input)) || IsDynamic(ib->GetShape(out))) {
    auto shape_out_sz = ib->DynSize(out, kFloat32);
    auto true_branch = [&](Emitter *e) -> NodePtrList { return {ib->Tensor(1, kFloat32)}; };
    auto false_branch = [&](Emitter *e) -> NodePtrList { return {ib->DynSize(input, kFloat32) / shape_out_sz}; };
    auto is_zero_out_sz = ib->Equal(shape_out_sz, ib->Tensor(0, kFloat32));
    auto div_shape = ib->Conditional(is_zero_out_sz, true_branch, false_branch);
    div_shape_node = ib->Cast(div_shape, ib->GetDtype(grad));
    dx = ib->Div(grad, div_shape_node);
  } else {
    auto shape_out_sz = ib->GetSize(out);
    auto div_shape = shape_out_sz == 0 ? 1 : ib->GetSize(input) / shape_out_sz;
    div_shape_node = ib->ValueByType(div_shape, ib->GetDtype(grad));
    dx = ib->Divs(grad, div_shape_node);
  }
  return dx;
}

DAttr Conv2DAttrs(BpropBuilder *ib) {
  DAttr attrs{{"pad_mode", ib->GetAttr("pad_mode")},
              {"pad", ib->GetAttr("pad")},
              {"dilation", ib->GetAttr("dilation")},
              {"stride", ib->GetAttr("stride")},
              {"group", ib->GetAttr("group")},
              {"groups", ib->GetAttr("group")},
              {"format", ib->GetAttr("format")},
              {"data_format", ib->GetAttr("format")},
              {"out_channel", ib->GetAttr("out_channel")},
              {"kernel_size", ib->GetAttr("kernel_size")},
              {"mode", MakeValue(1)}};
  return attrs;
}

DAttr Conv2DBackpropAttrs(BpropBuilder *ib) {
  DAttr attrs{{"mode", ib->GetAttr("mode")},
              {"dilation", ib->GetAttr("dilation")},
              {"stride", ib->GetAttr("stride")},
              {"group", ib->GetAttr("group")},
              {"groups", ib->GetAttr("group")},
              {"format", ib->GetAttr("format")},
              {"data_format", ib->GetAttr("format")},
              {"out_channel", ib->GetAttr("out_channel")},
              {"kernel_size", ib->GetAttr("kernel_size")},
              {"pad_mode", ib->GetAttr("pad_mode")},
              {"pad", ib->GetAttr("pad")},
              {"pad_list", ib->GetAttr("pad_list")}};
  return attrs;
}

NodePtrList Conv2DTransposeBpropExpander(BpropBuilder *ib) {
  auto x = ib->GetInput(i0);
  auto w = ib->GetInput(i1);
  auto f_sizes = ib->GetInput(i2);
  auto dout = ib->GetInput(i4);
  auto w_shape = ib->Shape(w);
  auto dx = x->need_compute_grad_out() ? ib->Emit(kConv2DOpName, {dout, w}, Conv2DAttrs(ib)) : ib->OutZeros(x);
  auto dw = w->need_compute_grad_out()
              ? ib->Emit(kConv2DBackpropFilterOpName, {x, dout, w_shape}, Conv2DBackpropAttrs(ib))
              : ib->OutZeros(w);
  return {dx, dw, ib->OutZeros(f_sizes)};
}

class BiasAddGradShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_BiasAddGrad", BiasAddGradShapeCalc)
  explicit BiasAddGradShapeCalc(int64_t format) : ShapeCalcFunctor("ShapeCalc_BiasAddGrad"), format_(format) {}
  ValuePtr ToValue() const override { return MakeValue(format_); }
  void FromValue(const ValuePtr &value) override { format_ = GetValue<int64_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    ShapeVector expanded_shape;
    ShapeVector tile_mults;
    ShapeVector one_vec{1};
    auto dy_shape = inputs.at(0);
    auto dout_shape = inputs.at(1);
    if (format_ == Format::NCHW) {
      // expanded_shape = np.concatenate([np.ones_like(shape[:1]), bias_shape, np.ones_like(shape[2:])], axis=0)
      expanded_shape = one_vec + dout_shape;
      expanded_shape = dy_shape.size() > i2 ? expanded_shape + ShapeVector(dy_shape.size() - i2, 1) : expanded_shape;
      // tile_mults = np.concatenate([shape[:1], [1], shape[2:]], axis=0)
      ShapeVector tmp{dy_shape[0], 1};
      tile_mults = tmp;
      tile_mults = dy_shape.size() > i2 ? tile_mults + ShapeVector(dy_shape.begin() + i2, dy_shape.end()) : tile_mults;
    } else {
      // expanded_shape = np.concatenate([np.ones_like(shape[:-1]), bias_shape], axis=0)
      expanded_shape = ShapeVector(1, dy_shape.size() - 1) + dout_shape;
      // tile_mults = np.concatenate([shape[:-1], [1]], axis=0)
      tile_mults = ShapeVector(dy_shape.begin(), dy_shape.end() - 1) + one_vec;
    }
    return {expanded_shape, tile_mults};
  }

  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t x_rank = IsDynamicRank(inputs.at(0)) ? -1 : SizeToLong(inputs.at(0).size() + inputs.at(1).size() - 1);
    int64_t y_rank = IsDynamicRank(inputs.at(1)) ? -1 : SizeToLong(inputs.at(0).size());
    return {x_rank, y_rank};
  }

 protected:
  int64_t format_;
};
REG_FUNCTOR("ShapeCalc_BiasAddGrad", BiasAddGradShapeCalc);

class ExtractImagePatchesShapeCalc : public ShapeCalcFunctor {
 public:
  DECLARE_SHAPE_CALC("ShapeCalc_ExtractImagePatches", ExtractImagePatchesShapeCalc)
  ExtractImagePatchesShapeCalc(int64_t ksizes_row, int64_t ksizes_col)
      : ShapeCalcFunctor("ShapeCalc_ExtractImagePatches"), ksizes_row_(ksizes_row), ksizes_col_(ksizes_col) {}
  ValuePtr ToValue() const override {
    auto values = {MakeValue(ksizes_row_), MakeValue(ksizes_col_)};
    return std::make_shared<ValueTuple>(values);
  }
  void FromValue(const ValuePtr &value) override {
    auto values = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(values);
    if (values->value().size() != i2) {
      MS_LOG(EXCEPTION) << "CalBatchGatherShapeCalc's value size should be 2, but got " << values->value().size();
    }
    ksizes_row_ = GetValue<int64_t>(values->value()[0]);
    ksizes_col_ = GetValue<int64_t>(values->value()[1]);
  }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs.at(0);
    auto x_batch = x_shape[0];
    auto x_depth = x_shape[1];
    auto x_row = x_shape[2];
    auto x_col = x_shape[3];
    auto x_indices_num = (x_row * x_col) + 1;
    auto out_shape = inputs.at(1);
    auto out_row = out_shape[2];
    auto out_col = out_shape[3];
    auto out_indices_num = ((out_row * out_col) * ksizes_row_) * ksizes_col_;
    return {{x_indices_num},
            {1, 1, x_row, x_col},
            {out_indices_num},
            {1, out_row, out_col, ksizes_row_ * ksizes_col_},
            {x_indices_num, out_indices_num},
            {x_indices_num - 1, out_indices_num},
            {x_batch, out_row, out_col, ksizes_row_, ksizes_col_, x_depth},
            {-1, x_batch * x_depth},
            {x_row, x_col, x_batch, x_depth}};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    return {1, 4, 1, 4, 2, 2, 6, 2, 4};
  }

 protected:
  int64_t ksizes_row_{0};
  int64_t ksizes_col_{0};
};
REG_FUNCTOR("ShapeCalc_ExtractImagePatches", ExtractImagePatchesShapeCalc);

// implements the function to free useless values which defined in other files.
void FreeTensorsOfMul(const PynativeCallback &cb);

void FreeTensorsOfLSTM(const PynativeCallback &cb) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device == "CPU") {
    cb.FreeOutputDeviceAddress({i4});
    MS_LOG(DEBUG) << "Clear CPU device address for outputs[4] of " << cb.opname();
  } else if (device == "GPU") {
    cb.FreeOutputDeviceAddress({i1, i2});
    MS_LOG(DEBUG) << "Clear GPU device address for outputs[1] and outputs[2] of " << cb.opname();
  }
}

void FreeTensorsOfThresholdGrad(const PynativeCallback &cb) {
  cb.FreeOutputDeviceAddress();
  auto &inputs = *cb.GetInputs();
  cb.FreeDeviceAddress(&inputs[i0]);
  cb.FreeDeviceAddress(&inputs[i3]);
  cb.FreeOutputDeviceAddress();
  if (cb.IsNotRequiresGrad(i0)) {
    cb.FreeDeviceAddress(&inputs[i1]);
    cb.FreeDeviceAddress(&inputs[i4]);
    MS_LOG(DEBUG) << "Clear device address for inputs[1] and inputs[4] of " << cb.opname();
  }
}

REG_BPROP_BUILDERS_BEGIN(GradNnOps)
REG_BPROP_BUILDER("Conv2D").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_shape = ib->Shape(x);
  auto w_shape = ib->Shape(w);
  auto format = GetValue<std::string>(ib->GetAttr("format"));
  auto dilation = GetValue<ShapeVector>(ib->GetAttr("dilation"));
  auto stride = GetValue<ShapeVector>(ib->GetAttr("stride"));
  auto pad_list = ib->GetAttr("pad_list");
  if (pad_list == nullptr) {
    auto prim = std::make_shared<Primitive>("Conv2D", ib->GetAttrs());
    (void)ops::Conv2dInfer(nullptr, prim, {x->abstract(), w->abstract()});
    pad_list = prim->GetAttr("pad_list");
  }
  auto dx = x->need_compute_grad_out()
              ? ib->Emit(kConv2DBackpropInputOpName, {dout, w, x_shape},
                         {{"mode", ib->GetAttr("mode")},
                          {"dilation", MakeValue(format == "NHWC" ? ConvToNHWC(dilation) : dilation)},
                          {"stride", MakeValue(format == "NHWC" ? ConvToNHWC(stride) : stride)},
                          {"group", ib->GetAttr("group")},
                          {"groups", ib->GetAttr("group")},
                          {"format", ib->GetAttr("format")},
                          {"data_format", ib->GetAttr("format")},
                          {"out_channel", ib->GetAttr("out_channel")},
                          {"kernel_size", ib->GetAttr("kernel_size")},
                          {"pad_mode", ib->GetAttr("pad_mode")},
                          {"pad", ib->GetAttr("pad")},
                          {"pad_list", pad_list}})
              : ib->OutZeros(x);
  auto dw = w->need_compute_grad_out() ? ib->Emit("Conv2DBackpropFilter", {dout, x, w_shape},
                                                  {{"mode", ib->GetAttr("mode")},
                                                   {"dilation", MakeValue(dilation)},
                                                   {"stride", MakeValue(stride)},
                                                   {"group", ib->GetAttr("group")},
                                                   {"groups", ib->GetAttr("group")},
                                                   {"format", ib->GetAttr("format")},
                                                   {"data_format", ib->GetAttr("format")},
                                                   {"out_channel", ib->GetAttr("out_channel")},
                                                   {"kernel_size", ib->GetAttr("kernel_size")},
                                                   {"pad_mode", ib->GetAttr("pad_mode")},
                                                   {"pad", ib->GetAttr("pad")},
                                                   {"pad_list", pad_list}})
                                       : ib->OutZeros(w);
  return {dx, dw};
});

DEF_PURE_SHAPE_CALC(g_conv_transpose2d)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(i0);
    if (input_shape.size() != i4) {
      MS_ASSERT(input_shape.size() == i3);
      input_shape.insert(input_shape.begin(), 1);
    }
    return {std::move(input_shape)};
  })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> std::vector<int64_t> { return {4}; });

REG_BPROP_BUILDER("ConvTranspose2D").SetUnusedInputs({i8}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &weight = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride = ib->GetInput(i3);
  const auto &padding = ib->GetInput(i4);
  const auto &output_padding = ib->GetInput(i5);
  const auto &groups = ib->GetInput(i6);
  const auto &dilation = ib->GetInput(i7);
  const auto &dout = ib->GetInput(i9);

  auto transposed = ib->Value<bool>(true);
  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {input->need_compute_grad_out(), weight->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->Value(output_mask_vec);

  // batchify input and dout
  auto new_input_shape = ib->ShapeCalc(g_conv_transpose2d, {input})[0];
  auto batched_input = ib->Reshape(input, new_input_shape);
  auto new_dout_shape = ib->ShapeCalc(g_conv_transpose2d, {dout})[0];
  auto batched_dout = ib->Reshape(dout, new_dout_shape);
  auto gradients = ib->ConvolutionGrad(batched_dout, batched_input, weight, bias, stride, padding, dilation, transposed,
                                       output_padding, groups, output_mask);
  // inverse batchify
  auto dinput = ib->Reshape(ib->TupleGetItem(gradients, i0), ib->Shape(input));
  auto dweight = ib->TupleGetItem(gradients, i1);
  auto dbias = ib->TupleGetItem(gradients, i2);
  return {dinput,
          dweight,
          dbias,
          ib->OutZeros(stride),
          ib->OutZeros(padding),
          ib->OutZeros(output_padding),
          ib->OutZeros(groups),
          ib->OutZeros(dilation)};
});

DEF_PURE_SHAPE_CALC(g_conv2d_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_sizes = inputs.at(i0);
    auto weight_sizes = inputs.at(i1);
    auto batchfy = (input_sizes.size() == weight_sizes.size());
    auto _batchfy = batchfy ? 1 : 0;
    return {{_batchfy}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

REG_BPROP_BUILDER("Conv2DExt").SetUnusedInputs({i7}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  const auto &group_value = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto transposed_value = ib->EmitValue(MakeValue<bool>(false));
  std::vector<int64_t> output_padding_vec = {0, 0};
  auto output_padding_value = ib->EmitValue(MakeValue(output_padding_vec));

  NodePtr nx, ndout, ndx;
  NodePtrList ret_batchfy = ib->ShapeCalc(g_conv2d_ext_shapecalc, {x, w});
  auto &batchfy = ret_batchfy[i0];
  auto batchfy_conditional = ib->Equal(ib->TupleGetItem(batchfy, i0), ib->Value<int64_t>(1));
  auto cond_out_batchfy = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {x, dout}; },
    [&](Emitter *e) -> NodePtrList { return {e->ExpandDims(x, i0), e->ExpandDims(dout, i0)}; });
  nx = ib->TupleGetItem(cond_out_batchfy, i0);
  ndout = ib->TupleGetItem(cond_out_batchfy, i1);

  auto conv2d_grad_out = ib->ConvolutionGrad(ndout, nx, w, bias, stride_value, pad_value, dilation_value,
                                             transposed_value, output_padding_value, group_value, output_mask);
  auto dx = ib->TupleGetItem(conv2d_grad_out, i0);
  ndx = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {dx}; },
    [&](Emitter *e) -> NodePtrList { return {e->Squeeze(dx, MakeValue(ShapeVector{0}))}; });
  auto dw = ib->TupleGetItem(conv2d_grad_out, i1);
  auto dbias = ib->TupleGetItem(conv2d_grad_out, i2);
  return {ndx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(group_value)};
});

DEF_PURE_SHAPE_CALC(g_conv2d_padding_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_sizes = inputs.at(i0);
    auto weight_sizes = inputs.at(i1);
    auto stride_values = inputs.at(i2);
    auto dilation_values = inputs.at(i3);
    auto batchfy = (input_sizes.size() == weight_sizes.size());
    auto batchfy_value = 1;
    if (!batchfy) {
      input_sizes.insert(input_sizes.begin(), 1);
      batchfy_value = 0;
    }
    auto dim = input_sizes.size() - 2;

    std::vector<int64_t> padding_l;
    std::vector<int64_t> padding_r;
    auto symmetric_padding_true = 1;
    auto symmetric_padding_false = 0;
    auto symmetric_padding = symmetric_padding_true;

    for (size_t i = 0; i < dim; ++i) {
      auto i_stride = i % stride_values.size();
      auto i_dilation = i % dilation_values.size();
      auto stride = stride_values[i_stride];
      auto dilation = dilation_values[i_dilation];
      auto inputSize = input_sizes[i + 2];
      auto kernelSize = weight_sizes[i + 2];
      auto total_padding = dilation * (kernelSize - 1);
      if (stride > 2 && (total_padding % 2 == 1)) {
        auto wiggle_room = inputSize % stride - 1;
        if (wiggle_room > 0) {
          --total_padding;
        }
      }
      auto left = total_padding / 2;
      auto right = total_padding - left;

      padding_l.push_back(left);
      padding_r.push_back(right);
      if (left != right) {
        symmetric_padding = symmetric_padding_false;
      }
    }

    if (symmetric_padding) {
      std::vector<int64_t> pad_nd_temp(2 * dim, 0);
      return {{symmetric_padding}, pad_nd_temp, padding_l, pad_nd_temp, {batchfy_value}};
    } else {
      std::vector<int64_t> pad_nd(2 * dim, 0);
      std::vector<int64_t> padding_neg_pad(2 * dim, 0);
      for (size_t i = 0; i < dim; ++i) {
        auto delta_pad = padding_r[i] - padding_l[i];
        auto pad_idx = 2 * (dim - 1 - i);  // F.pad goes from last dim to first
        if (delta_pad > 0) {
          pad_nd[pad_idx + 1] = delta_pad;
          padding_neg_pad[pad_idx + 1] = -delta_pad;
        } else {
          pad_nd[pad_idx] = delta_pad;
          padding_l[i] = padding_r[i];
          padding_neg_pad[pad_idx] = -delta_pad;
        }
      }
      return {{symmetric_padding}, pad_nd, padding_l, padding_neg_pad, {batchfy_value}};
    }
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1, 4, 2, 4, 1}; });

REG_BPROP_BUILDER("Conv2DPadding").SetUnusedInputs({i7}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  auto transposed_value = ib->Value<bool>(false);
  std::vector<int64_t> output_padding = {0, 0};
  auto output_padding_value = ib->EmitValue(MakeValue(output_padding));
  const auto &group_value = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto pad_values = pad_value->BuildValue();
  auto pad_int_value = GetValue<int64_t>(pad_values);

  NodePtr conv2d_grad_out;
  NodePtr dx;
  NodePtrList ret_shape = ib->ShapeCalc(g_conv2d_padding_shapecalc, {x, w, stride_value, dilation_value}, {2, 3});
  const auto &batchfy = ret_shape[i4];
  auto batchfy_conditional = ib->Equal(ib->TupleGetItem(batchfy, 0), ib->Value<int64_t>(1));

  auto conv_out_batchfy = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {x, dout}; },
    [&](Emitter *e) -> NodePtrList { return {e->ExpandDims(x, 0), e->ExpandDims(dout, 0)}; });

  auto batchfy_x = ib->TupleGetItem(conv_out_batchfy, 0);
  auto batchfy_dout = ib->TupleGetItem(conv_out_batchfy, 1);
  if (pad_int_value == PadMode::SAME) {
    const auto &symmetric_padding = ret_shape[i0];
    const auto &pad_nd = ret_shape[i1];
    const auto &padding_l = ret_shape[i2];
    const auto &padding_neg_pad = ret_shape[i3];

    // get conv2d_grad_out
    auto conv2d_grad_out_true = [&](Emitter *e) -> NodePtrList {
      return {e->ConvolutionGrad(batchfy_dout, batchfy_x, w, bias, stride_value, padding_l, dilation_value,
                                 transposed_value, output_padding_value, group_value, output_mask)};
    };
    auto conv2d_grad_out_false = [&](Emitter *e) -> NodePtrList {
      auto zero = e->EmitValue(MakeValue<int64_t>(0));
      auto x_new = e->ConstantPadND(batchfy_x, pad_nd, zero);
      return {e->ConvolutionGrad(batchfy_dout, x_new, w, bias, stride_value, padding_l, dilation_value,
                                 transposed_value, output_padding_value, group_value, output_mask)};
    };
    auto symmetric_padding_conditional = ib->Equal(ib->TupleGetItem(symmetric_padding, 0), ib->Value<int64_t>(1));
    conv2d_grad_out = ib->Conditional(symmetric_padding_conditional, conv2d_grad_out_true, conv2d_grad_out_false);

    // get dx
    auto dx_true = [&](Emitter *e) -> NodePtrList { return {e->TupleGetItem(conv2d_grad_out, 0)}; };

    auto dx_false = [&](Emitter *e) -> NodePtrList {
      auto zero = e->EmitValue(MakeValue<int64_t>(0));
      return {e->ConstantPadND(e->TupleGetItem(conv2d_grad_out, 0), padding_neg_pad, zero)};
    };
    dx = ib->Conditional(symmetric_padding_conditional, dx_true, dx_false);
  } else if (pad_int_value == PadMode::VALID) {
    std::vector<int64_t> pad_vector = {0, 0};
    conv2d_grad_out =
      ib->ConvolutionGrad(batchfy_dout, batchfy_x, w, bias, stride_value, ib->EmitValue(MakeValue(pad_vector)),
                          dilation_value, transposed_value, output_padding_value, group_value, output_mask);
    dx = ib->TupleGetItem(conv2d_grad_out, 0);
  } else {
    MS_LOG(EXCEPTION) << "For Conv2d, input padding string must be one of {'same', 'valid'}";
  }

  auto batchfy_true = [&](Emitter *e) -> NodePtrList { return {dx}; };
  auto batchfy_false = [&](Emitter *e) -> NodePtrList { return {e->Squeeze(dx, MakeValue(ShapeVector{0}))}; };
  auto squeeze_dx = ib->Conditional(batchfy_conditional, batchfy_true, batchfy_false);

  auto dw = ib->TupleGetItem(conv2d_grad_out, 1);
  auto dbias = ib->TupleGetItem(conv2d_grad_out, 2);
  return {squeeze_dx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(group_value)};
});

REG_BPROP_BUILDER("Convolution").SetUnusedInputs({i9}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  const auto &transposed_value = ib->GetInput(i6);
  const auto &output_padding_value = ib->GetInput(i7);
  const auto &group_value = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));

  auto conv2d_grad_out = ib->ConvolutionGrad(ib->GetInput(i10), x, w, bias, stride_value, pad_value, dilation_value,
                                             transposed_value, output_padding_value, group_value, output_mask);
  auto dx = ib->TupleGetItem(conv2d_grad_out, i0);
  auto dw = ib->TupleGetItem(conv2d_grad_out, i1);
  auto dbias = ib->TupleGetItem(conv2d_grad_out, i2);
  return {dx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(transposed_value),
          ib->OutZeros(output_padding_value),
          ib->OutZeros(group_value)};
});

DEF_PURE_SHAPE_CALC(g_conv3d_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_sizes = inputs.at(i0);
    auto weight_sizes = inputs.at(i1);
    auto batchfy = (input_sizes.size() == weight_sizes.size());
    auto _batchfy = batchfy ? 1 : 0;
    return {{_batchfy}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

REG_BPROP_BUILDER("Conv3DExt").SetUnusedInputs({i7}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  const auto &group_value = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto transposed_value = ib->EmitValue(MakeValue<bool>(false));
  std::vector<int64_t> output_padding_vec = {0, 0, 0};
  auto output_padding_value = ib->EmitValue(MakeValue(output_padding_vec));

  NodePtr nx, ndout, ndx;
  NodePtrList ret_batchfy = ib->ShapeCalc(g_conv3d_ext_shapecalc, {x, w});
  auto &batchfy = ret_batchfy[i0];
  auto batchfy_conditional = ib->Equal(ib->TupleGetItem(batchfy, i0), ib->Value<int64_t>(1));
  auto cond_out_batchfy = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {x, dout}; },
    [&](Emitter *e) -> NodePtrList { return {e->ExpandDims(x, i0), e->ExpandDims(dout, i0)}; });
  nx = ib->TupleGetItem(cond_out_batchfy, i0);
  ndout = ib->TupleGetItem(cond_out_batchfy, i1);

  auto conv3d_grad_out = ib->ConvolutionGrad(ndout, nx, w, bias, stride_value, pad_value, dilation_value,
                                             transposed_value, output_padding_value, group_value, output_mask);
  auto dx = ib->TupleGetItem(conv3d_grad_out, i0);

  ndx = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {dx}; },
    [&](Emitter *e) -> NodePtrList { return {e->Squeeze(dx, MakeValue(ShapeVector{0}))}; });

  auto dw = ib->TupleGetItem(conv3d_grad_out, i1);
  auto dbias = ib->TupleGetItem(conv3d_grad_out, i2);
  return {ndx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(group_value)};
});

DEF_PURE_SHAPE_CALC(g_convolution_str_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_sizes = inputs.at(i0);
    auto weight_sizes = inputs.at(i1);
    auto stride_values = inputs.at(i2);
    auto dilation_values = inputs.at(i3);
    auto dim = input_sizes.size() - 2;

    std::vector<int64_t> padding_l;
    std::vector<int64_t> padding_r;
    auto symmetric_padding_true = 1;
    auto symmetric_padding_false = 0;
    auto symmetric_padding = symmetric_padding_true;

    for (size_t i = 0; i < dim; ++i) {
      auto i_stride = i % stride_values.size();
      auto i_dilation = i % dilation_values.size();
      auto stride = stride_values[i_stride];
      auto dilation = dilation_values[i_dilation];
      auto inputSize = input_sizes[i + 2];
      auto kernelSize = weight_sizes[i + 2];
      auto total_padding = dilation * (kernelSize - 1);
      if (stride > 2 && (total_padding % 2 == 1)) {
        auto wiggle_room = inputSize % stride - 1;
        if (wiggle_room > 0) {
          --total_padding;
        }
      }
      auto left = total_padding / 2;
      auto right = total_padding - left;

      padding_l.push_back(left);
      padding_r.push_back(right);
      if (left != right) {
        symmetric_padding = symmetric_padding_false;
      }
    }

    if (symmetric_padding) {
      std::vector<int64_t> pad_nd_temp(2 * dim, 0);
      return {{symmetric_padding}, pad_nd_temp, padding_l, pad_nd_temp};
    } else {
      std::vector<int64_t> pad_nd(2 * dim, 0);
      std::vector<int64_t> padding_neg_pad(2 * dim, 0);
      for (size_t i = 0; i < dim; ++i) {
        auto delta_pad = padding_r[i] - padding_l[i];
        auto pad_idx = 2 * (dim - 1 - i);  // F.pad goes from last dim to first
        if (delta_pad > 0) {
          pad_nd[pad_idx + 1] = delta_pad;
          padding_neg_pad[pad_idx + 1] = -delta_pad;
        } else {
          pad_nd[pad_idx] = delta_pad;
          padding_l[i] = padding_r[i];
          padding_neg_pad[pad_idx] = -delta_pad;
        }
      }
      return {{symmetric_padding}, pad_nd, padding_l, padding_neg_pad};
    }
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1, 4, 2, 4}; });

REG_BPROP_BUILDER("ConvolutionStr").SetUnusedInputs({i9}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  const auto &transposed_value = ib->GetInput(i6);
  const auto &output_padding_value = ib->GetInput(i7);
  const auto &group_value = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));

  auto pad_values = pad_value->BuildValue();
  auto pad_int_value = GetValue<int64_t>(pad_values);

  std::vector<int64_t> pad_vector = {0, 0};
  NodePtr conv2d_grad_out;
  NodePtr dx;

  if (pad_int_value == PadMode::SAME) {
    NodePtrList ret_shape = ib->ShapeCalc(g_convolution_str_shapecalc, {x, w, stride_value, dilation_value}, {2, 3});
    const auto &symmetric_padding = ret_shape[i0];
    const auto &pad_nd = ret_shape[i1];
    const auto &padding_l = ret_shape[i2];
    const auto &padding_neg_pad = ret_shape[i3];

    auto zero = ib->EmitValue(MakeValue<int64_t>(0));
    auto x_new = ib->ConstantPadND(x, pad_nd, zero);

    // // get conv2d_grad_out
    auto d_out_ori = ib->GetInput(i10);
    auto conv2d_grad_out_true = [&](Emitter *e) -> NodePtrList {
      return {e->ConvolutionGrad(d_out_ori, x, w, bias, stride_value, padding_l, dilation_value, transposed_value,
                                 output_padding_value, group_value, output_mask)};
    };
    auto conv2d_grad_out_false = [&](Emitter *e) -> NodePtrList {
      auto zero = e->EmitValue(MakeValue<int64_t>(0));
      auto x_new = e->ConstantPadND(x, pad_nd, zero);
      return {e->ConvolutionGrad(d_out_ori, x_new, w, bias, stride_value, padding_l, dilation_value, transposed_value,
                                 output_padding_value, group_value, output_mask)};
    };
    auto symmetric_padding_conditional = ib->Equal(ib->TupleGetItem(symmetric_padding, 0), ib->Value<int64_t>(1));
    conv2d_grad_out = ib->Conditional(symmetric_padding_conditional, conv2d_grad_out_true, conv2d_grad_out_false);

    // // get dx
    auto dx_true = [&](Emitter *e) -> NodePtrList { return {e->TupleGetItem(conv2d_grad_out, 0)}; };

    auto dx_false = [&](Emitter *e) -> NodePtrList {
      auto zero = e->EmitValue(MakeValue<int64_t>(0));
      return {e->ConstantPadND(e->TupleGetItem(conv2d_grad_out, 0), padding_neg_pad, zero)};
    };
    dx = ib->Conditional(symmetric_padding_conditional, dx_true, dx_false);
  } else if (pad_int_value == PadMode::VALID) {
    pad_vector = {0, 0};
    conv2d_grad_out =
      ib->ConvolutionGrad(ib->GetInput(i10), x, w, bias, stride_value, ib->EmitValue(MakeValue(pad_vector)),
                          dilation_value, transposed_value, output_padding_value, group_value, output_mask);
    dx = ib->TupleGetItem(conv2d_grad_out, 0);
  } else {
    MS_LOG(EXCEPTION) << "For [ConvolutionStr], padding string must be one of {'same', 'valid'}";
  }

  auto dw = ib->TupleGetItem(conv2d_grad_out, 1);
  auto dbias = ib->TupleGetItem(conv2d_grad_out, 2);
  return {dx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(transposed_value),
          ib->OutZeros(output_padding_value),
          ib->OutZeros(group_value)};
});

DEF_PURE_SHAPE_CALC(g_conv1d_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_sizes = inputs.at(i0);
    auto weight_sizes = inputs.at(i1);
    auto batchfy = (input_sizes.size() == weight_sizes.size());
    auto _batchfy = batchfy ? 1 : 0;
    return {{_batchfy}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {1}; });

REG_BPROP_BUILDER("Conv1DExt").SetUnusedInputs({i7}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &stride_value = ib->GetInput(i3);
  const auto &pad_value = ib->GetInput(i4);
  const auto &dilation_value = ib->GetInput(i5);
  const auto &group_value = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);

  auto bias_type = bias->abstract()->BuildType();
  bool bias_mask = bias_type->isa<TypeNone>() ? false : bias->need_compute_grad_out();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), w->need_compute_grad_out(), bias_mask};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto transposed_value = ib->EmitValue(MakeValue<bool>(false));
  std::vector<int64_t> output_padding_vec = {0};
  auto output_padding_value = ib->EmitValue(MakeValue(output_padding_vec));

  NodePtr nx, ndout, ndx;
  NodePtrList ret_batchfy = ib->ShapeCalc(g_conv1d_ext_shapecalc, {x, w});
  auto &batchfy = ret_batchfy[i0];
  auto batchfy_conditional = ib->Equal(ib->TupleGetItem(batchfy, i0), ib->Value<int64_t>(1));
  auto cond_out_batchfy = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {x, dout}; },
    [&](Emitter *e) -> NodePtrList { return {e->ExpandDims(x, i0), e->ExpandDims(dout, i0)}; });
  nx = ib->TupleGetItem(cond_out_batchfy, i0);
  ndout = ib->TupleGetItem(cond_out_batchfy, i1);

  auto conv1d_grad_out = ib->ConvolutionGrad(ndout, nx, w, bias, stride_value, pad_value, dilation_value,
                                             transposed_value, output_padding_value, group_value, output_mask);
  auto dx = ib->TupleGetItem(conv1d_grad_out, i0);
  ndx = ib->Conditional(
    batchfy_conditional, [&](Emitter *e) -> NodePtrList { return {dx}; },
    [&](Emitter *e) -> NodePtrList { return {e->Squeeze(dx, MakeValue(ShapeVector{0}))}; });
  auto dw = ib->TupleGetItem(conv1d_grad_out, i1);
  auto dbias = ib->TupleGetItem(conv1d_grad_out, i2);
  return {ndx,
          dw,
          dbias,
          ib->OutZeros(stride_value),
          ib->OutZeros(pad_value),
          ib->OutZeros(dilation_value),
          ib->OutZeros(group_value)};
});

REG_BPROP_BUILDER("MaxPool").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto format = GetValue<std::string>(ib->GetAttr("format"));
  auto kernel_size = GetValue<ShapeVector>(ib->GetAttr("kernel_size"));
  auto strides = GetValue<ShapeVector>(ib->GetAttr("strides"));
  if (format == "NHWC") {
    kernel_size = PoolToNHWC(kernel_size);
    strides = PoolToNHWC(strides);
  }
  auto dx = ib->Emit(kMaxPoolGradOpName, {x, out, dout},
                     {{"kernel_size", MakeValue(kernel_size)},
                      {"strides", MakeValue(strides)},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"data_format", ib->GetAttr("format")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("Embedding").FreeUselessValues_IO({i1, i3, i4}, {}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &weight = ib->GetInput(i1);
  const auto &padding_idx = ib->GetInput(i2);
  const auto &norm_type = ib->GetInput(i4);
  const auto &scale_grad_by_freq = ib->GetInput(i5);

  const auto &dout = ib->GetInput(i7);

  auto weight_shape = ib->Shape(weight);
  auto num_weights = ib->TupleGetItem(weight_shape, 0);
  auto dx = ib->EmbeddingDenseBackward(dout, input, num_weights, padding_idx, scale_grad_by_freq);
  return {ib->OutZeros(input),       dx,
          ib->OutZeros(padding_idx), ib->OutZeros(norm_type),
          ib->OutZeros(norm_type),   ib->OutZeros(scale_grad_by_freq)};
});

REG_BPROP_BUILDER("BiasAdd").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &bias = ib->GetInput(i1);
  const auto &format = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto format_value = format->BuildValue();
  auto format_int_opt = GetScalarValue<int64_t>(format_value);
  NodePtr dx = input_x->need_compute_grad_out() ? dout : ib->OutZeros(input_x);
  NodePtr grad_bias;
  if (bias->need_compute_grad_out()) {
    if (format_int_opt.has_value()) {
      if (format_int_opt.value() == Format::NCDHW) {
        auto format_new = ib->EmitValue(MakeValue<int64_t>(Format::NCHW));
        grad_bias = ib->Emit(kBiasAddGradOpName, {dout, format_new});
      } else {
        grad_bias = ib->Emit(kBiasAddGradOpName, {dout, format});
      }
    } else {
      auto true_branch = [](Emitter *e) -> NodePtrList { return {e->EmitValue(MakeValue<int64_t>(Format::NCHW))}; };
      auto false_branch = [&format](const Emitter *e) -> NodePtrList { return {format}; };
      auto cond = ib->Equal(format, ib->Value<int64_t>(Format::NCDHW));
      auto cond_block = ib->Conditional(cond, true_branch, false_branch);
      grad_bias = ib->Emit(kBiasAddGradOpName, {dout, cond_block});
    }
  } else {
    grad_bias = ib->OutZeros(bias);
  }
  return {dx, grad_bias, ib->OutZeros(format)};
});

REG_BPROP_BUILDER("AddLayerNormV2").FreeUselessValues_IO({i3}, {0, 3}).SetBody(BODYFUNC(ib) {
  // x1, x2, gamma, beta, epsilon, additionalOut, (y, mean, rstd, x), (dy, dmean, drstd, dx)
  const auto &x1 = ib->GetInput(i0);
  const auto &x2 = ib->GetInput(i1);
  const auto &gamma = ib->GetInput(i2);
  const auto &epsilon = ib->GetInput(i4);
  const auto &additionalOut = ib->GetInput(i5);
  auto additional_out_opt = GetScalarValue<bool>(additionalOut->BuildValue());
  const auto &out = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i7);
  auto rstd = ib->TupleGetItem(out, i2);
  auto mean = ib->TupleGetItem(out, i1);
  auto dy = ib->TupleGetItem(dout, i0);
  auto sum_optional = ib->TupleGetItem(dout, i3);
  if (!additional_out_opt.has_value()) {
    auto true_branch = [&sum_optional](Emitter *e) -> NodePtrList { return {sum_optional}; };
    auto false_branch = [&dy](Emitter *e) -> NodePtrList {
      auto dy_dtype = e->Value(static_cast<int64_t>(dy->dtype()->type_id()));
      return {e->Zeros(e->Shape(dy), dy_dtype)};
    };
    auto additional_out_true = ib->Equal(additionalOut, ib->Value<bool>(true));
    sum_optional = ib->Conditional(additional_out_true, true_branch, false_branch);
  } else {
    if (!additional_out_opt.value()) {
      sum_optional = ib->Zeros(ib->Shape(dy), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dy))));
    }
  }
  auto grad_out = ib->AddLayerNormGrad(dy, x1, x2, rstd, mean, gamma, sum_optional);
  auto dx = ib->TupleGetItem(grad_out, i0);
  auto dgamma = ib->TupleGetItem(grad_out, i1);
  auto dbeta = ib->TupleGetItem(grad_out, i2);
  return {dx, dx, dgamma, dbeta, ib->OutZeros(epsilon), ib->OutZeros(additionalOut)};
});

DEF_PURE_SHAPE_CALC(g_dense_shapecalc0)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(i0);
    auto &w_shape = inputs.at(i1);
    auto &b_shape = inputs.at(i2);
    auto &dout_shape = inputs.at(i3);

    auto get_product_dim = [](const ShapeVector &shape) -> int64_t {
      return std::accumulate(shape.begin(), shape.end() - 1, static_cast<int64_t>(1), std::multiplies<int64_t>());
    };

    ShapeVector x_2d_shape = {get_product_dim(x_shape), x_shape.back()};
    ShapeVector w_2d_shape = {get_product_dim(w_shape), w_shape.back()};
    ShapeVector dout_2d_shape;
    if (dout_shape.size() == 0) {
      dout_2d_shape = {1, 1};
    } else if (w_shape.size() == 1) {
      dout_2d_shape = {-1, 1};
    } else {
      dout_2d_shape = {get_product_dim(dout_shape), dout_shape.back()};
    }
    ShapeVector b_reduce_shape;
    if (b_shape.size() > 0) {
      b_reduce_shape.push_back(0);
    }

    return {x_2d_shape, w_2d_shape, dout_2d_shape, b_reduce_shape, x_shape, w_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto &x_shape = inputs[i0];
    auto &w_shape = inputs[i1];
    auto &b_shape = inputs[i2];
    auto &dout_shape = inputs[i3];

    auto b_reduce_rank = -1LL;
    if (!IsDynamicRank(b_shape)) {
      if (b_shape.size() > 0) {
        b_reduce_rank = 1;
      } else {
        b_reduce_rank = 0;
      }
    }

    return {
      IsDynamicRank(x_shape) ? -1LL : 2LL,
      IsDynamicRank(w_shape) ? -1LL : 2LL,
      IsDynamicRank(dout_shape) ? -1LL : 2LL,
      b_reduce_rank,
      IsDynamicRank(x_shape) ? -1LL : static_cast<int64_t>(x_shape.size()),
      IsDynamicRank(w_shape) ? -1LL : static_cast<int64_t>(w_shape.size()),
    };
  });

REG_BPROP_BUILDER("Dense").FreeUselessValues_IO({i2}, {}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(i0);
  auto w = ib->GetInput(i1);
  const auto &b = ib->GetInput(i2);
  auto dout = ib->GetInput(i4);
  auto dtype = ib->GetDtype(x);
  bool is_complex = (*dtype) == (*kComplex64) || (*dtype) == (*kComplex128);
  bool no_bias = ib->GetDtype(b)->isa<TypeNone>();

  NodePtr dx = nullptr;
  NodePtr dw = nullptr;
  NodePtr db = nullptr;

  NodePtrList ret_shape = ib->ShapeCalc(g_dense_shapecalc0, {x, w, b, dout});
  const auto &x_2d_shape = ret_shape[i0];
  const auto &w_2d_shape = ret_shape[i1];
  const auto &dout_2d_shape = ret_shape[i2];
  const auto &b_reduce_shape = ret_shape[i3];
  const auto &x_shape = ret_shape[i4];
  const auto &w_shape = ret_shape[i5];

  dout = ib->Reshape(dout, dout_2d_shape);
  auto dout_conj = is_complex ? ib->Emit("Conj", {dout}) : dout;
  if (x->need_compute_grad_out()) {
    w = ib->Reshape(w, w_2d_shape);
    dx = no_bias ? ib->MatMulExt(dout_conj, w) : ib->MatMul(dout_conj, w, false, false);
    dx = is_complex ? ib->Emit("Conj", {dx}) : dx;
    dx = ib->Reshape(dx, x_shape);
  } else {
    dx = ib->OutZeros(x);
  }

  if (w->need_compute_grad_out()) {
    x = ib->Reshape(x, x_2d_shape);
    if (no_bias) {
      if (ops::UseOptimizedOpImpl()) {
        MS_LOG(DEBUG) << "Using optimized operator implementation in Dense's backward.";
        dw = ib->MatMulExt(ib->Transpose(dout_conj, ib->Value(ShapeVector{1, 0})), x);
      } else {
        x = ib->Transpose(x, ib->Value(ShapeVector{1, 0}));
        dw = ib->Transpose(ib->MatMulExt(x, dout_conj), ib->Value(ShapeVector{1, 0}));
      }
    } else {
      dw = ib->MatMul(dout_conj, x, true, false);
    }
    dw = is_complex ? ib->Emit("Conj", {dw}) : dw;
    dw = ib->Reshape(dw, w_shape);
  } else {
    dw = ib->OutZeros(w);
  }
  db = b->need_compute_grad_out() ? ib->SumExt(dout, b_reduce_shape, ib->Value(false)) : ib->OutZeros(b);
  return {dx, dw, db};
});

REG_BPROP_BUILDER("ReLU").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->ReluGrad(dout, out);
  return {dx};
});

REG_BPROP_BUILDER("InplaceReLU").FreeUselessValues_I({}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->ReluGrad(dout, out);
  return {dx};
});

REG_BPROP_BUILDER("InplaceSiLU").FreeUselessValues_O({}).CloneInplaceInput().SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->SiLUGrad(dout, x);
  return {dx};
});

REG_BPROP_BUILDER("Threshold").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &threshold = ib->GetInput(i1);
  const auto &value = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->ThresholdGrad(dout, input, threshold);
  return {dx, ib->OutZeros(threshold), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("InplaceThreshold").SetUnusedInputs({i3}).CloneInplaceInput().SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &threshold = ib->GetInput(i1);
  const auto &value = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->ThresholdGrad(dout, input, threshold);
  return {dx, ib->OutZeros(threshold), ib->OutZeros(value)};
});

REG_BPROP_BUILDER("ThresholdGrad").FreeUselessValues(FreeTensorsOfThresholdGrad).SetBody(BODYFUNC(ib) {
  const auto &grad_output = ib->GetInput(i0);
  const auto &input = ib->GetInput(i1);
  const auto &threshold = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (grad_output->need_compute_grad_out()) {
    dx = ib->ThresholdGrad(dout, input, threshold);
  } else {
    dx = ib->OutZeros(grad_output);
  }
  if (input->need_compute_grad_out()) {
    dy = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(dout))));
  } else {
    dy = ib->OutZeros(input);
  }
  return {dx, dy, ib->OutZeros(threshold)};
});

DEF_PURE_SHAPE_CALC(g_topk_1)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray { return {{-1, inputs.at(0).back()}}; })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> std::vector<int64_t> { return {2}; });

DEF_PURE_SHAPE_CALC(g_topk_2)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto in_shape = inputs.at(0);
    auto in_lastdim = in_shape.back();
    auto outerdim = inputs.at(1)[0];  // k
    auto in_shape_1d_x = ShapeVector(
      1, std::accumulate(in_shape.begin(), in_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>()));
    return {in_shape_1d_x, {outerdim * in_lastdim}, {in_lastdim}};
  })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> std::vector<int64_t> { return {1, 1, 1}; });

REG_BPROP_BUILDER("TopK").FreeUselessValues_IO({i0, i1}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto indices = ib->TupleGetItem(out, i1);
  auto dout0 = ib->TupleGetItem(dout, i0);
  auto in_shape = ib->GetShape(input_x);
  auto indices_shape = ib->GetShape(indices);
  if (IsDynamic(in_shape) || IsDynamic(indices_shape)) {
    auto re0 = ib->ShapeCalc(g_topk_1, {indices})[0];
    NodePtr ind_2d = ib->Reshape(indices, re0);
    auto res = ib->ShapeCalc(g_topk_2, {input_x, ind_2d});
    auto in_shape_1d = res[0];
    auto range_flatten_index =
      ib->Range(ib->Value<int64_t>(0), ib->TupleGetItem(res[1], 0), ib->TupleGetItem(res[2], 0));
    auto ind = ib->Reshape(ind_2d + ib->Reshape(range_flatten_index, {-1, 1}), {-1, 1});
    auto out_grad = ib->ScatterNd(ind, ib->Reshape(dout0, {-1}), in_shape_1d);
    out_grad = ib->Reshape(out_grad, ib->Shape(input_x));
    auto grad_k = ib->OutZeros(ib->GetInput(i1));
    return {out_grad, grad_k};
  } else {
    auto shape = ib->GetShape(indices);
    auto ind_lastdim = shape.back();
    auto ind_2d = ib->Reshape(indices, {-1, ind_lastdim});
    auto in_lastdim = in_shape.back();
    auto outerdim = ib->GetShape(ind_2d)[0];  // k
    std::vector<int64_t> range_flatten_index_vec(LongToSize(outerdim));
    for (int64_t i = 0; i < outerdim; i++) {
      range_flatten_index_vec[static_cast<size_t>(i)] = i * in_lastdim;
    }
    auto range_flatten_index = ib->Tensor(range_flatten_index_vec, ib->GetDtype(indices));
    auto in_shape_1d = ib->Value(ShapeVector(
      1, std::accumulate(in_shape.begin(), in_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>())));
    auto ind = ib->Reshape(ind_2d + ib->Reshape(range_flatten_index, {-1, 1}), {-1, 1});
    auto out_grad = ib->ScatterNd(ind, ib->Reshape(dout0, {-1}), in_shape_1d);
    out_grad = ib->Reshape(out_grad, in_shape);
    auto grad_k = ib->OutZeros(ib->GetInput(i1));
    return {out_grad, grad_k};
  }
});

REG_BPROP_BUILDER("TopkExt").FreeUselessValues_IO({i0, i3, i4}, {i0}).SetBody(BODYFUNC(ib) {
  // x, k, dim, largest, sorted, out(values, indices), dout(grad_values, grad_indices)
  const auto &input_x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto indices = ib->TupleGetItem(out, i1);
  auto dout0 = ib->TupleGetItem(dout, i0);
  const auto &dim = ib->GetInput(i2);
  auto out_grad = ib->Zeros(ib->Shape(input_x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input_x))));
  (void)ib->InplaceScatterSrc(out_grad, dim, indices, dout0);
  return {out_grad, ib->OutZeros(ib->GetInput(i1)), ib->OutZeros(ib->GetInput(i2)), ib->OutZeros(ib->GetInput(i3)),
          ib->OutZeros(ib->GetInput(i4))};
});

REG_BPROP_BUILDER("Kthvalue").FreeUselessValues_IO({i1}, {i0}).SetBody(BODYFUNC(ib) {
  // x, k, dim, keepdim, out(values, indices), dout(grad_values, grad_indices)
  const auto &input_x = ib->GetInput(i0);
  const auto &keepdim = ib->GetInput(i3);
  auto keepdim_opt = mindspore::GetScalarValue<bool>(keepdim->BuildValue());
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto indices = ib->TupleGetItem(out, i1);
  auto dout0 = ib->TupleGetItem(dout, i0);
  const auto &dim = ib->GetInput(i2);
  auto dim_opt = mindspore::GetScalarValue<int64_t>(dim->BuildValue());
  auto zeros = ib->Zeros(ib->Shape(input_x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input_x))));
  auto reduce = ib->Value(static_cast<int64_t>(Reduce::REDUCE_NONE));
  if (keepdim_opt.has_value()) {
    if (!keepdim_opt.value()) {
      auto dim_value = dim_opt.value();
      indices = ib->ExpandDims(indices, dim_value);
      dout0 = ib->ExpandDims(dout0, dim_value);
    }
  } else {
    auto true_branch = [&indices, &dout0](Emitter *e) -> NodePtrList { return {indices, dout0}; };
    auto false_branch = [&indices, &dout0, &dim](Emitter *e) -> NodePtrList {
      return {e->ExpandDims(indices, dim), e->ExpandDims(dout0, dim)};
    };
    auto unsqueezed_outputs = ib->Conditional(keepdim, true_branch, false_branch);
    indices = ib->TupleGetItem(unsqueezed_outputs, 0);
    dout0 = ib->TupleGetItem(unsqueezed_outputs, 1);
  }
  auto out_grad = ib->Scatter(zeros, dim, indices, dout0, reduce);
  return {out_grad, ib->OutZeros(ib->GetInput(i1)), ib->OutZeros(ib->GetInput(i2)), ib->OutZeros(ib->GetInput(i3))};
});

REG_BPROP_BUILDER("PReLU").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto res = ib->PReLUGrad(dout, x, w);
  auto dx = ib->TupleGetItem(res, i0);
  auto dw = ib->TupleGetItem(res, i1);
  return {dx, dw};
});

REG_BPROP_BUILDER("LeakyReLUExt").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &negative_slope = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->LeakyReLUGradExt(dout, input, negative_slope, ib->Value(false));
  return {dx, ib->OutZeros(negative_slope)};
});

REG_BPROP_BUILDER("SigmoidCrossEntropyWithLogits").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("SigmoidCrossEntropyWithLogitsGrad", {x, y, dout});
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("Pad").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto paddings = ib->GetAttr<std::vector<std::vector<int64_t>>>("paddings");
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  std::vector<int64_t> begin;
  for (const auto &item : paddings) {
    begin.push_back(item.at(0));
  }
  auto x_shape = ib->Shape(x);
  auto dx = ib->Slice(dout, ib->EmitValue(MakeValue(begin)), x_shape);
  return {dx};
});

REG_BPROP_BUILDER("ROIAlign").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &inputs = ib->GetInput(i0);
  const auto &rois = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shp = ib->GetShape(inputs);
  auto inputs_shape = ib->Shape(inputs);
  auto dx = ib->Emit("ROIAlignGrad", {dout, rois, inputs_shape},
                     {{"pooled_height", ib->GetAttr("pooled_height")},
                      {"pooled_width", ib->GetAttr("pooled_width")},
                      {"xdiff_shape", MakeValue(shp)},
                      {"spatial_scale", ib->GetAttr("spatial_scale")},
                      {"sample_num", ib->GetAttr("sample_num")}});
  return {dx, ib->OutZeros(rois)};
});

REG_BPROP_BUILDER("LRN").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("LRNGrad", {dout, x, out},
                     {{"depth_radius", ib->GetAttr("depth_radius")},
                      {"bias", ib->GetAttr("bias")},
                      {"alpha", ib->GetAttr("alpha")},
                      {"beta", ib->GetAttr("beta")}});
  return {dx};
});

REG_BPROP_BUILDER("Dropout").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &keep_prob = ib->GetInput(i1);
  const auto &seed0 = ib->GetInput(i2);
  const auto &seed1 = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto mask = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetInput(i1)->BuildValue()}});
  return {dx, ib->OutZeros(keep_prob), ib->OutZeros(seed0), ib->OutZeros(seed1)};
});

REG_BPROP_BUILDER("DropoutExt").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &p = ib->GetInput(i1);
  const auto &seed = ib->GetInput(i2);
  const auto &offset = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto mask = ib->TupleGetItem(out, i1);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dx = ib->DropoutGradExt(dy, mask, p);
  return {dx, ib->OutZeros(p), ib->OutZeros(seed), ib->OutZeros(offset)};
});

REG_BPROP_BUILDER("BinaryCrossEntropy").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &reduction = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->BinaryCrossEntropyGrad(x, y, dout, weight, reduction);
  NodePtr dy = nullptr;
  if (y->need_compute_grad_out()) {
    bool weight_type_none = ib->GetDtype(weight)->isa<TypeNone>();
    dy = ib->Mul(ib->Sub(ib->Log(ib->Sub(ib->Tensor(1, ib->GetDtype(x)), x)), ib->Log(x)), dout);
    if (!weight_type_none) {
      dy = ib->Mul(dy, weight);
    }
    auto reduction_value = GetValue<int64_t>(reduction->BuildValue());
    if (reduction_value == 1) {
      if (IsDynamic(ib->GetShape(dx))) {
        auto res = ib->DynSize(y, ib->GetDtype(dy));
        dy = ib->RealDiv(dy, res);
      } else {
        dy = ib->RealDiv(dy, ib->Tensor(ib->GetSize(y), ib->GetDtype(y)));
      }
    }
  } else {
    dy = ib->OutZeros(y);
  }
  return {dx, dy, ib->OutZeros(weight), ib->OutZeros(reduction)};
});

REG_BPROP_BUILDER("DropoutGrad").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &mask = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dy = dout;
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetAttr("keep_prob")}});
  return {dx, ib->OutZeros(mask)};
});

REG_BPROP_BUILDER("DeformableOffsets").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &offsets = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto out_grad = ib->Emit("DeformableOffsetsGrad", {dout, x, offsets},
                           {{"strides", ib->GetAttr("strides")},
                            {"pads", ib->GetAttr("pads")},
                            {"ksize", ib->GetAttr("ksize")},
                            {"dilations", ib->GetAttr("dilations")},
                            {"format", ib->GetAttr("format")},
                            {"data_format", ib->GetAttr("format")},
                            {"deformable_groups", ib->GetAttr("deformable_groups")},
                            {"modulated", ib->GetAttr("modulated")}});
  return {ib->TupleGetItem(out_grad, 0), ib->TupleGetItem(out_grad, 1)};
});

REG_BPROP_BUILDER("LSTM").FreeUselessValues(FreeTensorsOfLSTM).SetBody(BODYFUNC(ib) {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  auto proj_size = ib->GetAttr("proj_size");
  const auto &x = ib->GetInput(i0);
  const auto &hx = ib->GetInput(i1);
  const auto &cx = ib->GetInput(i2);
  const auto &w = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto target = ib->GetTargetFromContext();
  if (target == "CPU") {
    auto y = ib->TupleGetItem(out, i0);
    auto hy = ib->TupleGetItem(out, i1);
    auto cy = ib->TupleGetItem(out, i2);
    auto reserve = ib->TupleGetItem(out, i3);
    auto dy = ib->TupleGetItem(dout, i0);
    auto dhy = ib->TupleGetItem(dout, i1);
    auto dcy = ib->TupleGetItem(dout, i2);
    auto res = ib->Emit("LSTMGrad", {x, hx, cx, w, y, hy, cy, dy, dhy, dcy, reserve},
                        {{"input_size", input_size},
                         {"hidden_size", hidden_size},
                         {"num_layers", num_layers},
                         {"has_bias", has_bias},
                         {"bidirectional", bidirectional},
                         {"dropout", dropout},
                         {"proj_size", proj_size}});
    auto dx = ib->TupleGetItem(res, i0);
    auto dhx = ib->TupleGetItem(res, i1);
    auto dcx = ib->TupleGetItem(res, i2);
    auto dw = ib->TupleGetItem(res, i3);
    return {dx, dhx, dcx, dw};
  }
  auto y = ib->TupleGetItem(out, i0);
  auto reserve = ib->TupleGetItem(out, i3);
  auto state = ib->TupleGetItem(out, i4);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dhy = ib->TupleGetItem(dout, i1);
  auto dcy = ib->TupleGetItem(dout, i2);
  auto res1 = ib->Emit("LSTMGradData", {y, dy, dhy, dcy, w, hx, cx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout},
                        {"proj_size", proj_size}});
  auto dx = ib->TupleGetItem(res1, i0);
  auto dhx = ib->TupleGetItem(res1, i1);
  auto dcx = ib->TupleGetItem(res1, i2);
  auto dw = ib->Emit("LSTMGradWeight", {ib->Depend(x, dx), hx, y, reserve, state},
                     {{"input_size", input_size},
                      {"hidden_size", hidden_size},
                      {"num_layers", num_layers},
                      {"has_bias", has_bias},
                      {"bidirectional", bidirectional},
                      {"dropout", dropout},
                      {"proj_size", proj_size}});
  return {dx, dhx, dcx, dw};
});

REG_BPROP_BUILDER("CudnnGRU.NotReady").FreeUselessValues_O({i1}).SetBody(BODYFUNC(ib) {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  const auto &x = ib->GetInput(i0);
  const auto &hx = ib->GetInput(i1);
  const auto &w = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto y = ib->TupleGetItem(out, i0);
  auto reserve = ib->TupleGetItem(out, i2);
  auto state = ib->TupleGetItem(out, i3);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dhy = ib->TupleGetItem(dout, i1);
  auto res1 = ib->Emit("GruGradData", {y, dy, dhy, w, hx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout}});
  auto dx = ib->TupleGetItem(res1, i0);
  auto dhx = ib->TupleGetItem(res1, i1);
  auto dw = w->need_compute_grad_out() ? ib->Emit("GruGradWeight", {ib->Depend(x, dx), hx, y, reserve, state},
                                                  {{"input_size", input_size},
                                                   {"hidden_size", hidden_size},
                                                   {"num_layers", num_layers},
                                                   {"has_bias", has_bias},
                                                   {"bidirectional", bidirectional},
                                                   {"dropout", dropout}})
                                       : ib->OutZeros(w);
  return {dx, dhx, dw};
});

REG_BPROP_BUILDER("MirrorPad").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("MirrorPadGrad", {dout, paddings}, {{kAttrMode, ib->GetAttr(kAttrMode)}});
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("GLU").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto x = ib->GetInput(i0);
  const auto axis = ib->GetInput(i1);
  const auto dout = ib->GetInput(i3);
  const auto dx = ib->GluGrad(dout, x, axis);
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("MaxPoolWithArgmaxV2").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MaxPoolGradWithArgmaxV2", {x, ib->TupleGetItem(dout, i0), ib->TupleGetItem(out, i1)},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"argmax_type", ib->GetAttr("argmax_type")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolWithMask").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &strides = ib->GetInput(i2);
  const auto &pads = ib->GetInput(i3);
  const auto &dilation = ib->GetInput(i4);
  const auto &ceil_mode = ib->GetInput(i5);
  const auto &argmax_type = ib->GetInput(i6);
  const auto &out = ib->GetInput(i7);
  const auto &dout = ib->GetInput(i8);
  auto dx = ib->MaxPoolGradWithMask(x, ib->TupleGetItem(dout, i0), ib->TupleGetItem(out, i1), kernel_size, strides,
                                    pads, dilation, ceil_mode, argmax_type);
  auto g_kernel_size = ib->OutZeros(kernel_size);
  auto g_strides = ib->OutZeros(strides);
  auto g_pads = ib->OutZeros(pads);
  auto g_dilation = ib->OutZeros(dilation);
  auto g_ceil_mode = ib->OutZeros(ceil_mode);
  auto g_argmax_type = ib->OutZeros(argmax_type);
  return {dx, g_kernel_size, g_strides, g_pads, g_dilation, g_ceil_mode, g_argmax_type};
});

REG_BPROP_BUILDER("MaxPoolWithIndices").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &strides = ib->GetInput(i2);
  const auto &pads = ib->GetInput(i3);
  const auto &dilation = ib->GetInput(i4);
  const auto &ceil_mode = ib->GetInput(i5);
  const auto &argmax_type = ib->GetInput(i6);
  const auto &out = ib->GetInput(i7);
  const auto &dout = ib->GetInput(i8);
  auto dx = ib->MaxPoolGradWithIndices(x, ib->TupleGetItem(dout, i0), ib->TupleGetItem(out, i1), kernel_size, strides,
                                       pads, dilation, ceil_mode, argmax_type);
  auto g_kernel_size = ib->OutZeros(kernel_size);
  auto g_strides = ib->OutZeros(strides);
  auto g_pads = ib->OutZeros(pads);
  auto g_dilation = ib->OutZeros(dilation);
  auto g_ceil_mode = ib->OutZeros(ceil_mode);
  auto g_argmax_type = ib->OutZeros(argmax_type);
  return {dx, g_kernel_size, g_strides, g_pads, g_dilation, g_ceil_mode, g_argmax_type};
});

REG_BPROP_BUILDER("GroupNorm").FreeUselessValues_IO({i4}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &num_groups = ib->GetInput(i1);
  const auto &gamma = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &epsilon = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);

  auto result =
    ib->GroupNormGrad(ib->TupleGetItem(dout, 0), x, ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2), gamma,
                      num_groups, ib->Value<bool>(x->need_compute_grad_out()),
                      ib->Value<bool>(gamma->need_compute_grad_out()), ib->Value<bool>(beta->need_compute_grad_out()));

  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  auto grad_group = ib->OutZeros(num_groups);
  auto grad_epsilon = ib->OutZeros(epsilon);
  return {d_x, grad_group, d_gamma, d_beta, grad_epsilon};
});

REG_BPROP_BUILDER("LayerNormExt").FreeUselessValues_IO({i4}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &normalized_shape = ib->GetInput(i1);
  const auto &gamma = ib->GetInput(i2);
  const auto &beta = ib->GetInput(i3);
  const auto &eps = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto normalized_shape_ptr = normalized_shape->BuildValue();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), gamma->need_compute_grad_out(),
                                          beta->need_compute_grad_out()};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto result = ib->LayerNormGradExt(ib->TupleGetItem(dout, 0), x, normalized_shape, ib->TupleGetItem(out, 1),
                                     ib->TupleGetItem(out, 2), gamma, beta, output_mask);
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  auto grad_normalized_shape = ib->OutZeros(normalized_shape);
  auto grad_eps = ib->OutZeros(eps);
  return {d_x, grad_normalized_shape, d_gamma, d_beta, grad_eps};
});

REG_BPROP_BUILDER("LayerNorm").FreeUselessValues_IO({i2, i5}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &gamma = ib->GetInput(i1);
  const auto &begin_norm_axis = ib->GetInput(i3);
  const auto &begin_params_axis = ib->GetInput(i4);
  const auto &epsilon = ib->GetInput(i5);
  const auto &out = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i7);
  DAttr attrs;
  attrs.push_back(std::make_pair("epsilon", epsilon->BuildValue()));
  auto result = ib->Emit("LayerNormGrad",
                         {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 2), ib->TupleGetItem(out, 1), gamma,
                          begin_norm_axis, begin_params_axis},
                         attrs);
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  auto grad_begin_norm_axis = ib->OutZeros(begin_norm_axis);
  auto grad_begin_params_axis = ib->OutZeros(begin_params_axis);
  auto grad_epsilon = ib->OutZeros(epsilon);
  return {d_x, d_gamma, d_beta, grad_begin_norm_axis, grad_begin_params_axis, grad_epsilon};
});

REG_BPROP_BUILDER("LayerNormV3").FreeUselessValues_IO({i2}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &gamma = ib->GetInput(i1);
  const auto &begin_norm_axis = ib->GetInput(i3);
  const auto &begin_params_axis = ib->GetInput(i4);
  const auto &epsilon = ib->GetInput(i5);
  const auto &out = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i7);
  DAttr attrs;
  attrs.push_back(std::make_pair("epsilon", epsilon->BuildValue()));
  auto result = ib->Emit("LayerNormGradV3",
                         {ib->TupleGetItem(dout, 0), x, ib->TupleGetItem(out, 2), ib->TupleGetItem(out, 1), gamma,
                          begin_norm_axis, begin_params_axis},
                         attrs);
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  auto grad_begin_norm_axis = ib->OutZeros(begin_norm_axis);
  auto grad_begin_params_axis = ib->OutZeros(begin_params_axis);
  auto grad_epsilon = ib->OutZeros(epsilon);
  return {d_x, d_gamma, d_beta, grad_begin_norm_axis, grad_begin_params_axis, grad_epsilon};
});

REG_BPROP_BUILDER("LayerNormGrad").FreeUselessValues_I({i7}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dy = ib->GetInput(i1);
  const auto &variance = ib->GetInput(i2);
  const auto &mean = ib->GetInput(i3);
  const auto &gamma = ib->GetInput(i4);
  const auto &begin_norm_axis = ib->GetInput(i5);
  const auto &begin_params_axis = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);
  auto result = ib->Emit("LayerNormGradGrad",
                         {x, dy, variance, mean, gamma, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1),
                          ib->TupleGetItem(dout, 2), begin_norm_axis, begin_params_axis},
                         {});

  auto d_x = ib->TupleGetItem(result, 0);
  auto d_dy = ib->TupleGetItem(result, 1);
  auto d_gamma = ib->TupleGetItem(result, 2);
  auto grad_begin_norm_axis = ib->OutZeros(begin_norm_axis);
  auto grad_begin_params_axis = ib->OutZeros(begin_params_axis);
  return {d_x, d_dy, ib->OutZeros(variance), ib->OutZeros(mean), d_gamma, grad_begin_norm_axis, grad_begin_params_axis};
});

REG_BPROP_BUILDER("L2Normalize").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx =
    ib->Emit("L2NormalizeGrad", {x, out, dout}, {{"axis", ib->GetAttr("axis")}, {"epsilon", ib->GetAttr("epsilon")}});
  return {dx};
});

REG_BPROP_BUILDER("SoftmaxCrossEntropyWithLogits").FreeUselessValues_IO({i0, i1}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &labels = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto grad = ib->TupleGetItem(out, 1);
  grad = ib->Mul(grad, (ib->ExpandDims(ib->TupleGetItem(dout, 0), -1)));
  return {grad, ib->OutZeros(labels)};
});

REG_BPROP_BUILDER("NLLLoss").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &logits = ib->GetInput(i0);
  const auto &labels = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &reduction = ib->GetInput(i3);
  const auto &ignore_index = ib->GetInput(i4);

  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto total_weight = ib->TupleGetItem(out, 1);
  auto dout_x = ib->TupleGetItem(dout, 0);
  auto dx = ib->NLLLossGrad(logits, dout_x, labels, weight, total_weight, reduction, ignore_index);
  return {dx, ib->OutZeros(labels), ib->OutZeros(weight), ib->OutZeros(reduction), ib->OutZeros(ignore_index)};
});

REG_BPROP_BUILDER("NLLLoss2d").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  // logits,labels weight reduction
  const auto &logits = ib->GetInput(i0);
  const auto &labels = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &reduction = ib->GetInput(i3);
  const auto &ignore_index = ib->GetInput(i4);

  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto total_weight = ib->TupleGetItem(out, 1);
  auto dout_x = ib->TupleGetItem(dout, 0);
  auto dx = ib->NLLLoss2dGrad(dout_x, logits, labels, weight, reduction, ignore_index, total_weight);
  return {dx, ib->OutZeros(labels), ib->OutZeros(weight), ib->OutZeros(reduction), ib->OutZeros(ignore_index)};
});

REG_BPROP_BUILDER("ResizeBilinear").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("ResizeBilinearGrad", {dout, x, ib->EmitValue(ib->GetAttr("align_corners")),
                                            ib->EmitValue(ib->GetAttr("half_pixel_centers"))});
  return {dx};
});

REG_BPROP_BUILDER("OneHot").SetUnusedInputs({i0, i1, i2, i3, i5, i6}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i0);
  const auto &depth = ib->GetInput(i1);
  const auto &on_value = ib->GetInput(i2);
  const auto &off_value = ib->GetInput(i3);
  const auto &axis = ib->GetInput(i4);
  return {ib->OutZeros(indices), ib->OutZeros(ib->Tensor(0, ib->GetDtype(depth))), ib->OutZeros(on_value),
          ib->OutZeros(off_value), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("OneHotExt").SetUnusedInputs({i0, i1, i2, i3, i5, i6}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i0);
  const auto &depth = ib->GetInput(i1);
  const auto &on_value = ib->GetInput(i2);
  const auto &off_value = ib->GetInput(i3);
  const auto &axis = ib->GetInput(i4);
  return {ib->OutZeros(indices), ib->OutZeros(ib->Tensor(0, ib->GetDtype(depth))), ib->OutZeros(on_value),
          ib->OutZeros(off_value), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("SmoothL1Loss").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &prediction = ib->GetInput(i0);
  const auto &target = ib->GetInput(i1);
  const auto &beta = ib->GetInput(i2);
  const auto &reduction = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = prediction->need_compute_grad_out() ? ib->SmoothL1LossGrad(prediction, target, dout, beta, reduction)
                                                : ib->OutZeros(prediction);
  auto dy = target->need_compute_grad_out() ? ib->SmoothL1LossGrad(target, prediction, dout, beta, reduction)
                                            : ib->OutZeros(target);
  return {dx, dy, ib->OutZeros(beta), ib->OutZeros(reduction)};
});

REG_BPROP_BUILDER("L1LossExt").SetUnusedInputs({i3}).SetBody((BODYFUNC(ib) {
  // input, target, reduction, out, dout
  auto grad_output = ib->GetInput(i4);
  auto input = ib->GetInput(i0);
  auto target = ib->GetInput(i1);
  auto reduction = ib->GetInput(i2);

  auto dx = ib->L1LossBackwardExt(grad_output, input, target, reduction);
  auto dy = ib->L1LossBackwardExt(grad_output, target, input, reduction);
  std::vector<NodePtr> ret = BinopGradCommon(ib, input, target, dx, dy);
  ret[i0] = ib->Cast(ret[i0], ib->GetDtype(input));
  ret[i1] = ib->Cast(ret[i1], ib->GetDtype(target));
  ret.emplace_back(ib->OutZeros(reduction));
  return ret;
}));
REG_BPROP_BUILDER("MSELossExt").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &target = ib->GetInput(i1);
  const auto &reduction = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr dx = nullptr;
  NodePtr dtarget = nullptr;
  if (input->need_compute_grad_out()) {
    dx = ib->MSELossGradExt(dout, input, target, reduction);
    if (target->need_compute_grad_out()) {
      dtarget = ib->Neg(dx);
    }
  } else {
    if (target->need_compute_grad_out()) {
      dtarget = ib->MSELossGradExt(dout, target, input, reduction);
    }
  }
  std::vector<NodePtr> ret = BinopGradCommon(ib, input, target, dx, dtarget);
  ret.emplace_back(ib->OutZeros(reduction));
  return ret;
});

REG_BPROP_BUILDER("L2Loss").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Mul(x, dout);
  return {dx};
});

REG_BPROP_BUILDER("RNNTLoss").FreeUselessValues_IO({}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &labels = ib->GetInput(i1);
  const auto &act_lens = ib->GetInput(i2);
  const auto &label_lens = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  auto grad = ib->TupleGetItem(out, 1);
  return {grad, ib->OutZeros(labels), ib->OutZeros(act_lens), ib->OutZeros(label_lens)};
});

REG_BPROP_BUILDER("Conv3D").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto x_shape = ib->Shape(x);
  auto w_shape = ib->Shape(w);
  auto dx = x->need_compute_grad_out() ? ib->Emit("Conv3DBackpropInput", {w, dout, x_shape},
                                                  {{"pad_mode", ib->GetAttr("pad_mode")},
                                                   {"pad", ib->GetAttr("pad")},
                                                   {"strides", ib->GetAttr("strides")},
                                                   {"dilations", ib->GetAttr("dilations")},
                                                   {"stride", ib->GetAttr("strides")},
                                                   {"dilation", ib->GetAttr("dilations")},
                                                   {"group", ib->GetAttr("groups")},
                                                   {"groups", ib->GetAttr("groups")},
                                                   {"format", ib->GetAttr("format")},
                                                   {"data_format", ib->GetAttr("format")},
                                                   {"out_channel", ib->GetAttr("out_channel")},
                                                   {"kernel_size", ib->GetAttr("kernel_size")},
                                                   {"input_size", MakeValue(ib->GetShape(x))},
                                                   {"mode", ib->GetAttr("mode")}})
                                       : ib->OutZeros(x);
  NodePtr dw;
  if (w->need_compute_grad_out()) {
    dw = ib->Emit("Conv3DBackpropFilter", {x, dout, w_shape},
                  {{"pad_mode", ib->GetAttr("pad_mode")},
                   {"pad", ib->GetAttr("pad")},
                   {"strides", ib->GetAttr("strides")},
                   {"dilations", ib->GetAttr("dilations")},
                   {"stride", ib->GetAttr("strides")},
                   {"dilation", ib->GetAttr("dilations")},
                   {"group", ib->GetAttr("groups")},
                   {"groups", ib->GetAttr("groups")},
                   {"format", ib->GetAttr("format")},
                   {"data_format", ib->GetAttr("format")},
                   {"out_channel", ib->GetAttr("out_channel")},
                   {"kernel_size", ib->GetAttr("kernel_size")},
                   {"filter_size", MakeValue(ib->GetShape(w))},
                   {"mode", ib->GetAttr("mode")}});
    dw = ib->Cast(dw, ib->GetDtype(x));
  } else {
    dw = ib->OutZeros(w);
  }
  return {dx, dw};
});

REG_BPROP_BUILDER("Conv3DTranspose").FreeUselessValues(FreeTensorsOfMul).SetBody(BODYFUNC(ib) {
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilations"));
  std::vector<int64_t> stride = {strides.at(i2), strides.at(i3), strides.at(i4)};
  std::vector<int64_t> dilation = {dilations.at(i2), dilations.at(i3), dilations.at(i4)};
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto w_shape = ib->Shape(w);
  auto dx = x->need_compute_grad_out() ? ib->Emit("Conv3D", {dout, w},
                                                  {{"out_channel", ib->GetAttr("in_channel")},
                                                   {"kernel_size", ib->GetAttr("kernel_size")},
                                                   {"mode", ib->GetAttr("mode")},
                                                   {"pad_mode", MakeValue("pad")},
                                                   {"pad", ib->GetAttr("pad_list")},
                                                   {"strides", ib->GetAttr("strides")},
                                                   {"dilations", ib->GetAttr("dilations")},
                                                   {"stride", MakeValue(stride)},
                                                   {"dilation", MakeValue(dilation)},
                                                   {"group", ib->GetAttr("groups")},
                                                   {"groups", ib->GetAttr("groups")},
                                                   {"offset_x", MakeValue<int64_t>(0)},
                                                   {"format", ib->GetAttr("format")},
                                                   {"data_format", ib->GetAttr("format")}})
                                       : ib->OutZeros(x);
  auto dw = w->need_compute_grad_out() ? ib->Emit("Conv3DBackpropFilter", {dout, x, w_shape},
                                                  {{"out_channel", ib->GetAttr("in_channel")},
                                                   {"kernel_size", ib->GetAttr("kernel_size")},
                                                   {"filter_size", MakeValue(ib->GetShape(w))},
                                                   {"mode", ib->GetAttr("mode")},
                                                   {"pad_mode", MakeValue("pad")},
                                                   {"pad", ib->GetAttr("pad_list")},
                                                   {"strides", ib->GetAttr("strides")},
                                                   {"dilations", ib->GetAttr("dilations")},
                                                   {"stride", ib->GetAttr("strides")},
                                                   {"dilation", ib->GetAttr("dilations")},
                                                   {"group", ib->GetAttr("groups")},
                                                   {"groups", ib->GetAttr("groups")},
                                                   {"format", ib->GetAttr("format")},
                                                   {"data_format", ib->GetAttr("format")}})
                                       : ib->OutZeros(w);
  return {dx, dw};
});

REG_BPROP_BUILDER("MaxPoolWithArgmax").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MaxPoolGradWithArgmax", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolGradGrad").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x1 = ib->GetInput(i0);
  const auto &x2 = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  auto dx1 = ib->OutZeros(x1);
  auto dx2 = ib->OutZeros(x2);
  auto dgrad = ib->Emit("MaxPoolGrad", {x1, x2, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"data_format", MakeValue("NCHW")},
                         {"format", MakeValue("NCHW")}});
  return {dx1, dx2, dgrad};
});

DEF_PURE_SHAPE_CALC(g_max_pool_grad)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x2_shape = inputs.at(0);
    auto b = x2_shape.at(i0);
    auto c = x2_shape.at(i1);
    auto h = x2_shape.at(i2);
    auto w = x2_shape.at(i3);
    return {{b}, {b, -1}, {1, c * h * w}};
  })
  .SetInfer([](const ShapeArray &, const HashSet<size_t> &) -> std::vector<int64_t> { return {1, 2, 2}; });
REG_BPROP_BUILDER("MaxPoolGrad").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto device_target = ib->GetTargetFromContext();
  auto is_ascend = device_target == "Ascend";
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  if (device_target == "CPU") {
    MS_LOG(EXCEPTION) << "MaxPoolGradGrad does not support on CPU!";
  }
  if (device_target == "GPU") {
    if ((GetValue<std::string>(ib->GetAttr("format"))) != "NCHW") {
      MS_LOG(EXCEPTION) << "MaxPoolGradGrad does not support NHWC!";
    }
    kernel_size = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
    if (kernel_size.size() == i4) {
      kernel_size = {1, kernel_size[i2], kernel_size[i3], 1};
    }
    strides = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
    if (strides.size() == i4) {
      strides = {1, strides[i2], strides[i3], 1};
    }
  }
  const auto &x1 = ib->GetInput(i0);
  const auto &x2 = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  auto dx1 = ib->OutZeros(x1);
  auto dx2 = ib->OutZeros(x2);
  NodePtr dgrad = nullptr;
  if (is_ascend) {
    dgrad = ib->Emit("MaxPoolGradGrad", {x1, x2, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"data_format", MakeValue("NCHW")},
                      {"format", MakeValue("NCHW")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  } else {
    auto tmp = ib->Emit("MaxPoolWithArgmax", {x1},
                        {{"kernel_size", MakeValue(kernel_size)},
                         {"strides", MakeValue(strides)},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"data_format", MakeValue("NCHW")},
                         {"format", MakeValue("NCHW")}});
    auto ind = ib->TupleGetItem(tmp, 1);
    auto x2_shape = ib->GetShape(x2);
    if (IsDynamic(x2_shape)) {
      auto shape = ib->Emit("Shape", {x2});
      auto res = ib->ShapeCalc(g_max_pool_grad, {x2});
      auto batch = ib->Cast(ib->Range(ib->TupleGetItem(res[i0], 0)), kInt32);
      batch = ib->Tile(ib->Reshape(batch, {-1, 1}), res[i2]);
      int64_t axis = -1;
      auto gather_ind = ib->Stack({batch, ib->Reshape(ind, res[i1])}, axis);
      dgrad = ib->Reshape(ib->GatherNd(ib->Reshape(dout, res[i1]), gather_ind), shape);
    } else {
      auto b = x2_shape.at(i0);
      auto c = x2_shape.at(i1);
      auto h = x2_shape.at(i2);
      auto w = x2_shape.at(i3);
      auto batch = ib->Tensor(Range(b), TypeIdToType(TypeId::kNumberTypeInt32));
      batch = ib->Tile(ib->Reshape(batch, {-1, 1}), {1, (c * h) * w});
      int64_t axis = -1;
      auto gather_ind = ib->Stack({batch, ib->Reshape(ind, {b, -1})}, axis);
      dgrad = ib->Reshape(ib->GatherNd(ib->Reshape(dout, {b, -1}), gather_ind), {b, c, h, w});
    }
  }
  return {dx1, dx2, dgrad};
});

REG_BPROP_BUILDER("UpsampleNearest1D").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->UpsampleNearest1DGrad(dout, x_shape, output_size, scales);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales)};
});

REG_BPROP_BUILDER("UpsampleLinear1D").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &align_corners = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->UpsampleLinear1DGrad(dout, x_shape, output_size, scales, align_corners);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales), ib->OutZeros(align_corners)};
});

REG_BPROP_BUILDER("UpsampleNearest2D").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->UpsampleNearest2DGrad(dout, x_shape, output_size, scales);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales)};
});

REG_BPROP_BUILDER("UpsampleBilinear2D").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &align_corners = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->UpsampleBilinear2DGrad(dout, x_shape, output_size, scales, align_corners);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales), ib->OutZeros(align_corners)};
});

REG_BPROP_BUILDER("UpsampleBicubic2D").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &align_corners = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->UpsampleBicubic2DGrad(dout, x_shape, output_size, scales, align_corners);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales), ib->OutZeros(align_corners)};
});

REG_BPROP_BUILDER("UpsampleNearest3D").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->UpsampleNearest3DGrad(dout, x_shape, output_size, scales);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales)};
});

REG_BPROP_BUILDER("UpsampleTrilinear3D").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_shape = ib->Shape(x);
  const auto &output_size = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &align_corners = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto dx = ib->UpsampleTrilinear3DGrad(dout, x_shape, output_size, scales, align_corners);
  return {dx, ib->OutZeros(output_size), ib->OutZeros(scales), ib->OutZeros(align_corners)};
});

REG_BPROP_BUILDER("Dropout2D").FreeUselessValues_IO({i0}, {i0}).SetBody(Dropout2DBpropExpander);
REG_BPROP_BUILDER("Dropout3D").FreeUselessValues_IO({i0}, {i0}).SetBody(Dropout2DBpropExpander);

REG_BPROP_BUILDER("CTCLoss").FreeUselessValues_IO({}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &labels_indices = ib->GetInput(i1);
  const auto &labels_values = ib->GetInput(i2);
  const auto &sequence_length = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto grad_loss = ib->TupleGetItem(out, 1);
  auto grad = ib->Mul(grad_loss, (ib->ExpandDims(ib->TupleGetItem(dout, 0), -1)));
  return {grad, ib->OutZeros(labels_indices), ib->OutZeros(labels_values), ib->OutZeros(sequence_length)};
});

REG_BPROP_BUILDER("MaxPool3D").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MaxPool3DGrad", {x, out, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPool3DGrad").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  auto dgrad = ib->Emit("MaxPool3DGradGrad", {x, y, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"format", ib->GetAttr("format")}});
  return {ib->OutZeros(x), ib->OutZeros(y), dgrad};
});

REG_BPROP_BUILDER("MaxPool3DGradGrad").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  ShapeVector pad_list(i6);
  auto dgrad = ib->Emit("MaxPool3DGrad", {x, y, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"format", ib->GetAttr("format")},
                         {"pad_list", MakeValue(pad_list)}});
  return {ib->OutZeros(x), ib->OutZeros(y), dgrad};
});

REG_BPROP_BUILDER("AvgPool").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &strides = ib->GetInput(i2);
  const auto &pad_mode = ib->GetInput(i3);
  const auto &format = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto dx = ib->Emit("AvgPoolGrad", {x, out, dout, kernel_size, strides, pad_mode, format}, {});
  return {dx, ib->OutZeros(kernel_size), ib->OutZeros(strides), ib->OutZeros(pad_mode), ib->OutZeros(format)};
});

REG_BPROP_BUILDER("AvgPool3D").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->Shape(x);
  auto dx = ib->Emit("AvgPool3DGrad", {x_shape, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"count_include_pad", ib->GetAttr("count_include_pad")},
                      {"divisor_override", ib->GetAttr("divisor_override")},
                      {"format", ib->GetAttr("format")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  return {dx};
});

REG_BPROP_BUILDER("AvgPool3DExt").SetUnusedInputs({i7}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &kernel_size = ib->GetInput(i1);
  const auto &stride = ib->GetInput(i2);
  const auto &padding = ib->GetInput(i3);
  const auto &ceil_mode = ib->GetInput(i4);
  const auto &count_include_pad = ib->GetInput(i5);
  const auto &divisor_override = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);
  auto dx =
    ib->AvgPool3DGradExt(dout, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return {dx,
          ib->OutZeros(kernel_size),
          ib->OutZeros(stride),
          ib->OutZeros(padding),
          ib->OutZeros(ceil_mode),
          ib->OutZeros(count_include_pad),
          ib->OutZeros(divisor_override)};
});

REG_BPROP_BUILDER("Mish").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx1 = ib->Tanh(ib->Emit("Softplus", {x}));
  auto dx2 = ib->Emit("SoftplusGrad", {ib->TanhGrad(dx1, ib->Mul(x, dout)), x});
  auto dx = ib->Add((ib->Mul(dx1, dout)), dx2);
  return {dx};
});

REG_BPROP_BUILDER("MishExt").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->MishGradExt(dout, x);
  return {dx};
});

REG_BPROP_BUILDER("SeLU").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto scale = 1.0507009873554805;
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto tmp_grad = ib->Emit("EluGrad", {dout, out});
  auto dx = ib->Muls(tmp_grad, ib->ValueByType(scale, ib->GetDtype(tmp_grad)));
  return {dx};
});

REG_BPROP_BUILDER("SeLUExt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->SeluGrad(dout, out);
  return {dx};
});

REG_BPROP_BUILDER("Swiglu").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->SwigluGrad(dout, input_x, dim);
  return {dx, ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("ReLU6").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("ReLU6Grad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("Hardtanh").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &min_val = ib->GetInput(i1);
  const auto &max_val = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->HardtanhGrad(dout, x, min_val, max_val);
  return {dx, ib->OutZeros(min_val), ib->OutZeros(max_val)};
});

REG_BPROP_BUILDER("InplaceHardtanh").CloneInplaceInput().SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &min_val = ib->GetInput(i1);
  const auto &max_val = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->HardtanhGrad(dout, x, min_val, max_val);
  return {dx, ib->OutZeros(min_val), ib->OutZeros(max_val)};
});

REG_BPROP_BUILDER("BiasAddGrad").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &dy = ib->GetInput(i0);
  const auto &format = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto format_data = format->BuildValue();
  MS_EXCEPTION_IF_CHECK_FAIL(format_data != nullptr, "The input format of 'BiasAddGrad' must be constant.");
  auto res = ib->ShapeCalc(std::make_shared<BiasAddGradShapeCalc>(GetValue<int64_t>(format_data)), {dy, dout});
  NodePtr expanded_shape = res[0];
  NodePtr tile_mults = res[1];

  auto expanded_grad = ib->Reshape(dout, expanded_shape);
  auto tiled_grad = ib->Tile(expanded_grad, tile_mults);
  return {tiled_grad, ib->OutZeros(format)};
});

REG_BPROP_BUILDER("ExtractImagePatches").SetUnusedInputs({i0, i5}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &ksizes = ib->GetInput(i1);
  const auto &strides = ib->GetInput(i2);
  const auto &rates = ib->GetInput(i3);
  const auto &padding = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);

  auto x_shape = ib->GetShape(x);
  auto out_shape = ib->GetShape(out);

  auto ksizes_opt = GetArrayValue<int64_t>(ksizes->abstract());
  int64_t ksizes_row = 1;
  int64_t ksizes_col = 1;
  if (ksizes_opt.has_value()) {
    ksizes_row = ksizes_opt.value()[0];
    ksizes_col = ksizes_opt.value()[1];
  } else {
    MS_LOG(EXCEPTION) << "For ExtractImagePatches bprop, get 'ksize' data failed.";
  }

  if (IsDynamic(x_shape) || IsDynamic(out_shape)) {
    auto res = ib->ShapeCalc(std::make_shared<ExtractImagePatchesShapeCalc>(ksizes_row, ksizes_col), {x, out});
    auto x_idx =
      ib->Cast(ib->Range(ib->Value<int64_t>(1), ib->TupleGetItem(res[0], 0), ib->Value<int64_t>(1)), kFloat32);
    x_idx = ib->Reshape(x_idx, res[1]);
    auto x_idx_patch = ib->Cast(ib->Emit("ExtractImagePatches", {x_idx, ksizes, strides, rates, padding}), kInt32);
    x_idx_patch = ib->Transpose(x_idx_patch, {0, 2, 3, 1});
    auto out_idx = ib->Cast(ib->Range(ib->TupleGetItem(res[2], 0)), kInt32);
    out_idx = ib->Reshape(out_idx, res[3]);
    auto idx_tensor = ib->Concat({ib->ExpandDims(x_idx_patch, -1), ib->ExpandDims(out_idx, -1)}, -1);
    idx_tensor = ib->Reshape(idx_tensor, {-1, 2});
    auto ones = ib->Fill(1.0, res[2], ib->GetDtype(dout)->type_id());
    auto sp_tensor = ib->ScatterNd(idx_tensor, ones, res[4]);
    sp_tensor = ib->Slice(sp_tensor, ib->Value<ShapeVector>({1, 0}), res[5]);
    auto grad = ib->Transpose(dout, {0, 2, 3, 1});
    grad = ib->Reshape(grad, res[6]);
    grad = ib->Transpose(grad, {1, 2, 3, 4, 0, 5});
    grad = ib->Reshape(grad, res[7]);
    auto jac = ib->MatMul(sp_tensor, grad, false, false);
    auto dx = ib->Reshape(jac, res[8]);
    dx = ib->Transpose(dx, {2, 3, 0, 1});
    return {dx, ib->OutZeros(ksizes), ib->OutZeros(strides), ib->OutZeros(rates), ib->OutZeros(padding)};
  } else {
    auto x_batch = x_shape[0];
    auto x_depth = x_shape[1];
    auto x_row = x_shape[2];
    auto x_col = x_shape[3];
    auto x_indices_num = (x_row * x_col) + 1;
    auto x_idx = ib->Tensor(Range(1, x_indices_num), kFloat32);
    x_idx = ib->Reshape(x_idx, {1, 1, x_row, x_col});
    auto x_idx_patch = ib->Cast(ib->Emit("ExtractImagePatches", {x_idx, ksizes, strides, rates, padding}), kInt32);
    x_idx_patch = ib->Transpose(x_idx_patch, {0, 2, 3, 1});
    auto out_row = out_shape[2];
    auto out_col = out_shape[3];
    auto out_indices_num = ((out_row * out_col) * ksizes_row) * ksizes_col;
    auto out_idx = ib->Tensor(Range(out_indices_num), kInt32);
    out_idx = ib->Reshape(out_idx, {1, out_row, out_col, ksizes_row * ksizes_col});
    auto idx_tensor = ib->Concat({ib->ExpandDims(x_idx_patch, -1), ib->ExpandDims(out_idx, -1)}, -1);
    idx_tensor = ib->Reshape(idx_tensor, {-1, 2});
    std::vector<int64_t> sp_shape = {x_indices_num, out_indices_num};
    std::vector<int64_t> ones(out_indices_num, 1);
    auto sp_tensor = ib->ScatterNd(idx_tensor, ib->Tensor(ones, ib->GetDtype(dout)), ib->Value<ShapeVector>(sp_shape));
    sp_tensor = ib->Slice(sp_tensor, ib->Value<ShapeVector>({1, 0}),
                          ib->Value<ShapeVector>({x_indices_num - 1, out_indices_num}));
    auto grad = ib->Transpose(dout, {0, 2, 3, 1});
    grad = ib->Reshape(grad, {x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth});
    grad = ib->Transpose(grad, {1, 2, 3, 4, 0, 5});
    grad = ib->Reshape(grad, {-1, x_batch * x_depth});
    auto jac = ib->MatMul(sp_tensor, grad, false, false);
    auto dx = ib->Reshape(jac, {x_row, x_col, x_batch, x_depth});
    dx = ib->Transpose(dx, {2, 3, 0, 1});
    return {dx, ib->OutZeros(ksizes), ib->OutZeros(strides), ib->OutZeros(rates), ib->OutZeros(padding)};
  }
});

REG_BPROP_BUILDER("HSwish").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->HSwishGrad(dout, x);
  return {dx};
});

REG_BPROP_BUILDER("HSigmoid").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->HSigmoidGrad(dout, x);
  return {dx};
});

REG_BPROP_BUILDER("Elu").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &alpha = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("EluGrad", {dout, out});
  return {dx, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("EluExt").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &alpha = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->EluGradExt(dout, x, alpha, ib->Value<bool>(false));
  return {dx, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceElu").FreeUselessValues_I({i0}).SetBody(BODYFUNC(ib) {
  const auto &alpha = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->EluGradExt(dout, out, alpha, ib->Value<bool>(true));
  return {dx, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("InplaceSigmoid").FreeUselessValues_I({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->SigmoidGrad(out, dout);
  return {dx};
});

REG_BPROP_BUILDER("Sigmoid").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->SigmoidGrad(out, dout);
  return {dx};
});

REG_BPROP_BUILDER("SigmoidGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dy =
    y->need_compute_grad_out()
      ? ib->Mul((ib->Mul(dout, grad)),
                (ib->Sub(ib->Tensor(1, ib->GetDtype(grad)), (ib->Muls(y, ib->ValueByType(2, ib->GetDtype(y)))))))
      : ib->OutZeros(y);
  auto dgrad = grad->need_compute_grad_out() ? ib->SigmoidGrad(y, dout) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("LogSigmoid").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto buffer = ib->TupleGetItem(out, i1);
  auto dy = ib->TupleGetItem(dout, i0);
  return {ib->LogSigmoidGrad(dy, input, buffer)};
});

REG_BPROP_BUILDER("LogSoftmax").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &axis = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->LogSoftmaxGrad(out, dout, axis);
  return {dx, ib->OutZeros(axis)};
});

DEF_PURE_SHAPE_CALC(g_log_softmax_ext_shape)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(i0);
    size_t ndim = x_shape.size();
    int64_t ret;
    if (ndim == 0 || ndim == 1 || ndim == 3) {
      ret = 0;
    } else {
      ret = 1;
    }
    return {{ret}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto shape_out = inputs.at(i0);
    if (IsDynamicRank(shape_out)) {
      return {-1};
    }
    return {1};
  });

REG_BPROP_BUILDER("LogSoftmaxExt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &dim = ib->GetInput(i1);
  const auto &dtype = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto new_dim = dim;
  if (ib->GetDtype(dim)->isa<TypeNone>()) {
    new_dim = ib->ShapeCalc(g_log_softmax_ext_shape, {input})[0];
    new_dim = ib->TupleGetItem(new_dim, 0);
  }
  auto dx = ib->LogSoftmaxGrad(out, dout, new_dim);
  if (ib->GetDtype(input) != ib->GetDtype(dx)) {
    dx = ib->Cast(dx, ib->GetDtype(input));
  }
  return {dx, ib->OutZeros(new_dim), ib->OutZeros(dtype)};
});

REG_BPROP_BUILDER("Softplus").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("SoftplusGrad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("SoftplusExt").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &beta = ib->GetInput(i1);
  const auto &threshold = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->SoftplusGradExt(dout, x, beta, threshold);
  return {dx, ib->OutZeros(beta), ib->OutZeros(threshold)};
});

REG_BPROP_BUILDER("SoftplusGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &dy = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto ddy = dy->need_compute_grad_out() ? ib->Emit("SoftplusGrad", {dout, x}) : ib->OutZeros(dy);
  auto d2x =
    x->need_compute_grad_out()
      ? ib->Div(ib->Mul(dout, dy),
                ib->Add(ib->AddScalar(ib->Exp(x), ib->ValueByType(kConstNumberTwo, ib->GetDtype(dy)), ib->Value(1)),
                        ib->Exp(-x)))
      : ib->OutZeros(x);
  return {ddy, d2x};
});

REG_BPROP_BUILDER("Softsign").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto dx =
    ib->Mul(dout, ib->Div(ib->Tensor(1, ib->GetDtype(x)),
                          ib->Square(ib->AddScalar(ib->Abs(x), ib->ValueByType(1, ib->GetDtype(x)), ib->Value(1)))));
  return {dx};
});

REG_BPROP_BUILDER("Tanh").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    dout = ib->Conj(dout);
    dx = ib->TanhGrad(out, dout);
    dx = ib->Conj(dx);
  } else {
    dx = ib->TanhGrad(out, dout);
  }
  return {dx};
});

REG_BPROP_BUILDER("InplaceTanh").FreeUselessValues_I({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    dout = ib->Conj(dout);
    dx = ib->TanhGrad(out, dout);
    dx = ib->Conj(dx);
  } else {
    dx = ib->TanhGrad(out, dout);
  }
  return {dx};
});

REG_BPROP_BUILDER("TanhGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i0);
  const auto &grad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dy = y->need_compute_grad_out()
              ? ib->Mul((ib->Mul((ib->Muls(dout, ib->ValueByType(-2.0, ib->GetDtype(dout)))), grad)), y)
              : ib->OutZeros(y);
  auto dgrad = grad->need_compute_grad_out() ? ib->TanhGrad(y, dout) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("GeLU").SetBody(GeLUBpropExpander);
REG_BPROP_BUILDER("Gelu").SetBody(GeLUBpropExpander);

REG_BPROP_BUILDER("GeluExt").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  const auto &approximate = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dinput = ib->GeluGradExt(dout, input, approximate);
  return {dinput, ib->OutZeros(approximate)};
});

REG_BPROP_BUILDER("FastGeLU").SetUnusedInputs({i1}).SetBody(FastGeLUBpropExpander);
REG_BPROP_BUILDER("FastGelu").SetUnusedInputs({i1}).SetBody(FastGeLUBpropExpander);

REG_BPROP_BUILDER("InstanceNorm").FreeUselessValues_IO({i2, i3, i4}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &gamma = ib->GetInput(i1);
  const auto &mean = ib->GetInput(i3);
  const auto &variance = ib->GetInput(i4);
  auto out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto saved_mean = ib->TupleGetItem(out, 1);
  auto saved_variance = ib->TupleGetItem(out, 2);
  out = ib->Emit("InstanceNormGrad", {ib->TupleGetItem(dout, 0), x, gamma, saved_mean, saved_variance},
                 {{"epsilon", ib->GetAttr("epsilon")}, {"momentum", ib->GetAttr("momentum")}});
  auto dx = ib->TupleGetItem(out, 0);
  auto dgamma = ib->TupleGetItem(out, 1);
  auto dbeta = ib->TupleGetItem(out, 2);
  return {dx, dgamma, dbeta, ib->OutZeros(mean), ib->OutZeros(variance)};
});

REG_BPROP_BUILDER("BatchNormExt").FreeUselessValues_IO({i2}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &weight = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &running_mean = ib->GetInput(i3);
  const auto &running_var = ib->GetInput(i4);
  const auto &training = ib->GetInput(i5);
  const auto &eps = ib->GetInput(i7);
  const auto &momentum = ib->GetInput(i6);
  const auto &out = ib->GetInput(i8);
  const auto &dout = ib->GetInput(i9);
  auto is_training_value_ptr = training->BuildValue();
  std::vector<int64_t> output_mask_vec = {x->need_compute_grad_out(), weight->need_compute_grad_out(),
                                          bias->need_compute_grad_out()};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto result = ib->BatchNormGradExt(ib->TupleGetItem(dout, 0), x, weight, running_mean, running_var,
                                     ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2), training, eps, output_mask);
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_weight = ib->TupleGetItem(result, 1);
  auto d_bias = ib->TupleGetItem(result, 2);

  bool weight_type_none = ib->GetDtype(weight)->isa<TypeNone>();
  bool bias_type_none = ib->GetDtype(bias)->isa<TypeNone>();

  if (!weight_type_none && weight->need_compute_grad_out() && ib->GetDtype(d_weight) != ib->GetDtype(weight)) {
    d_weight = ib->Cast(d_weight, ib->GetDtype(weight));
  }
  if (!bias_type_none && bias->need_compute_grad_out() && ib->GetDtype(d_bias) != ib->GetDtype(bias)) {
    d_bias = ib->Cast(d_bias, ib->GetDtype(bias));
  }
  return {d_x,
          d_weight,
          d_bias,
          ib->OutZeros(running_mean),
          ib->OutZeros(running_var),
          ib->OutZeros(training),
          ib->OutZeros(momentum),
          ib->OutZeros(eps)};
});

REG_BPROP_BUILDER("BatchNorm").FreeUselessValues_IO({i2}, {i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &scale = ib->GetInput(i1);
  const auto &bias = ib->GetInput(i2);
  const auto &mean = ib->GetInput(i3);
  const auto &variance = ib->GetInput(i4);
  const auto &is_training = ib->GetInput(i5);
  const auto &epsilon = ib->GetInput(i6);
  const auto &momentum = ib->GetInput(i7);
  const auto &data_format = ib->GetInput(i8);
  const auto &out = ib->GetInput(i9);
  const auto &dout = ib->GetInput(i10);

  NodePtr saved_mean{nullptr};
  NodePtr saved_variance{nullptr};
  auto is_training_value_ptr = is_training->BuildValue();
  auto training_value_opt = GetScalarValue<bool>(is_training_value_ptr);
  if (training_value_opt.has_value()) {
    if (training_value_opt.value()) {
      saved_mean = ib->TupleGetItem(out, 3);
      saved_variance = ib->TupleGetItem(out, 4);
    } else {
      saved_mean = mean;
      saved_variance = variance;
    }
  } else {
    auto cond_out = ib->Conditional(
      is_training, [&out](Emitter *e) -> NodePtrList { return {e->TupleGetItem(out, 3), e->TupleGetItem(out, 4)}; },
      [&mean, &variance](Emitter *e) -> NodePtrList { return {mean, variance}; });
    saved_mean = ib->TupleGetItem(cond_out, 0);
    saved_variance = ib->TupleGetItem(cond_out, 1);
  }
  auto reserve = ib->TupleGetItem(out, 2);
  bool is_scale_or_bias_grad = (scale->need_compute_grad_out() || bias->need_compute_grad_out());
  auto new_out = ib->BatchNormGrad(
    {ib->TupleGetItem(dout, 0), x, scale, saved_mean, saved_variance, reserve, is_training, epsilon, data_format},
    is_scale_or_bias_grad);
  auto dx = ib->TupleGetItem(new_out, 0);
  auto dscale = ib->TupleGetItem(new_out, 1);
  auto dbias = ib->TupleGetItem(new_out, 2);
  return {dx,
          dscale,
          dbias,
          ib->OutZeros(mean),
          ib->OutZeros(variance),
          ib->OutZeros(is_training),
          ib->OutZeros(epsilon),
          ib->OutZeros(momentum),
          ib->OutZeros(data_format)};
});

DEF_PURE_SHAPE_CALC(moe_token_permute_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto indices_shape = inputs.at(i0);
    auto indices_shape_rank = indices_shape.size();
    auto num_topk = 1;
    if (indices_shape_rank == i2) {
      num_topk = indices_shape[i1];
    }
    return {indices_shape, {num_topk}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto indices_shape = inputs.at(i0);
    if (!unknown_inputs.empty() || IsDynamicRank(indices_shape)) {
      return {-1, 1};
    }
    auto size = SizeToLong(indices_shape.size());
    return {size, 1};
  });

REG_BPROP_BUILDER("MoeTokenPermute").FreeUselessValues_IO({i0, i1, i2}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i1);
  const auto &num_out_tokens = ib->GetInput(i2);
  auto padded_mode = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto sorted_indices = ib->TupleGetItem(out, i1);
  auto d_permuted_tokens = ib->TupleGetItem(dout, 0);
  auto indices_shape = ib->GetShape(indices);
  auto padded_mode_type = padded_mode->abstract()->BuildType();
  NodePtr res_grad;
  padded_mode = padded_mode_type->isa<TypeNone>() ? ib->Value<bool>(false) : padded_mode;
  NodePtr num_topk = ib->Value<int64_t>(1);
  num_topk = ib->TupleGetItem(ib->ShapeCalc(moe_token_permute_shapecalc, {indices})[i1], i0);
  res_grad = ib->MoeTokenPermuteGrad(d_permuted_tokens, sorted_indices, num_topk, padded_mode);
  return {res_grad, ib->OutZeros(indices), ib->OutZeros(num_out_tokens), ib->OutZeros(padded_mode)};
});

REG_BPROP_BUILDER("CrossEntropyLoss").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &target = ib->GetInput(kIndex1);
  const auto &weight = ib->GetInput(kIndex2);
  const auto &reduction = ib->GetInput(kIndex3);
  const auto &ignore_index = ib->GetInput(kIndex4);
  const auto &label_smoothing = ib->GetInput(kIndex5);
  const auto &lse_square_scale_for_zloss = ib->GetInput(kIndex6);
  const auto &return_zloss = ib->GetInput(kIndex7);
  const auto &out = ib->GetInput(kIndex8);
  const auto &dout = ib->GetInput(kIndex9);
  auto grad_loss = ib->TupleGetItem(dout, kIndex0);
  auto grad_zloss = ib->TupleGetItem(dout, kIndex2);
  auto log_prob = ib->TupleGetItem(out, kIndex1);
  auto lse_for_zloss = ib->TupleGetItem(out, kIndex3);

  auto input_grad = ib->CrossEntropyLossGrad(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction,
                                             ignore_index, label_smoothing, lse_square_scale_for_zloss);
  return {input_grad,
          ib->OutZeros(target),
          ib->OutZeros(weight),
          ib->OutZeros(reduction),
          ib->OutZeros(ignore_index),
          ib->OutZeros(label_smoothing),
          ib->OutZeros(lse_square_scale_for_zloss),
          ib->OutZeros(return_zloss)};
});

REG_BPROP_BUILDER("BatchNormGradExt").SetUnusedInputs({i3, i4, i9}).SetBody(BODYFUNC(ib) {
  const auto &dy = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &running_mean = ib->GetInput(i3);
  const auto &running_var = ib->GetInput(i4);
  const auto &saved_mean = ib->GetInput(i5);
  const auto &saved_rstd = ib->GetInput(i6);
  const auto &training = ib->GetInput(i7);
  const auto &eps = ib->GetInput(i8);
  const auto &dout = ib->GetInput(i10);
  auto format = ib->EmitValue(MakeValue<int64_t>(Format::NCHW));

  NodePtr mean{nullptr};
  NodePtr var{nullptr};
  auto training_value_ptr = training->BuildValue();
  auto training_value_opt = GetScalarValue<bool>(training_value_ptr);
  if (training_value_opt.has_value()) {
    if (training_value_opt.value()) {
      mean = saved_mean;
      var = saved_rstd;
    } else {
      mean = running_mean;
      var = running_var;
    }
  } else {
    auto cond_out = ib->Conditional(
      training, [&saved_mean, &saved_rstd](Emitter *e) -> NodePtrList { return {saved_mean, saved_rstd}; },
      [&running_mean, &running_var](Emitter *e) -> NodePtrList { return {running_mean, running_var}; });
    mean = ib->TupleGetItem(cond_out, 0);
    var = ib->TupleGetItem(cond_out, 1);
  }

  auto tmp =
    ib->Emit("BatchNormGradGrad", {x, dy, weight, mean, var, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1),
                                   ib->TupleGetItem(dout, 2), training, eps, format});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto ddy = ib->TupleGetItem(tmp, 1);
  auto dweight = ib->TupleGetItem(tmp, 2);
  return {ddy,
          dx,
          dweight,
          ib->OutZeros(running_mean),
          ib->OutZeros(running_var),
          ib->OutZeros(saved_mean),
          ib->OutZeros(saved_rstd),
          ib->OutZeros(training),
          ib->OutZeros(eps)};
});

REG_BPROP_BUILDER("BatchNormGrad").SetUnusedInputs({i5, i9}).SetBody(BODYFUNC(ib) {
  const auto &dy = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &scale = ib->GetInput(i2);
  const auto &mean = ib->GetInput(i3);
  const auto &variance = ib->GetInput(i4);
  const auto &reserve = ib->GetInput(i5);
  const auto &is_training = ib->GetInput(i6);
  const auto &epsilon = ib->GetInput(i7);
  const auto &data_format = ib->GetInput(i8);
  const auto &dout = ib->GetInput(i10);
  auto tmp =
    ib->Emit("BatchNormGradGrad", {x, dy, scale, mean, variance, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1),
                                   ib->TupleGetItem(dout, 2), is_training, epsilon, data_format});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto ddy = ib->TupleGetItem(tmp, 1);
  auto dscale = ib->TupleGetItem(tmp, 2);
  return {ddy,
          dx,
          dscale,
          ib->OutZeros(mean),
          ib->OutZeros(variance),
          ib->OutZeros(reserve),
          ib->OutZeros(is_training),
          ib->OutZeros(epsilon),
          ib->OutZeros(data_format)};
});

REG_BPROP_BUILDER("Softmax").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &axis = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dim = ib->TupleGetItem(axis, 0);
  auto dx = ib->SoftmaxBackward(dout, out, dim);
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("SoftmaxBackward").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &grad_output = ib->GetInput(i0);
  const auto &output = ib->GetInput(i1);
  const auto &dim = ib->GetInput(i2);
  const auto &grad = ib->GetInput(i4);

  NodePtr grad_dout{nullptr};
  if (grad_output->need_compute_grad_out()) {
    grad_dout = ib->SoftmaxBackward(grad, output, dim);
  } else {
    grad_dout = ib->OutZeros(grad_output);
  }

  // grad_out = grad_output * grad - (output * grad_output).sum(dim, true) * grad -
  // grad_output * (output * grad).sum(dim, true)
  auto softmax_double_backward_func = [&]() -> NodePtr {
    auto dims = ib->MakeTuple({dim});
    auto part1 = ib->Mul(grad_output, grad);
    auto part2 = ib->Mul(ib->ReduceSum(ib->Mul(output, grad_output), dims, true), grad);
    auto part3 = ib->Mul(grad_output, ib->ReduceSum(ib->Mul(output, grad), dims, true));
    auto grad_out = part1 - part2 - part3;
    return grad_out;
  };
  NodePtr grad_out{nullptr};
  if (output->need_compute_grad_out()) {
    grad_out = softmax_double_backward_func();
  } else {
    grad_out = ib->OutZeros(output);
  }

  return {grad_dout, grad_out, ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("SparseSoftmaxCrossEntropyWithLogits").SetBody(BODYFUNC(ib) {
  auto is_grad = ib->GetAttr<bool>(kAttrIsGrad);
  const auto &labels = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  if (is_grad) {
    return {ib->TensorGetItem(out, 0), ib->OutZeros(labels)};
  }
  // is_grad is false
  const auto &logits = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i3);
  auto grad = ib->SparseSoftmaxCrossEntropyWithLogits({logits, labels}, {{kAttrIsGrad, MakeValue(true)}}, out, dout,
                                                      ib->IsGraphMode());
  return {grad, ib->OutZeros(labels)};
});

REG_BPROP_BUILDER("DynamicRNN").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &w = ib->GetInput(i1);
  const auto &b = ib->GetInput(i2);
  const auto &init_h = ib->GetInput(i4);
  const auto &init_c = ib->GetInput(i5);
  const auto &out = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i7);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dh = ib->TupleGetItem(dout, i1);
  auto dc = ib->TupleGetItem(dout, i2);
  dh = ib->TensorGetItem(dh, -1);
  dc = ib->TensorGetItem(dc, -1);
  auto y = ib->TupleGetItem(out, i0);
  auto h = ib->TupleGetItem(out, i1);
  auto c = ib->TupleGetItem(out, i2);
  auto i = ib->TupleGetItem(out, i3);
  auto j = ib->TupleGetItem(out, i4);
  auto f = ib->TupleGetItem(out, i5);
  auto o = ib->TupleGetItem(out, i6);
  auto tanhct = ib->TupleGetItem(out, i7);
  auto tmp = ib->Emit(
    "DynamicRNNGrad",
    {x, w, b, y, ib->TensorGetItem(init_h, 0), ib->TensorGetItem(init_c, 0), h, c, dy, dh, dc, i, j, f, o, tanhct},
    {{"cell_type", ib->GetAttr("cell_type")},
     {"direction", ib->GetAttr("direction")},
     {"cell_depth", ib->GetAttr("cell_depth")},
     {"use_peephole", ib->GetAttr("use_peephole")},
     {"keep_prob", ib->GetAttr("keep_prob")},
     {"cell_clip", ib->GetAttr("cell_clip")},
     {"num_proj", ib->GetAttr("num_proj")},
     {"time_major", ib->GetAttr("time_major")},
     {"forget_bias", ib->GetAttr("forget_bias")}});
  auto dw = ib->TupleGetItem(tmp, i0);
  auto db = ib->TupleGetItem(tmp, i1);
  auto dx = ib->TupleGetItem(tmp, i2);
  auto dh_prev = ib->TupleGetItem(tmp, i3);
  auto dc_prev = ib->TupleGetItem(tmp, i4);
  dh_prev = ib->ExpandDims(dh_prev, 0);
  dc_prev = ib->ExpandDims(dc_prev, 0);
  constexpr int64_t zero = 0;
  return {dx, dw, db, ib->OutZeros(ib->Tensor(zero)), dh_prev, dc_prev};
});

REG_BPROP_BUILDER("GRUV2").FreeUselessValues_O({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &hx = ib->GetInput(i1);
  const auto &w = ib->GetInput(i2);
  const auto &seq_length = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto y = ib->TupleGetItem(out, i0);
  auto hy = ib->TupleGetItem(out, i1);
  auto reverse = ib->TupleGetItem(out, i2);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dhy = ib->TupleGetItem(dout, i1);
  auto tmp = ib->Emit("GRUV2Grad", {x, hx, w, seq_length, y, hy, dy, dhy, reverse},
                      {{"input_size", ib->GetAttr("input_size")},
                       {"hidden_size", ib->GetAttr("hidden_size")},
                       {"num_layers", ib->GetAttr("num_layers")},
                       {"has_bias", ib->GetAttr("has_bias")},
                       {"bidirectional", ib->GetAttr("bidirectional")},
                       {"dropout", ib->GetAttr("dropout")}});
  auto dx = ib->TupleGetItem(tmp, i0);
  auto dhx = ib->TupleGetItem(tmp, i1);
  auto dw = ib->TupleGetItem(tmp, i2);
  return {dx, dhx, dw, ib->OutZeros(seq_length)};
});

REG_BPROP_BUILDER("DynamicGRUV2").SetUnusedInputs({i3, i4, i5}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &winput = ib->GetInput(i1);
  const auto &whidden = ib->GetInput(i2);
  const auto &init_h = ib->GetInput(i6);
  const auto &out = ib->GetInput(i7);
  const auto &dout = ib->GetInput(i8);
  auto y = ib->TupleGetItem(out, i0);
  auto out_h = ib->TupleGetItem(out, i1);
  auto update = ib->TupleGetItem(out, i2);
  auto reset = ib->TupleGetItem(out, i3);
  auto new_t = ib->TupleGetItem(out, i4);
  auto hidden_new = ib->TupleGetItem(out, i5);
  auto dy = ib->TupleGetItem(dout, i0);
  auto dout_h = ib->TupleGetItem(dout, i1);
  auto tmp = ib->Emit("DynamicGRUV2Grad",
                      {x, winput, whidden, y, init_h, out_h, dy, ib->TensorGetItem(dout_h, -1), update, reset, new_t,
                       hidden_new, ib->EmitValue(kNone), ib->EmitValue(kNone)},
                      {{"direction", ib->GetAttr("direction")},
                       {"cell_depth", ib->GetAttr("cell_depth")},
                       {"keep_prob", ib->GetAttr("keep_prob")},
                       {"cell_clip", ib->GetAttr("cell_clip")},
                       {"num_proj", ib->GetAttr("num_proj")},
                       {"time_major", ib->GetAttr("time_major")},
                       {"gate_order", ib->GetAttr("gate_order")},
                       {"reset_after", ib->GetAttr("reset_after")}});
  auto dw_input = ib->TupleGetItem(tmp, i0);
  auto dw_hidden = ib->TupleGetItem(tmp, i1);
  auto db_input = ib->TupleGetItem(tmp, i2);
  auto db_hidden = ib->TupleGetItem(tmp, i3);
  auto dx = ib->TupleGetItem(tmp, i4);
  auto dh_prev = ib->TupleGetItem(tmp, i5);
  constexpr int64_t zero = 0;
  return {dx, dw_input, dw_hidden, db_input, db_hidden, ib->OutZeros(ib->Tensor(zero)), dh_prev};
});

REG_BPROP_BUILDER("AdaptiveMaxPool2D").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto index = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit("AdaptiveMaxPool2DGrad", {dy, x, index});
  return {dx, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("AdaptiveMaxPool3D").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto index = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit("AdaptiveMaxPool3DGrad", {dy, x, index});
  return {dx, ib->ZerosLikeExt(output_size, ib->EmitValue(kNone))};
});

REG_BPROP_BUILDER("Conv2DTranspose").SetUnusedInputs({i2, i3}).SetBody(Conv2DTransposeBpropExpander);
REG_BPROP_BUILDER("Conv2DBackpropInput").SetUnusedInputs({i2, i3}).SetBody(Conv2DTransposeBpropExpander);

REG_BPROP_BUILDER("Conv2DBackpropFilter").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &dy = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  const auto &filter_size = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto x_shape = ib->Shape(x);
  auto dw_dy = dy->need_compute_grad_out() ? ib->Emit(kConv2DOpName, {x, dout}, Conv2DAttrs(ib)) : ib->OutZeros(dy);
  auto dw_dx = x->need_compute_grad_out()
                 ? ib->Emit(kConv2DBackpropInputOpName, {dy, dout, x_shape}, Conv2DBackpropAttrs(ib))
                 : ib->OutZeros(x);
  return {dw_dy, dw_dx, ib->OutZeros(filter_size)};
});

REG_BPROP_BUILDER("BCEWithLogitsLoss").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  // input, target, weight, posWeight, reduction, out, dout
  const auto &dout = ib->GetInput(i6);
  const auto &input = ib->GetInput(i0);
  auto target = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &posweight = ib->GetInput(i3);
  const auto &reduction = ib->GetInput(i4);
  bool posweight_type_none = ib->GetDtype(posweight)->isa<TypeNone>();
  bool weight_type_none = ib->GetDtype(weight)->isa<TypeNone>();

  NodePtr grad_input = nullptr;
  if (input->need_compute_grad_out()) {
    if (ib->GetDtype(input) != ib->GetDtype(target)) {
      MS_LOG(DEBUG) << "For 'BinaryCrossEntropyWithLogitsBackward', cast 'input' dtype to 'target' dtype, input: "
                    << input->ToString() << ", target: " << target->ToString();
      target = ib->Cast(target, ib->GetDtype(input));
    }
    grad_input = ib->BinaryCrossEntropyWithLogitsBackward(dout, input, target, weight, posweight, reduction);
  } else {
    grad_input = ib->OutZeros(input);
  }

  NodePtr grad_target = nullptr;
  if (target->need_compute_grad_out()) {
    if (!posweight_type_none) {
      auto sigmoid_input = ib->Sigmoid(input);
      grad_target = ib->Mul(ib->Sub(ib->Log(ib->Sub(ib->Tensor(1, ib->GetDtype(sigmoid_input)), sigmoid_input)),
                                    ib->Mul(posweight, ib->Log(sigmoid_input))),
                            dout);
    } else {
      grad_target = ib->Mul(input, ib->Neg(dout));
    }

    if (!weight_type_none) {
      grad_target = ib->Mul(grad_target, weight);
    }

    auto reduction_value = reduction->BuildValue();
    auto reduction_int_value = GetScalarValue<int64_t>(reduction_value);
    if (reduction_int_value == Reduction::MEAN) {
      if (IsDynamic(ib->GetShape(grad_input))) {
        auto res2 = ib->DynSize(target, ib->GetDtype(grad_target));
        grad_target = ib->Div(grad_target, res2);
      } else {
        grad_target = ib->Divs(grad_target, ib->ValueByType(ib->GetSize(target), ib->GetDtype(grad_target)));
      }
    }
  } else {
    grad_target = ib->OutZeros(target);
  }

  return {grad_input, grad_target, ib->OutZeros(weight), ib->OutZeros(posweight), ib->OutZeros(reduction)};
});

REG_BPROP_BUILDER("KLDivLoss").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto reduction = GetValue<std::string>(ib->GetAttr("reduction"));
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx;
  if (reduction == "mean") {
    dx = ib->Emit("KLDivLossGrad", {dout, x, y}, {{"reduction", MakeValue("sum")}});
    if (IsDynamic(ib->GetShape(x))) {
      auto res = ib->DynSize(dx, ib->GetDtype(dx));
      dx = ib->RealDiv(dx, res);
    } else {
      dx = ib->RealDiv(dx, ib->Tensor(ib->GetSize(x), ib->GetDtype(dx)));
    }
  } else {
    dx = ib->Emit("KLDivLossGrad", {dout, x, y}, {{"reduction", MakeValue(reduction)}});
  }
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("HShrink").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &features = ib->GetInput(i0);
  const auto &lambd = ib->GetInput(i1);
  const auto &gradients = ib->GetInput(i3);
  auto dx = ib->HShrinkGrad(gradients, features, lambd);
  return {dx, ib->OutZeros(lambd)};
});

REG_BPROP_BUILDER("SoftShrink").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &lambd = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->SoftShrinkGrad(dout, input_x, lambd);
  return {dx, ib->OutZeros(lambd)};
});

REG_BPROP_BUILDER("SoftMarginLoss").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &predict = ib->GetInput(i0);
  const auto &label = ib->GetInput(i1);
  const auto &reduction = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx =
    predict->need_compute_grad_out() ? ib->SoftMarginLossGrad(predict, label, dout, reduction) : ib->OutZeros(predict);
  auto dy =
    label->need_compute_grad_out() ? ib->SoftMarginLossGrad(label, predict, dout, reduction) : ib->OutZeros(label);
  return {dx, dy, ib->OutZeros(reduction)};
});

REG_BPROP_BUILDER("MultilabelMarginLoss").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &target = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("MultilabelMarginLossGrad", {ib->TupleGetItem(dout, 0), x, target, ib->TupleGetItem(out, 1)},
                     {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->OutZeros(target)};
});

REG_BPROP_BUILDER("Dilation2D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &_filter = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = x->need_compute_grad_out() ? ib->Emit("Dilation2DBackpropInput", {x, _filter, dout},
                                                  {{"stride", ib->GetAttr("stride")},
                                                   {"dilation", ib->GetAttr("dilation")},
                                                   {"pad_mode", ib->GetAttr("pad_mode")},
                                                   {"format", ib->GetAttr("format")}})
                                       : ib->OutZeros(x);
  auto dfilter = _filter->need_compute_grad_out() ? ib->Emit("Dilation2DBackpropFilter", {x, _filter, dout},
                                                             {{"stride", ib->GetAttr("stride")},
                                                              {"dilation", ib->GetAttr("dilation")},
                                                              {"pad_mode", ib->GetAttr("pad_mode")},
                                                              {"format", ib->GetAttr("format")}})
                                                  : ib->OutZeros(_filter);
  return {dx, dfilter};
});

REG_BPROP_BUILDER("CeLU").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto x_dtype = ib->GetDtype(x);
  const auto &alpha = ib->GetInput(i1);
  auto alpha_value = GetValue<float>(alpha->BuildValue());
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto greater = ib->GreaterEqualScalar(x, ib->ValueByType(0.0, x_dtype));

  auto dx = ib->Mul(dout, ib->Select(greater, ib->Fill(1.0, ib->Shape(x), x_dtype->type_id()),
                                     ib->AddScalar((ib->RealDiv(out, ib->Tensor(alpha_value, x_dtype))),
                                                   ib->ValueByType(1.0, x_dtype), ib->Value(1))));
  return {dx, ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Pdist").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("PdistGrad", {dout, x, out}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("MultiMarginLoss").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &target = ib->GetInput(i1);
  const auto &weight = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx =
    ib->Emit("MultiMarginLossGrad", {dout, x, target, weight},
             {{"p", ib->GetAttr("p")}, {"margin", ib->GetAttr("margin")}, {"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->OutZeros(target), ib->OutZeros(weight)};
});

REG_BPROP_BUILDER("DropoutGenMask").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("DropoutDoMask").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i1);
  const auto &keep_prob = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  return {ib->Emit("DropoutDoMask", {dout, y, keep_prob}), ib->OutZeros(y), ib->OutZeros(keep_prob)};
});

REG_BPROP_BUILDER("ReluGrad").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dgrad = ib->ReluGrad(dout, y);
  return {dgrad, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("GridSampler3D").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &grid = ib->GetInput(i1);
  const auto &interpolation_mode = ib->GetInput(i2);
  const auto &padding_mode = ib->GetInput(i3);
  const auto &align_corners = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  std::vector<int64_t> output_mask_vec = {input_x->need_compute_grad_out(), grid->need_compute_grad_out()};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto tmp = ib->GridSampler3DGrad(dout, input_x, grid, interpolation_mode, padding_mode, align_corners, output_mask);
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dgrid = ib->TupleGetItem(tmp, 1);
  auto grad_interpolation_mode = ib->OutZeros(interpolation_mode);
  auto grad_padding_mode = ib->OutZeros(padding_mode);
  auto grad_align_corners = ib->OutZeros(align_corners);
  return {dx, dgrid, grad_interpolation_mode, grad_padding_mode, grad_align_corners};
});

REG_BPROP_BUILDER("ReLUV3").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dgrad = ib->ReluGrad(dout, out);
  return {dgrad};
});

REG_BPROP_BUILDER("GridSampler2D").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &grid = ib->GetInput(i1);
  const auto &interpolation_mode = ib->GetInput(i2);
  const auto &padding_mode = ib->GetInput(i3);
  const auto &align_corners = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i6);
  std::vector<int64_t> output_mask_vec = {input_x->need_compute_grad_out(), grid->need_compute_grad_out()};
  auto output_mask = ib->EmitValue(MakeValue(output_mask_vec));
  auto tmp = ib->GridSampler2DGrad(dout, input_x, grid, interpolation_mode, padding_mode, align_corners, output_mask);
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dgrid = ib->TupleGetItem(tmp, 1);
  auto grad_interpolation_mode = ib->OutZeros(interpolation_mode);
  auto grad_padding_mode = ib->OutZeros(padding_mode);
  auto grad_align_corners = ib->OutZeros(align_corners);
  return {dx, dgrid, grad_interpolation_mode, grad_padding_mode, grad_align_corners};
});

REG_BPROP_BUILDER("ResizeLinear1D").SetUnusedInputs({i1, i3}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &size = ib->GetInput(i1);
  const auto &coordinate_transformation_mode = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  auto dx = ib->Emit("ResizeLinear1DGrad", {dout, input_x, coordinate_transformation_mode});
  return {dx, ib->OutZeros(size), ib->OutZeros(coordinate_transformation_mode)};
});

REG_BPROP_BUILDER("MaxPool3DWithArgmax").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MaxPool3DGradWithArgmax", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"format", ib->GetAttr("format")},
                      {"argmax_type", ib->GetAttr("argmax_type")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxUnpool2D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &argmax = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("MaxUnpool2DGrad", {x, dout, argmax},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  auto dargmax = ib->OutZeros(argmax);
  return {dx, dargmax};
});

DEF_PURE_SHAPE_CALC(max_unpool2d_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(i0);
    ShapeVector x_2d_shape{x_shape.begin(), x_shape.end() - 2};
    x_2d_shape.push_back(-1);
    return {x_2d_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto &x_shape = inputs[i0];
    return {
      IsDynamicRank(x_shape) ? -1LL : static_cast<int64_t>(x_shape.size()) - 1LL,
    };
  });

REG_BPROP_BUILDER("MaxUnpool2DExt").SetUnusedInputs({i0, i6}).SetBody(BODYFUNC(ib) {
  const auto &indices = ib->GetInput(i1);
  const auto &kernel_size = ib->GetInput(i2);
  const auto &strides = ib->GetInput(i3);
  const auto &padding = ib->GetInput(i4);
  const auto &output_shape = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i7);
  auto indices_shape = ib->Shape(indices);
  auto indices_shape_vec = ib->GetShape(indices);
  NodePtr dx;
  if (IsDynamic(indices_shape_vec)) {
    NodePtrList ret_shape = ib->ShapeCalc(max_unpool2d_ext_shapecalc, {indices});
    auto indices_view = ib->Reshape(indices, ret_shape[0]);
    auto dout_view = ib->Reshape(dout, ret_shape[0]);
    auto dx_gather = ib->GatherD(dout_view, ib->Value<int64_t>(-1), indices_view);
    dx = ib->Reshape(dx_gather, indices_shape);
  } else if (indices_shape_vec.size() != 0) {
    ShapeVector indices_size{indices_shape_vec.begin(), indices_shape_vec.end() - 2};
    indices_size.push_back(-1);
    auto indices_view = ib->Reshape(indices, indices_size);
    auto dout_view = ib->Reshape(dout, indices_size);
    auto dx_gather = ib->GatherD(dout_view, ib->Value<int64_t>(-1), indices_view);
    dx = ib->Reshape(dx_gather, indices_shape);
  } else {
    dx = ib->OutZeros(indices);
  }
  return {dx,
          ib->OutZeros(indices),
          ib->OutZeros(kernel_size),
          ib->OutZeros(strides),
          ib->OutZeros(padding),
          ib->OutZeros(output_shape)};
});

REG_BPROP_BUILDER("MaxUnpool3D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &argmax = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("MaxUnpool3DGrad", {x, dout, argmax},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  auto dargmax = ib->OutZeros(argmax);
  return {dx, dargmax};
});

REG_BPROP_BUILDER("NthElement").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &n = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  auto dout = ib->GetInput(i3);
  auto indicators = ib->Equal(ib->ExpandDims(out, -1), input_x, kFloat32);
  dout = ib->ExpandDims(dout, -1);
  auto num_select = ib->ExpandDims(ib->ReduceSum(indicators, {-1}), -1);
  return {ib->Cast(ib->Mul(ib->Div(indicators, num_select), dout), ib->GetDtype(input_x)), ib->OutZeros(n)};
});

REG_BPROP_BUILDER("AdaptiveAvgPool3D").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->Shape(x, true);
  auto dx = ib->Emit("AdaptiveAvgPool3DGrad", {dout, ib->Cast(x_shape, kInt32)});
  return {dx};
});

REG_BPROP_BUILDER("AdaptiveAvgPool3DExt").FreeUselessValues_O({}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dim1d = 1;
  auto out_shape = out->shape();

  if (IsDynamic(out_shape)) {
    return {ib->AdaptiveAvgPool3DGradExt(dout, x), ib->OutZeros(output_size)};
  } else {
    ShapeVector out_shape_last_3d(out_shape.begin() + out_shape.size() - 3, out_shape.end());
    if (out_shape_last_3d[i0] == dim1d && out_shape_last_3d[i1] == dim1d && out_shape_last_3d[i2] == dim1d) {
      return {MeanExtGrad(ib, x, out, dout), ib->OutZeros(output_size)};
    } else {
      return {ib->AdaptiveAvgPool3DGradExt(dout, x), ib->OutZeros(output_size)};
    }
  }
});

REG_BPROP_BUILDER("AdaptiveAvgPool2D").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  auto shape = ib->Shape(x, true);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("AdaptiveAvgPool2DGrad", {dout, ib->Cast(shape, kInt64)});
  return {dx};
});

DEF_PURE_SHAPE_CALC(g_adaptive_pool1d_squeeze)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    auto output_size = inputs.at(1);

    MS_EXCEPTION_IF_CHECK_FAIL(output_size.size() == 1, "output_size should be a scalar.");
    auto output_size_value = output_size[0];

    output_size_value = output_size_value < 0 ? output_size_value + x_shape.size() : output_size_value;
    ShapeVector squeeze_shape = x_shape;
    MS_EXCEPTION_IF_CHECK_FAIL(squeeze_shape.size() > 2, "shape size should be greater than 2.");
    squeeze_shape.erase(squeeze_shape.end() - 2);
    return {squeeze_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    auto output_size = inputs.at(1);
    if (!unknown_inputs.empty() || IsDynamicRank(x) || IsDynamicRank(output_size)) {
      return {-1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size - 1};
  });

REG_BPROP_BUILDER("AdaptiveAvgPool1D").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);

  auto dout_expand_dim = ib->ExpandDims(dout, -2);
  auto x_expand_dim = ib->ExpandDims(x, -2);
  auto dx = ib->AdaptiveAvgPool2DGradExt(dout_expand_dim, x_expand_dim);
  auto res_shape = ib->ShapeCalc(g_adaptive_pool1d_squeeze, {dx, output_size}, {1});
  auto dx_squeeze = ib->Reshape(dx, res_shape[0]);
  return {dx_squeeze, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("AdaptiveAvgPool2DExt").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->AdaptiveAvgPool2DGradExt(dout, x);
  return {dx, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("AdaptiveMaxPool1D").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &output_size = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto out_index = ib->TupleGetItem(out, i1);
  auto dy = ib->TupleGetItem(dout, i0);

  auto dy_expand_dim = ib->ExpandDims(dy, -2);
  auto x_expand_dim = ib->ExpandDims(x, -2);
  auto out_index_expand_dim = ib->ExpandDims(out_index, -2);
  auto dx = ib->Emit("AdaptiveMaxPool2DGrad", {dy_expand_dim, x_expand_dim, out_index_expand_dim});
  auto res_shape = ib->ShapeCalc(g_adaptive_pool1d_squeeze, {dx, output_size}, {1});
  auto dx_squeeze = ib->Reshape(dx, res_shape[0]);
  return {dx_squeeze, ib->OutZeros(output_size)};
});

REG_BPROP_BUILDER("FractionalMaxPool").FreeUselessValues_O({i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit(
    "FractionalMaxPoolGrad",
    {x, ib->TupleGetItem(out, 0), ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2)},
    {{"overlapping", ib->GetAttr("overlapping")}});

  return {dx};
});

REG_BPROP_BUILDER("FractionalMaxPool3DWithFixedKsize").FreeUselessValues_IO({i1}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &random_samples = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("FractionalMaxPool3DGradWithFixedKsize", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"format", ib->GetAttr("format")}});
  return {dx, ib->OutZeros(random_samples)};
});

REG_BPROP_BUILDER("FractionalAvgPool").FreeUselessValues_IO({i0}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto x_shape = ib->Shape(x, true);
  auto dx = ib->Emit("FractionalAvgPoolGrad",
                     {x_shape, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2)},
                     {{"overlapping", ib->GetAttr("overlapping")}, {"max_length", MakeValue<int64_t>(1000000)}});
  return {dx};
});

REG_BPROP_BUILDER("PSROIPooling").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto spatial_scale = ib->GetAttr("spatial_scale");
  auto group_size = ib->GetAttr("group_size");
  auto output_dim = ib->GetAttr("output_dim");
  const auto &x = ib->GetInput(i0);
  const auto &rois = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto shape = ib->GetShape(x);
  ShapeVector input_size;
  if (IsDynamicRank(shape)) {
    input_size = shape;
  } else {
    for (size_t i = 2; i < shape.size(); i++) {
      input_size.push_back(shape[i]);
    }
  }
  auto dx = ib->Emit("PSROIPoolingGrad", {dout, rois},
                     {
                       {"input_size", MakeValue(input_size)},
                       {"spatial_scale", spatial_scale},
                       {"group_size", group_size},
                       {"output_dim", output_dim},
                     });
  return {dx, ib->OutZeros(rois)};
});

REG_BPROP_BUILDER("AvgPoolV1").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i2);
  auto orig_input_shape = ib->Shape(x, true);
  auto dx = ib->Emit("AvgPoolGradV1", {orig_input_shape, dout},
                     {
                       {"kernel_size", ib->GetAttr("kernel_size")},
                       {"strides", ib->GetAttr("strides")},
                       {"pad_mode", ib->GetAttr("pad_mode")},
                       {"format", ib->GetAttr("format")},
                     });
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolV1").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &out = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i2);
  auto dx = ib->Emit("MaxPoolGradV1", {x, out, dout},
                     {
                       {"kernel_size", ib->GetAttr("kernel_size")},
                       {"strides", ib->GetAttr("strides")},
                       {"pad_mode", ib->GetAttr("pad_mode")},
                       {"format", ib->GetAttr("format")},
                     });
  return {dx};
});

REG_BPROP_BUILDER("CTCLossV2").SetBody(BODYFUNC(ib) {
  const auto &log_probs = ib->GetInput(i0);
  const auto &targets = ib->GetInput(i1);
  const auto &input_lengths = ib->GetInput(i2);
  const auto &target_lengths = ib->GetInput(i3);
  const auto &out = ib->GetInput(i4);
  const auto &dout = ib->GetInput(i5);
  auto grad = ib->Emit("CTCLossV2Grad",
                       {ib->TupleGetItem(dout, 0), log_probs, targets, input_lengths, target_lengths,
                        ib->TupleGetItem(out, 0), ib->TupleGetItem(out, 1)},
                       {{"blank", ib->GetAttr("blank")},
                        {"reduction", ib->GetAttr("reduction")},
                        {"zero_infinity", ib->GetAttr("zero_infinity")}});
  return {grad, ib->OutZeros(targets), ib->OutZeros(input_lengths), ib->OutZeros(target_lengths)};
});

REG_BPROP_BUILDER("InstanceNormV2").FreeUselessValues_IO({i2}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &gamma = ib->GetInput(i1);
  const auto &mean = ib->GetInput(i3);
  const auto &variance = ib->GetInput(i4);
  const auto &out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
  auto saved_mean = ib->TupleGetItem(out, 1);
  auto saved_variance = ib->TupleGetItem(out, 2);
  auto grad_ops_out =
    ib->Emit("InstanceNormV2Grad", {ib->TupleGetItem(dout, 0), x, gamma, mean, variance, saved_mean, saved_variance},
             {{"is_training", ib->GetAttr("is_training")},
              {"epsilon", ib->GetAttr("epsilon")},
              {"momentum", ib->GetAttr("momentum")}});
  auto dx = ib->TupleGetItem(grad_ops_out, 0);
  auto dgamma = ib->TupleGetItem(grad_ops_out, 1);
  auto dbeta = ib->TupleGetItem(grad_ops_out, 2);
  return {dx, dgamma, dbeta, ib->OutZeros(mean), ib->OutZeros(variance)};
});

REG_BPROP_BUILDER("FractionalMaxPoolWithFixedKsize").FreeUselessValues_IO({i1}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &random_samples = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->Emit("FractionalMaxPoolGradWithFixedKsize", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  return {dx, ib->OutZeros(random_samples)};
});

REG_BPROP_BUILDER("SparseSoftmaxCrossEntropyWithLogitsV2").FreeUselessValues_IO({i1}, {i0}).SetBody(BODYFUNC(ib) {
  const auto &logits = ib->GetInput(i0);
  const auto &labels = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto grad_loss = ib->TupleGetItem(dout, 0);
  auto softmax_grad = ib->TupleGetItem(out, 1);
  int64_t axis_1 = -1;
  int64_t axis_2 = 2;
  grad_loss = ib->ExpandDims(grad_loss, axis_1);
  auto grad = ib->Mul(grad_loss, softmax_grad);
  if (ib->TupleGetItem(dout, 1) != nullptr) {
    auto softmax = ib->Softmax(logits, ib->Value<ShapeVector>({1}));
    auto x = ib->ExpandDims(ib->TupleGetItem(dout, 1), 1);
    auto y = ib->ExpandDims(softmax, axis_2);
    auto matmul_tmp = ib->BatchMatMul(x, y);
    grad = grad + (ib->TupleGetItem(dout, 1) - ib->Squeeze(matmul_tmp, MakeValue(ShapeVector{1}))) * softmax;
  }
  return {grad, ib->OutZeros(labels)};
});

void FreeTensorsOfPadV3(const PynativeCallback &cb) {
  cb.FreeInputDeviceAddress({i0});
  cb.FreeOutputDeviceAddress({i0});
}

REG_BPROP_BUILDER("PadV3").FreeUselessValues(FreeTensorsOfPadV3).SetBody(BODYFUNC(ib) {
  const auto &paddings = ib->GetInput(i1);
  bool has_constant_values = ib->GetInputs().size() == i5;
  auto dout = has_constant_values ? ib->GetInput(i4) : ib->GetInput(i3);
  auto mode = GetValue<std::string>(ib->GetAttr("mode"));
  NodePtr dx;

  if (mode == "constant") {
    MS_EXCEPTION_IF_NULL(paddings);
    auto pad_value = GetIntList(paddings);
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
      (void)CheckAndConvertUtils::CheckPositiveVector("paddings", pad_value, "PadV3Grad");
      const auto &x = ib->GetInput(i0);
      auto x_shape = ib->GetShape(x);
      std::vector<std::vector<int64_t>> ordered_paddings(x_shape.size(), {0, 0});
      const size_t step_2 = 2;
      for (size_t i = 0; i < pad_value.size(); i += step_2) {
        std::vector<int64_t> split_paddings = {pad_value[i], pad_value[i + 1]};
        ordered_paddings[x_shape.size() - (i / step_2) - 1] = split_paddings;
      }
      std::vector<int64_t> begin;
      for (const auto &item : ordered_paddings) {
        begin.emplace_back(item[0]);
      }
      dx = ib->Slice(dout, ib->EmitValue(MakeValue(begin)), ib->EmitValue(MakeValue(x_shape)));
    } else {
      (void)std::transform(pad_value.begin(), pad_value.end(), pad_value.begin(), [](const int64_t &c) { return -c; });
      auto constant_values = ib->GetInput(i2);
      dx = ib->Emit("PadV3", {dout, ib->Tensor(pad_value), ib->ZerosLike(constant_values)},
                    {{"mode", ib->GetAttr("mode")}, {"paddings_contiguous", ib->GetAttr("paddings_contiguous")}});
    }
  } else {
    dx = ib->Emit("PadV3Grad", {dout, paddings},
                  {{"mode", ib->GetAttr("mode")}, {"paddings_contiguous", ib->GetAttr("paddings_contiguous")}});
  }
  if (has_constant_values) {
    auto constant_values = ib->GetInput(i2);
    return {dx, ib->OutZeros(paddings), ib->OutZeros(constant_values)};
  } else {
    return {dx, ib->OutZeros(paddings)};
  }
});

REG_BPROP_BUILDER("ConstantPadND").FreeUselessValues_IO({i0}, {}).SetBody(BODYFUNC(ib) {
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i4);
  NodePtr neg_pad;

  MS_EXCEPTION_IF_NULL(paddings);
  auto pad_opt = GetArrayValue<int64_t>(paddings->BuildValue());
  if (pad_opt.has_value()) {
    auto pad_value = pad_opt.value().ToVector();
    (void)std::transform(pad_value.begin(), pad_value.end(), pad_value.begin(), [](const int64_t &c) { return -c; });
    neg_pad = ib->Value<ShapeVector>(pad_value);
  } else {
    auto pad_tensor = ib->SequenceToTensor(paddings);
    auto neg_pad_tensor = ib->Neg(pad_tensor);
    neg_pad = ib->TensorToTuple(neg_pad_tensor);
  }

  auto constant_values = ib->GetInput(i2);
  auto dx = ib->ConstantPadND(dout, neg_pad, ib->EmitValue(MakeValue<int64_t>(0)));
  return {dx, ib->OutZeros(paddings), ib->OutZeros(constant_values)};
});

REG_BPROP_BUILDER("ReflectionPad1D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReflectionPad1DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("ReflectionPad2D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReflectionPad2DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("ReflectionPad3D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReflectionPad3DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("ReplicationPad1D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReplicationPad1DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("ReplicationPad2D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReplicationPad2DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("ReplicationPad3D").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->GetInput(i0);
  const auto &paddings = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = ib->ReplicationPad3DGrad(dout, input_x, paddings);
  return {dx, ib->OutZeros(paddings)};
});

REG_BPROP_BUILDER("WKV").FreeUselessValues_IO({i4, i5, i6}, {}).SetBody(BODYFUNC(ib) {
  const auto &w = ib->GetInput(i0);
  const auto &u = ib->GetInput(i1);
  const auto &k = ib->GetInput(i2);
  const auto &v = ib->GetInput(i3);
  const auto &sp = ib->GetInput(i4);
  const auto &sq = ib->GetInput(i5);
  const auto &sm = ib->GetInput(i6);
  const auto &dout = ib->GetInput(i8);
  auto dy = ib->TupleGetItem(dout, i0);
  auto grad = ib->Emit("WKVGrad", {w, u, k, v, dy});
  std::vector<int64_t> axis = {0};
  auto gw = w->need_compute_grad_out() ? ib->ReduceSum(ib->TupleGetItem(grad, i0), axis) : ib->OutZeros(w);
  auto gu = u->need_compute_grad_out() ? ib->ReduceSum(ib->TupleGetItem(grad, i1), axis) : ib->OutZeros(u);
  auto gk = k->need_compute_grad_out() ? ib->TupleGetItem(grad, i2) : ib->OutZeros(k);
  auto gv = v->need_compute_grad_out() ? ib->TupleGetItem(grad, i3) : ib->OutZeros(v);
  return {gw, gu, gk, gv, ib->ZerosLike(sp), ib->ZerosLike(sq), ib->ZerosLike(sm)};
});

REG_BPROP_BUILDER("FlashAttentionScore").SetBody((BODYFUNC(ib) {
  auto query = ib->GetInput(i0);
  auto key = ib->GetInput(i1);
  auto value = ib->GetInput(i2);
  auto pse_shift = ib->GetInput(i3);
  auto drop_mask = ib->GetInput(i4);
  auto padding_mask = ib->GetInput(i5);
  auto attn_mask = ib->GetInput(i6);
  auto prefix = ib->GetInput(i7);
  auto actual_seq_qlen = ib->GetInput(i8);
  auto actual_seq_kvlen = ib->GetInput(i9);
  auto head_num = ib->GetInput(i10);
  auto keep_prob = ib->GetInput(i11);
  auto scale_value = ib->GetInput(i12);
  auto pre_tokens = ib->GetInput(i13);
  auto next_tokens = ib->GetInput(i14);
  auto inner_precise = ib->GetInput(i15);
  auto input_layout = ib->GetInput(i16);
  auto sparse_mode = ib->GetInput(i17);
  auto out = ib->GetInput(i18);
  auto softmax_max = ib->TupleGetItem(out, i0);
  auto softmax_sum = ib->TupleGetItem(out, i1);
  auto softmax_out = ib->TupleGetItem(out, i2);
  auto attention_out = ib->TupleGetItem(out, i3);
  auto dout = ib->GetInput(i19);
  auto d_attention_out = ib->TupleGetItem(dout, i3);
  auto grad = ib->FlashAttentionScoreGrad(query, key, value, d_attention_out, pse_shift, drop_mask, padding_mask,
                                          attn_mask, softmax_max, softmax_sum, softmax_out, attention_out, prefix,
                                          actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value,
                                          pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode);
  auto g_query = ib->TupleGetItem(grad, i0);
  auto g_key = ib->TupleGetItem(grad, i1);
  auto g_value = ib->TupleGetItem(grad, i2);
  auto g_pse_shift = ib->TupleGetItem(grad, i3);
  auto g_drop_mask = ib->OutZeros(drop_mask);
  auto g_padding_mask = ib->OutZeros(padding_mask);
  auto g_attn_mask = ib->OutZeros(attn_mask);
  auto g_prefix = ib->OutZeros(prefix);
  auto g_actual_seq_qlen = ib->OutZeros(actual_seq_qlen);
  auto g_actual_seq_kvlen = ib->OutZeros(actual_seq_kvlen);
  auto g_head_num = ib->OutZeros(head_num);
  auto g_keep_prob = ib->OutZeros(keep_prob);
  auto g_scale_value = ib->OutZeros(scale_value);
  auto g_pre_tokens = ib->OutZeros(pre_tokens);
  auto g_next_tokens = ib->OutZeros(next_tokens);
  auto g_inner_precise = ib->OutZeros(inner_precise);
  auto g_input_layout = ib->OutZeros(input_layout);
  auto g_sparse_mode = ib->OutZeros(sparse_mode);
  return {g_query,       g_key,        g_value,           g_pse_shift,        g_drop_mask,    g_padding_mask,
          g_attn_mask,   g_prefix,     g_actual_seq_qlen, g_actual_seq_kvlen, g_head_num,     g_keep_prob,
          g_scale_value, g_pre_tokens, g_next_tokens,     g_inner_precise,    g_input_layout, g_sparse_mode};
}));

REG_BPROP_BUILDER("SpeedFusionAttention").SetBody((BODYFUNC(ib) {
  auto query = ib->GetInput(i0);
  auto key = ib->GetInput(i1);
  auto value = ib->GetInput(i2);
  auto head_num = ib->GetInput(i3);
  auto input_layout = ib->GetInput(i4);
  auto seed_in = ib->GetInput(i5);
  auto offset_in = ib->GetInput(i6);
  auto pse = ib->GetInput(i7);
  auto padding_mask = ib->GetInput(i8);
  auto atten_mask = ib->GetInput(i9);
  auto scale = ib->GetInput(i10);
  auto keep_prob = ib->GetInput(i11);
  auto pre_tokens = ib->GetInput(i12);
  auto next_tokens = ib->GetInput(i13);
  auto inner_precise = ib->GetInput(i14);
  auto prefix = ib->GetInput(i15);
  auto actual_seq_qlen = ib->GetInput(i16);
  auto actual_seq_kvlen = ib->GetInput(i17);
  auto sparse_mode = ib->GetInput(i18);
  auto gen_mask_parallel = ib->GetInput(i19);
  auto sync = ib->GetInput(i20);
  auto pse_type = ib->GetInput(i21);
  auto q_start_idx = ib->GetInput(i22);
  auto kv_start_idx = ib->GetInput(i23);
  auto out = ib->GetInput(i24);
  auto dout = ib->GetInput(i25);

  auto attention_out = ib->TupleGetItem(out, i0);
  auto softmax_max = ib->TupleGetItem(out, i1);
  auto softmax_sum = ib->TupleGetItem(out, i2);
  auto softmax_out = ib->TupleGetItem(out, i3);
  auto seed = ib->TupleGetItem(out, i4);
  auto offset = ib->TupleGetItem(out, i5);
  auto numels = ib->TupleGetItem(out, i6);
  auto dy = ib->TupleGetItem(dout, i0);

  auto ret = ib->SpeedFusionAttentionGrad(
    query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_out,
    attention_out, scale, keep_prob, pre_tokens, next_tokens, inner_precise, seed, offset, numels, prefix,
    actual_seq_qlen, actual_seq_kvlen, sparse_mode, gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);

  auto dq = ib->TupleGetItem(ret, i0);
  auto dk = ib->TupleGetItem(ret, i1);
  auto dv = ib->TupleGetItem(ret, i2);
  auto dp = ib->TupleGetItem(ret, i3);

  return {dq,
          dk,
          dv,
          ib->OutZeros(head_num),
          ib->OutZeros(input_layout),
          ib->OutZeros(seed_in),
          ib->OutZeros(offset_in),
          dp,
          ib->OutZeros(padding_mask),
          ib->OutZeros(atten_mask),
          ib->OutZeros(scale),
          ib->OutZeros(keep_prob),
          ib->OutZeros(pre_tokens),
          ib->OutZeros(next_tokens),
          ib->OutZeros(inner_precise),
          ib->OutZeros(prefix),
          ib->OutZeros(actual_seq_qlen),
          ib->OutZeros(actual_seq_kvlen),
          ib->OutZeros(sparse_mode),
          ib->OutZeros(gen_mask_parallel),
          ib->OutZeros(sync),
          ib->OutZeros(pse_type),
          ib->OutZeros(q_start_idx),
          ib->OutZeros(kv_start_idx)};
}));

REG_BPROP_BUILDER("NsaSelectAttention").SetBody((BODYFUNC(ib) {
  auto query = ib->GetInput(i0);
  auto key = ib->GetInput(i1);
  auto value = ib->GetInput(i2);
  auto topk_indices = ib->GetInput(i3);
  auto scale_value = ib->GetInput(i4);
  auto head_num = ib->GetInput(i5);
  auto select_block_size = ib->GetInput(i6);
  auto select_block_count = ib->GetInput(i7);
  auto atten_mask = ib->GetInput(i8);
  auto actual_seq_qlen = ib->GetInput(i9);
  auto actual_seq_kvlen = ib->GetInput(i10);
  auto out = ib->GetInput(i11);
  auto dout = ib->GetInput(i12);

  auto attention_out = ib->TupleGetItem(out, i0);
  auto softmax_max = ib->TupleGetItem(out, i1);
  auto softmax_sum = ib->TupleGetItem(out, i2);

  auto grad = ib->TupleGetItem(dout, i0);

  auto ret = ib->Emit("NsaSelectAttentionGrad",
                      {grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale_value,
                       head_num, select_block_size, select_block_count, atten_mask, actual_seq_qlen, actual_seq_kvlen});

  auto dq = ib->TupleGetItem(ret, i0);
  auto dk = ib->TupleGetItem(ret, i1);
  auto dv = ib->TupleGetItem(ret, i2);

  return {dq,
          dk,
          dv,
          ib->OutZeros(topk_indices),
          ib->OutZeros(scale_value),
          ib->OutZeros(head_num),
          ib->OutZeros(select_block_size),
          ib->OutZeros(select_block_count),
          ib->OutZeros(atten_mask),
          ib->OutZeros(actual_seq_qlen),
          ib->OutZeros(actual_seq_kvlen)};
}));

REG_BPROP_BUILDER("RmsNorm").FreeUselessValues_IO({i2}, {i0}).SetBody((BODYFUNC(ib) {
  auto x = ib->GetInput(i0);
  auto gamma = ib->GetInput(i1);
  auto eps = ib->GetInput(i2);
  auto out = ib->GetInput(i3);
  auto dout = ib->GetInput(i4);
  auto rstd = ib->TupleGetItem(out, i1);
  auto dy = ib->TupleGetItem(dout, i0);

  auto grad = ib->RmsNormGrad(dy, x, rstd, gamma);
  auto dx = ib->TupleGetItem(grad, i0);
  auto dgamma_raw = ib->TupleGetItem(grad, i1);
  auto dgamma = ib->Cast(dgamma_raw, ib->GetDtype(gamma));
  return {dx, dgamma, ib->OutZeros(eps)};
}));

REG_BPROP_BUILDER("AddRmsNorm").FreeUselessValues_IO({i0, i1, i3}, {i0}).SetBody((BODYFUNC(ib) {
  auto x1 = ib->GetInput(i0);
  auto x2 = ib->GetInput(i1);
  auto gamma = ib->GetInput(i2);
  auto eps = ib->GetInput(i3);
  auto out = ib->GetInput(i4);
  auto dout = ib->GetInput(i5);
  auto rstd = ib->TupleGetItem(out, i1);
  auto x_sum = ib->TupleGetItem(out, i2);
  auto dy = ib->TupleGetItem(dout, i0);

  auto grad = ib->RmsNormGrad(dy, x_sum, rstd, gamma);
  auto dx1 = x1->need_compute_grad_out() ? ib->TupleGetItem(grad, i0) : ib->OutZeros(x1);
  auto dx2 = x2->need_compute_grad_out() ? ib->TupleGetItem(grad, i0) : ib->OutZeros(x2);
  auto dgamma_raw = ib->TupleGetItem(grad, i1);
  auto dgamma = gamma->need_compute_grad_out() ? ib->Cast(dgamma_raw, ib->GetDtype(gamma)) : ib->OutZeros(gamma);
  return {dx1, dx2, dgamma, ib->OutZeros(eps)};
}));

REG_BPROP_BUILDER("AvgPool2D").FreeUselessValues_O({}).SetBody((BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto kernel_size = ib->GetInput(i1);
  auto stride = ib->GetInput(i2);
  auto padding = ib->GetInput(i3);
  auto ceil_mode = ib->GetInput(i4);
  auto count_include_pad = ib->GetInput(i5);
  auto divisor_override = ib->GetInput(i6);
  auto dout = ib->GetInput(i8);

  auto dx =
    ib->AvgPool2DGrad(dout, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return {dx,
          ib->OutZeros(kernel_size),
          ib->OutZeros(stride),
          ib->OutZeros(padding),
          ib->OutZeros(ceil_mode),
          ib->OutZeros(count_include_pad),
          ib->OutZeros(divisor_override)};
}));

REG_BPROP_BUILDER("AvgPool2DGrad").FreeUselessValues_O({}).SetBody((BODYFUNC(ib) {
  auto grad_output = ib->GetInput(i0);
  auto image = ib->GetInput(i1);
  auto kernel_size = ib->GetInput(i2);
  auto stride = ib->GetInput(i3);
  auto padding = ib->GetInput(i4);
  auto ceil_mode = ib->GetInput(i5);
  auto count_include_pad = ib->GetInput(i6);
  auto divisor_override = ib->GetInput(i7);
  auto dout = ib->GetInput(i9);

  auto grad_dout = ib->AvgPool2D(dout, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  return {grad_dout,
          ib->OutZeros(image),
          ib->OutZeros(kernel_size),
          ib->OutZeros(stride),
          ib->OutZeros(padding),
          ib->OutZeros(ceil_mode),
          ib->OutZeros(count_include_pad),
          ib->OutZeros(divisor_override)};
}));

DEF_PURE_SHAPE_CALC(g_avg_pool1d_squeeze)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    ShapeVector squeeze_shape = x_shape;
    MS_EXCEPTION_IF_CHECK_FAIL(squeeze_shape.size() > 2, "shape size should be greater than 2.");
    squeeze_shape.erase(squeeze_shape.end() - 2);
    return {squeeze_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto x = inputs.at(0);
    if (!unknown_inputs.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    auto size = SizeToLong(x.size());
    return {size - 1};
  });

REG_BPROP_BUILDER("AvgPool1D").SetBody((BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto kernel_size = ib->GetInput(i1);
  auto stride_opt = ib->GetInput(i2);
  auto stride = ib->GetDtype(stride_opt)->isa<TypeNone>() ? kernel_size : stride_opt;
  auto padding = ib->GetInput(i3);
  auto ceil_mode = ib->GetInput(i4);
  auto count_include_pad = ib->GetInput(i5);
  auto divisor_override = ib->EmitValue(kNone);

  auto expanded_kernel_size =
    ib->MakeTuple(std::vector<NodePtr>{ib->Value<int64_t>(1), ib->TupleGetItem(kernel_size, 0)});
  auto expanded_stride = ib->MakeTuple(std::vector<NodePtr>{ib->Value<int64_t>(1), ib->TupleGetItem(stride, 0)});
  auto expanded_padding = ib->MakeTuple(std::vector<NodePtr>{ib->Value<int64_t>(0), ib->TupleGetItem(padding, 0)});

  auto dout = ib->GetInput(i7);
  auto dout_expand_dim = ib->ExpandDims(dout, -2);
  auto x_expand_dim = ib->ExpandDims(input, -2);

  auto dx = ib->AvgPool2DGrad(dout_expand_dim, x_expand_dim, expanded_kernel_size, expanded_stride, expanded_padding,
                              ceil_mode, count_include_pad, divisor_override);

  auto res_shape = ib->ShapeCalc(g_avg_pool1d_squeeze, {dx});
  auto dx_squeeze = ib->Reshape(dx, res_shape[0]);

  return {dx_squeeze,
          ib->OutZeros(kernel_size),
          ib->OutZeros(stride),
          ib->OutZeros(padding),
          ib->OutZeros(ceil_mode),
          ib->OutZeros(count_include_pad)};
}));

REG_BPROP_BUILDER("KLDiv").SetUnusedInputs({i4}).SetBody((BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto target = ib->GetInput(i1);
  auto reduction = ib->GetInput(i2);
  auto log_target = ib->GetInput(i3);
  auto dout = ib->GetInput(i5);

  auto dx = ib->KLDivGrad(dout, input, target, reduction, log_target);
  return {dx, ib->OutZeros(target), ib->OutZeros(reduction), ib->OutZeros(log_target)};
}));

REG_BPROP_BUILDER("Generator").SetBody(ReturnZeros);

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
