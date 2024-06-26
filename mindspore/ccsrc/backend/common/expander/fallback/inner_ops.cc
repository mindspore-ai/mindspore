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

#include <vector>
#include <string>
#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace expander {
REG_FALLBACK_BUILDER("VmapStackAssign").SetBody(BODYFUNC(ib) {
  constexpr const size_t kInputLowerLimit = 4;
  auto &inputs = ib->GetInputs();
  if (inputs.size() < kInputLowerLimit) {
    MS_LOG(INFO) << "In VmapStackAssign, inputs size must be greater than 3, but got " << inputs.size() << ".";
    return {};
  }
  // The input format is: [stacked_parameter, param1, param2, ...(a batch of parameters), UMonad].
  auto params_size = inputs.size() - kIndex2;
  NodePtrList params_to_stack;
  params_to_stack.reserve(params_size);
  (void)params_to_stack.emplace_back(ib->Depend(inputs[kIndex1], inputs.back()));
  (void)params_to_stack.insert(params_to_stack.end(), inputs.begin() + kIndex2, inputs.end() - kIndex1);
  std::vector<int64_t> dyn_input_sizes = {SizeToLong(params_size)};
  auto stacked_para = ib->Emit("Stack", params_to_stack,
                               {{"axis", MakeValue<int64_t>(0LL)},
                                {"N", MakeValue<int64_t>(SizeToLong(params_size))},
                                {kAttrDynInputSizes, MakeValue(dyn_input_sizes)}});
  auto assigned_param = ib->Emit("Assign", {inputs[0], stacked_para});
  // the VmapStackAssign's output shape is always [1], and dtype is kInt32.
  auto result = ib->Depend(ib->Tensor(std::vector<int>{0}, kInt32), assigned_param);
  return {result};
});

std::string FFNExtConvertEnumToString(int64_t id) {
  static const std::vector<std::string> activation_mode = {
    "no_activation", "relu", "sigmoid", "relu6",   "elu",      "leaky_relu",    "abs",    "relu1",     "softsign",
    "softplus",      "tanh", "selu",    "hswish",  "hsigmoid", "thresholdrelu", "linear", "hard_tanh", "sign",
    "swish",         "gelu", "glu",     "unknown", "fastgelu", "silu",          "geglu",  "swiglu",    "reglu"};
  if (id < 0 || id >= static_cast<int64_t>(activation_mode.size())) {
    MS_LOG(EXCEPTION) << "Invalid moe ffn activation " << id;
    return "";
  }
  return activation_mode[id];
}

REG_FALLBACK_BUILDER("FFNExt").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto weight1 = ib->GetInput(kIndex1);
  auto weight2 = ib->GetInput(kIndex2);
  auto expert_tokens = ib->GetInput(kIndex3);
  auto bias1 = ib->GetInput(kIndex4);
  auto bias2 = ib->GetInput(kIndex5);
  auto scale = ib->GetInput(kIndex6);
  auto offset = ib->GetInput(kIndex7);
  auto deq_scale1 = ib->GetInput(kIndex8);
  auto deq_scale2 = ib->GetInput(kIndex9);
  auto antiquant_scale1 = ib->GetInput(kIndex10);
  auto antiquant_scale2 = ib->GetInput(kIndex11);
  auto antiquant_offset1 = ib->GetInput(kIndex12);
  auto antiquant_offset2 = ib->GetInput(kIndex13);
  auto activation = ib->GetInput(kIndex14);
  auto activation_ptr = activation->BuildValue();
  auto activation_val = ops::GetValueWithCheck<int64_t>(activation_ptr);
  auto activation_string = FFNExtConvertEnumToString(activation_val);
  auto inner_precise = ib->GetInput(kIndex15);
  auto inner_precise_ptr = inner_precise->BuildValue();
  auto inner_precise_val = ops::GetValueWithCheck<int64_t>(inner_precise_ptr);
  auto out = ib->Emit(
    "FFN",
    {x, weight1, weight2, expert_tokens, bias1, bias2, scale, offset, deq_scale1, deq_scale2, antiquant_scale1,
     antiquant_scale2, antiquant_offset1, antiquant_offset2},
    {{"activation", MakeValue<string>(activation_string)}, {"inner_precise", MakeValue<int64_t>(inner_precise_val)}});
  return {out};
});

REG_FALLBACK_BUILDER("VmapUnstackAssign").SetBody(BODYFUNC(ib) {
  constexpr const size_t kInputLowerLimit = 4;
  auto &inputs = ib->GetInputs();
  if (inputs.size() < kInputLowerLimit) {
    MS_LOG(INFO) << "In VmapUnstackAssign, inputs size must be greater than 3, but got " << inputs.size() << ".";
    return {};
  }
  // The input format is: [stacked_parameter, param1, param2, ...(a batch of parameters), UMonad].
  auto params_size = inputs.size() - kIndex2;
  auto input_param = ib->Emit("Load", {inputs[kIndex0], inputs.back()});
  auto unstacked_para = ib->Emit("Unstack", {input_param}, {{"axis", MakeValue<int64_t>(0LL)}});

  NodePtrList res;
  res.reserve(params_size);
  for (size_t i = 0; i < params_size; i++) {
    auto getitem = ib->TupleGetItem(unstacked_para, i);
    auto result = ib->Emit("Assign", {inputs[i + kIndex1], getitem});
    (void)res.emplace_back(std::move(result));
  }
  auto res_tuple = ib->MakeTuple(res);
  // the VmapStackAssign's output shape is always [1], and dtype is kInt32.
  auto result = ib->Depend(ib->Tensor(std::vector<int>{0}, kInt32), res_tuple);
  return {result};
});

REG_FALLBACK_BUILDER("BinaryCrossEntropyWithLogitsBackward").SetBody(BODYFUNC(ib) {
  // dout, input, target, weight, posweight, reduction
  auto dout = ib->GetInput(kIndex0);
  auto predict = ib->GetInput(kIndex1);
  auto target = ib->GetInput(kIndex2);
  auto weight = ib->GetInput(kIndex3);
  auto pos_weight = ib->GetInput(kIndex4);
  if (ib->GetDtype(weight)->isa<TypeNone>()) {
    weight = ib->Emit("OnesLike", {target});
  }
  if (ib->GetDtype(pos_weight)->isa<TypeNone>()) {
    pos_weight = ib->Emit("OnesLike", {target});
  }
  auto reduction = ib->GetInput(kIndex5);
  auto sigmoid_input = ib->Emit("Sigmoid", {predict});
  auto t = ib->Mul(target, pos_weight);
  auto dx =
    ib->Mul(ib->Sub(ib->Mul(ib->Sub(ib->Add(t, ib->Tensor(1, ib->GetDtype(t))), target), sigmoid_input), t), dout);
  dx = ib->Mul(dx, weight);

  auto reduction_value = reduction->BuildValue();
  auto reduction_int_value = ops::GetScalarValue<int64_t>(reduction_value);
  if (reduction_int_value == Reduction::MEAN) {
    if (IsDynamic(ib->GetShape(dx))) {
      auto res = ib->DynSize(dx, ib->GetDtype(dx));
      dx = ib->RealDiv(dx, res);
    } else {
      dx = ib->RealDiv(dx, ib->Tensor(ib->GetSize(dx), ib->GetDtype(dx)));
    }
  }
  return {dx};
});

REG_FALLBACK_BUILDER("Contiguous").SetBody(BODYFUNC(ib) { return {ib->GetInput(kIndex0)}; });
}  // namespace expander
}  // namespace mindspore
