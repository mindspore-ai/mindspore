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
#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"

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
}  // namespace expander
}  // namespace mindspore
