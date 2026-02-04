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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_backward_fusion.h"

#include <string>
#include <utility>

#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "ops/op_def_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/array_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_common_utils.h"

namespace mindspore::prim {
void CheckGmmBackwardFusionInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const std::string &op_name = primitive->name();
  CheckNumOfSequenceTensor(input_args.at(kIndex1), kIndex1, op_name, "weight");
  primitive->AddAttr("weight_num", MakeValue<int64_t>(1));
}

BeginFunction(GmmBackwardFusion, grad, weight, group_list, group_list_type) {
  auto ForEachTranspose = [&](const NodePtr &tensors, int64_t num) {
    std::vector<NodePtr> new_tensors;
    for (int64_t i = 0; i < num; ++i) {
      auto tensor_i = GetItem(tensors, Value(i));
      auto tensor_i_t = Call(Prim(TransposeExtView), tensor_i, Value(-1), Value(-2));
      new_tensors.push_back(std::move(tensor_i_t));
    }
    return MakeTuple(new_tensors);
  };

  auto Gmm = [&](const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                 int64_t group_type_value) -> NodePtr {
    auto split_item = Value(3);
    auto group_type = Value(group_type_value);
    auto none = NewValueNode(kNone);
    return Call(Prim(GroupedMatmulV2), x, weight, none, none, none, none, none, group_list, split_item, group_type);
  };

  const auto &primitive = prim();
  auto w_num = GetValue<int64_t>(primitive->GetAttr("weight_num"));

  auto wt = ForEachTranspose(weight, w_num);
  auto dx = Gmm(grad, wt, group_list, 0);

  Return(dx);
}
EndFunction(GmmBackwardFusion)
}  // namespace mindspore::prim
