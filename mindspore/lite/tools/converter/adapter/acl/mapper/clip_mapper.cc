/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/clip_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"
#include "ops/array_ops.h"
namespace mindspore {
namespace lite {
namespace {
const size_t kNumInputIndex2 = 2;
const size_t kNumInputIndex3 = 3;
const size_t kNumInputSize3 = 3;
}  // namespace
STATUS ClipMapper::Mapper(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto prim = ops::GetOperator<ops::Clip>(cnode->input(0));
  CHECK_NULL_RETURN(prim);
  auto min_val = prim->get_min();
  auto max_val = prim->get_max();
  auto min_param = opt::BuildFloatValueParameterNode(func_graph, min_val, cnode->fullname_with_scope() + "_min");
  auto max_param = opt::BuildFloatValueParameterNode(func_graph, max_val, cnode->fullname_with_scope() + "_max");
  if (min_param == nullptr || max_param == nullptr) {
    MS_LOG(ERROR) << "Build parameter node failed.";
    return RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::ClipByValue>();
  CHECK_NULL_RETURN(dst_prim);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  CHECK_NULL_RETURN(value_node);
  value_node->set_value(dst_prim);

  auto inputs = cnode->inputs();
  const size_t input_size_with_min = 3;  // prim, data, min
  const size_t input_size_with_max = 4;  // prim, data, min, max
  auto value_ptr = prim->GetAttr("empty_input_index");
  std::vector<AnfNodePtr> new_inputs;
  if (value_ptr != nullptr && inputs.size() == kNumInputSize3 && GetValue<int>(value_ptr) == 1) {
    MS_LOG(INFO) << "empty input index: " << GetValue<int>(value_ptr);
    new_inputs = {inputs[0], inputs[1], min_param, inputs[kNumInputIndex2]};
    cnode->set_inputs(new_inputs);
    auto cast_int64_node_2 =
      NewCNode(cnode, prim::kPrimCast, {cnode->input(kNumInputIndex2), NewValueNode(TypeIdToType(kNumberTypeInt32))},
               cnode->input(1)->abstract(), cnode->fullname_with_scope() + "_input2_cast_int32");
    cnode->set_input(kNumInputIndex2, cast_int64_node_2);

    auto cast_int64_node_3 =
      NewCNode(cnode, prim::kPrimCast, {cnode->input(kNumInputIndex3), NewValueNode(TypeIdToType(kNumberTypeInt32))},
               cnode->input(1)->abstract(), cnode->fullname_with_scope() + "_input3_cast_int32");
    cnode->set_input(kNumInputIndex3, cast_int64_node_3);
  } else {
    MS_LOG(INFO) << "clip cnode inputs size: " << cnode->inputs().size();
    if (inputs.size() < input_size_with_min) {
      inputs.push_back(min_param);
    }
    if (inputs.size() < input_size_with_max) {
      inputs.push_back(max_param);
    }
    cnode->set_inputs(inputs);
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameClip, ClipMapper)
}  // namespace lite
}  // namespace mindspore
