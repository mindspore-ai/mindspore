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

#include "tools/converter/adapter/acl/mapper/topk_fusion_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "src/common/log_util.h"
#include "ops/topk.h"
#include "ops/op_utils.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNumFlagThree = 3;
constexpr size_t kInputNumThree = 3;
constexpr size_t kInputNumTwo = 2;
}  // namespace

STATUS TopKFusionMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<acl::TopKV2>();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "TopKFusionMapper mapper failed.";
    return RET_ERROR;
  }
  auto topk_prim = ops::GetOperator<ops::TopKFusion>(cnode->input(0));
  auto largest_attr = topk_prim->GetAttr("largest");
  if (largest_attr != nullptr) {
    dst_prim->AddAttr("largest", MakeValue<bool>(topk_prim->get_largest() != 0));
  } else {
    MS_LOG(INFO) << "Current model does not have largest attr value";
  }

  auto inputs = cnode->inputs();
  if (inputs.size() != kInputNumThree && inputs.size() != kInputNumTwo) {
    MS_LOG(ERROR) << "Inputs num must be three or two, real num " << inputs.size();
    return RET_ERROR;
  }
  // convert last const parameter to value node
  if (inputs.size() == kInputNumThree) {
    auto k_input = cnode->input(kInputNumThree - 1);
    MS_CHECK_TRUE_MSG(k_input != nullptr, lite::RET_ERROR, "k_input is nullptr.");
    if (!utils::isa<ParameterPtr>(k_input)) {
      MS_LOG(ERROR) << "The k node is not parameter.";
      return lite::RET_ERROR;
    }
    ParameterPtr k_param = k_input->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(k_param != nullptr, lite::RET_ERROR, "ParameterPtr casts failed.");
    auto data = acl::GetIntParameterData(k_param);
    if (data.size() != 1) {
      MS_LOG(ERROR) << "The k node data size must be 1, but real size " << data.size();
      return RET_ERROR;
    }
    ValueNodePtr value_node = NewValueNode<int64_t>(static_cast<int64_t>(data[0]));
    MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_ERROR, "New value node failed.");
    std::vector<int64_t> shape_vec = {};
    auto abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec);
    value_node->set_abstract(abstract);
    cnode->set_input(kInputNumThree - 1, value_node);
    return lite::RET_OK;
  }
  // convert attr k to input
  auto attr_val = dst_prim->GetAttr("K");
  if (attr_val == nullptr) {
    MS_LOG(INFO) << "There is no attr k";
    return lite::RET_OK;
  }
  int64_t k_val;
  auto data_type = attr_val->type()->number_type();
  if (data_type == kNumberTypeInt64) {
    k_val = GetValue<int64_t>(attr_val);
  } else if (data_type == kNumberTypeInt || data_type == kNumberTypeInt32) {
    k_val = static_cast<int64_t>(GetValue<int32_t>(attr_val));
  } else {
    MS_LOG(ERROR) << "Not supported data type: " << static_cast<int64_t>(data_type);
    return RET_ERROR;
  }
  ValueNodePtr value_node = NewValueNode<int64_t>(k_val);
  MS_CHECK_TRUE_MSG(value_node != nullptr, lite::RET_ERROR, "New value node failed.");
  cnode->add_input(value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameTopKFusion, TopKFusionMapper)
}  // namespace lite
}  // namespace mindspore
