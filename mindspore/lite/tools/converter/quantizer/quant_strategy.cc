/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API

#include "tools/converter/quantizer/quant_strategy.h"
#include <set>
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::lite::quant {
bool QuantStrategy::CanOpFullQuantized(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_int8_ops,
                                       const std::set<PrimitivePtr> &skip_check_dtype_ops,
                                       const std::set<mindspore::ActivationType> &support_activation) {
  MS_CHECK_TRUE_RET(cnode != nullptr, false);
  // The return node does not need to be quantified.
  if (opt::CheckPrimitiveType(cnode, prim::kPrimReturn) || opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    return false;
  }
  auto type = NodePrimitiveType(cnode);
  if (!support_int8_ops.empty() && !CheckNodeInSet(cnode, support_int8_ops)) {
    MS_LOG(WARNING) << "node:" << cnode->fullname_with_scope() << " type:" << type << " will not quantify.";
    return false;
  }
  if (CheckNodeInSet(cnode, skip_check_dtype_ops)) {
    MS_LOG(INFO) << " type:" << type << " node name is" << cnode->fullname_with_scope()
                 << " not need to check data type.";
    return true;
  }
  TypeId type_id;
  auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fetch DataType from cnode failed.";
    return false;
  }

  bool is_data_type_fp32 = type_id == kNumberTypeFloat32;
  if (!is_data_type_fp32) {
    MS_LOG(WARNING) << " type:" << type << " node name is " << cnode->fullname_with_scope() << ", type_id " << type_id
                    << " is not float32 and will not be quantified.";
    return false;
  }

  // Check Activation
  if (!support_activation.empty() && opt::CheckPrimitiveType(cnode, prim::kPrimActivation)) {
    auto value_ptr = GetValueNode<PrimitivePtr>(cnode->input(0))->GetAttr(ops::kActivationType);
    if (value_ptr == nullptr) {
      return false;
    }
    auto activation = mindspore::ActivationType(GetValue<int64_t>(value_ptr));
    if (support_activation.find(activation) == support_activation.end()) {
      return false;
    }
  }
  return true;
}

bool QuantStrategy::IsSkipOp(const std::string &skip_node_name) {
  return !(skip_node_.find(skip_node_name) == skip_node_.end());
}
}  // namespace mindspore::lite::quant
