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

#include "kernel/ascend/aclnn/pyboost_impl/customize/gmm.h"
#include "kernel/ascend/aclnn/pyboost_impl/customize/gmm_v2.h"

#include <memory>
#include <functional>
#include <vector>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void CheckGroupTypeValue(const Int64ImmPtr &group_type_ptr, const std::string &op_name) {
  auto group_type = group_type_ptr->value();
  static std::set<int64_t> valid_group_type_list{0, 2};
  if (MS_UNLIKELY(valid_group_type_list.find(group_type) == valid_group_type_list.end())) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', group_type should be 0 or 2, but got " << group_type;
  }
}
}  // namespace

void GmmAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                        const ValueTuplePtr &weight_tensor_list, const std::optional<ValueTuplePtr> &bias_tensor_list,
                        const std::optional<ValueTuplePtr> &group_list, const Int64ImmPtr &group_type,
                        const Int64ImmPtr &group_list_type) {
  MS_LOG(DEBUG) << "GMM Func Op launch start.";
  CheckGroupTypeValue(group_type, "Gmm");
  static const auto split_item = std::make_shared<Int64Imm>(3);
  auto out = grouped_matmul_v2(x_tensor_list, weight_tensor_list, bias_tensor_list, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, group_list, split_item, group_type);

  op->set_outputs(out);
  MS_LOG(DEBUG) << "GMM Func Op launch end.";
}

void GmmV2AscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                          const ValueTuplePtr &weight_tensor_list, const std::optional<ValueTuplePtr> &bias_tensor_list,
                          const std::optional<TensorPtr> &group_list, const Int64ImmPtr &group_type,
                          const Int64ImmPtr &group_list_type) {
  MS_LOG(DEBUG) << "GMMV2 Func Op launch start.";
  CheckGroupTypeValue(group_type, "GmmV2");
  static const auto split_item = std::make_shared<Int64Imm>(3);
  static const auto act_type = std::make_shared<Int64Imm>(0);
  auto out = grouped_matmul_v4(x_tensor_list, weight_tensor_list, bias_tensor_list, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, group_list, std::nullopt, std::nullopt,
                               std::nullopt, split_item, group_type, group_list_type, act_type, std::nullopt);
  op->set_outputs(out);
  MS_LOG(DEBUG) << "GMMV2 Func Op launch end.";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
