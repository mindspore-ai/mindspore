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
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_funcs.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"

namespace mindspore::opt {
bool ApplyAdagradV2PreCheck(const CNodePtr &node) {
  // check input var data type
  auto data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  if (data_type == kNumberTypeFloat32 || data_type == kNumberTypeFloat16) {
    return true;
  }
  return false;
}
}  // namespace mindspore::opt
