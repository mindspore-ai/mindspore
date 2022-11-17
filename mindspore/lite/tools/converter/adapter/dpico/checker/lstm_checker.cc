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

#include "checker/lstm_checker.h"
#include <string>
#include "common/op_attr.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr uint32_t kLstmMaxNumOutput = 5456;
}  // namespace

bool LstmChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto num_outputs_ptr = primitive->GetAttr(dpico::kNumOutput);
  if (num_outputs_ptr != nullptr) {
    auto num_outputs_data = static_cast<uint32_t>(api::GetValue<int64_t>(num_outputs_ptr));
    if (num_outputs_data > kLstmMaxNumOutput) {
      MS_LOG(WARNING) << "num_output should less than " << kLstmMaxNumOutput;
      return false;
    }
  }
  return true;
}

OpCheckerRegistrar g_LstmChecker("Lstm", new LstmChecker());
}  // namespace dpico
}  // namespace mindspore
