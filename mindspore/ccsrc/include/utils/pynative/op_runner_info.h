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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_OP_RUNNER_INFO_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_OP_RUNNER_INFO_H_

#include <vector>
#include <string>
#include "ir/anf.h"
#include "abstract/abstract_value.h"
#include "utils/simple_info.h"

namespace mindspore::runtime {
struct OpRunnerInfo {
  const PrimitivePtr &prim;
  device::DeviceType device_target;
  const std::vector<ValuePtr> &inputs;
  const abstract::AbstractBasePtrList &inputs_abs;
  const std::vector<InputType> &inputs_mask;
  abstract::AbstractBasePtr output_abs;
  ValueSimpleInfoPtr output_value_simple_info{nullptr};
};
}  // namespace mindspore::runtime
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_OP_RUNNER_INFO_H_
