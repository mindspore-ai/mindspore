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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_CHECKER_COMMON_CHECKER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_CHECKER_COMMON_CHECKER_H_

#include <string>
#include "checker/op_checker.h"

namespace mindspore {
namespace dpico {
class CommonChecker : public OpChecker {
 public:
  CommonChecker() : OpChecker("Common") {}
  ~CommonChecker() override = default;
  bool Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) override;
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_CHECKER_COMMON_CHECKER_H_
