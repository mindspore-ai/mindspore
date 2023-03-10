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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_H_

#define USE_DEPRECATED_API
#include <memory>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore {
namespace opt {
class AclPassImpl;
using AclPassImplPtr = std::shared_ptr<AclPassImpl>;

class AclPass : public Pass {
 public:
  explicit AclPass(const std::shared_ptr<ConverterPara> &param);
  ~AclPass() override = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  AclPassImplPtr impl_ = nullptr;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_H_
