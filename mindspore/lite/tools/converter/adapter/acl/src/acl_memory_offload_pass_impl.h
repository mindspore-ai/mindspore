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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MEM_OFFLOAD_PASS_IMPL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MEM_OFFLOAD_PASS_IMPL_H_

#include <memory>
#include "tools/converter/adapter/acl/src/acl_pass_impl.h"

namespace mindspore {
namespace opt {
class AclMemoryOffloadPassImpl : public AclPassImpl {
 public:
  explicit AclMemoryOffloadPassImpl(const std::shared_ptr<ConverterPara> &param) : AclPassImpl(param) {}
  ~AclMemoryOffloadPassImpl() = default;

  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  /* build func graph */
  STATUS BuildGraph(const FuncGraphPtr &func_graph) override;

 private:
  std::shared_ptr<mindspore::ops::Custom> CreateCustomPrim();
  FuncGraphPtr CreateSingleOpFuncGraph(const CNodePtr &cnode);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MEM_OFFLOAD_PASS_IMPL_H_
