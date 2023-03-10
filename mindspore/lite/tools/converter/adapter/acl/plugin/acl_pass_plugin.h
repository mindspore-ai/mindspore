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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_PLUGIN_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_PLUGIN_H_

#include <memory>
#include <string>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/optimizer_manager.h"

namespace mindspore {
namespace opt {
class AclPassPlugin {
 public:
  static std::shared_ptr<Pass> CreateAclPass(const std::shared_ptr<ConverterPara> &param);

 private:
  AclPassPlugin();
  ~AclPassPlugin();

  bool GetPluginSoPath();
  std::shared_ptr<Pass> CreateAclPassInner(const std::shared_ptr<ConverterPara> &param);

  typedef mindspore::opt::Pass *(*AclPassCreatorFunc)(const std::shared_ptr<ConverterPara> &);

  void *handle_;
  AclPassCreatorFunc creator_func_ = nullptr;
  static std::mutex mutex_;
  std::string real_path_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_ACL_PASS_PLUGIN_H_
