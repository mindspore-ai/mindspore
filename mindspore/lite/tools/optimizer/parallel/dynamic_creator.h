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

#ifndef MINDSPORE_LITE_SRC_PASS_PARALLEL_DYNAMIC_CREATOR_H_
#define MINDSPORE_LITE_SRC_PASS_PARALLEL_DYNAMIC_CREATOR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "tools/optimizer/parallel/operator_info.h"

namespace mindspore {
namespace opt {
#define REGISTER(className)                                                            \
  OperatorInfoPtr objectCreator##className(std::string name, SplitStrategy strategy) { \
    return std::make_shared<className>(name, strategy);                                \
  }                                                                                    \
  RegisterAction className##Register(#className, (CreatFn)objectCreator##className);

typedef OperatorInfoPtr (*CreatFn)(const std::string &name, const SplitStrategy &strategy);

class DynCreator {
 public:
  ~DynCreator() = default;

  // create static singleton dyn_creator instance
  static DynCreator &Instance() {
    static DynCreator fac = DynCreator();
    return fac;
  }
  // register
  void Register(std::string name, CreatFn func) { (void)Function_map_.insert(std::make_pair(name, func)); }
  // creator
  OperatorInfoPtr Create(const std::string &name, const SplitStrategy &strategy) {
    auto iter = Function_map_.find(name);
    if (iter == Function_map_.end()) {
      MS_LOG(INFO) << name << " is not register yet";
      return nullptr;
    }
    return iter->second(name, strategy);
  }

 private:
  DynCreator() = default;
  std::map<std::string, CreatFn> Function_map_;
};

class RegisterAction {
 public:
  RegisterAction(const std::string &name, CreatFn creatfn) : name_(name) {
    DynCreator::Instance().Register(name, creatfn);
  }
  ~RegisterAction() = default;

 private:
  std::string name_;
};

OperatorInfoPtr OperatorInstance(const std::string &type_name, const std::string &orig_name,
                                 const SplitStrategy &strategy);

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_PASS_PARALLEL_DYNAMIC_CREATOR_H_
