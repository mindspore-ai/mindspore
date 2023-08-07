/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
#define REGISTER2(opName, className)                                                                      \
  OperatorInfoPtr objectCreator##opName(std::string name, Shapes in, Shapes out, PrimitiveAttrs &attrs) { \
    return std::make_shared<className>(name, in, out, attrs);                                             \
  }                                                                                                       \
  RegisterAction opName##Register(#opName, reinterpret_cast<CreatFn>(objectCreator##opName));

typedef OperatorInfoPtr (*CreatFn)(const std::string &name, const Shapes &shape_in, const Shapes shape_out,
                                   const PrimitiveAttrs &attrs);

#define REGISTER(className) REGISTER2(className, className)

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
  OperatorInfoPtr Create(const std::string &name, const Shapes &shape_in, const Shapes &shape_out,
                         const PrimitiveAttrs &attrs, size_t count) {
    std::string op_name = name + std::to_string(count);
    const auto iter = Function_map_.find(name);
    if (iter == Function_map_.end()) {
      MS_LOG(INFO) << name << " is not register yet";
      return nullptr;
    }
    return iter->second(op_name, shape_in, shape_out, attrs);
  }

 private:
  DynCreator() = default;
  std::map<std::string, CreatFn> Function_map_;
};

class RegisterAction {
 public:
  RegisterAction(const std::string &name, CreatFn creatfn) noexcept : name_(name) {
    DynCreator::Instance().Register(name, creatfn);
  }
  ~RegisterAction() = default;

 private:
  std::string name_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DYNAMIC_CREATOR_H_
