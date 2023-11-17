/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_PYBOOST_OP_REGISTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_PYBOOST_OP_REGISTER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include "kernel/pyboost/op_base.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
static const auto kOneScalar = std::make_shared<Int64Imm>(1);

template <typename T>
class BACKEND_EXPORT OpFactory {
 public:
  using OpCreator = std::function<std::shared_ptr<T>()>;
  static OpFactory<T> &Get();
  void Register(const std::string &name, const std::string &device, OpCreator &&func) {
    MS_LOG(DEBUG) << "Reg for op " << name << " on device " << device;
    auto ret = op_creator_.try_emplace(device, func);
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicate op creator for " << name << " on device " << device;
    }
  }

  std::shared_ptr<T> Create(const std::string &name, const std::string &device);

  bool IsRegistered(const std::string &device) const { return op_creator_.find(device) != op_creator_.end(); }

 private:
  OpFactory() = default;
  ~OpFactory() = default;
  DISABLE_COPY_AND_ASSIGN(OpFactory);
  std::map<std::string, OpCreator> op_creator_;
};

template <typename T>
class OpRegister {
 public:
  using OpCreator = std::function<std::shared_ptr<T>()>;
  OpRegister(const std::string &name, const std::string &device, OpCreator &&fun) {
    OpFactory<T>::Get().Register(name, device, std::move(fun));
  }
  ~OpRegister() = default;
};

#define MS_REG_PYBOOST_OP_REG(DEVICE, clazz)                                                \
  static_assert(std::is_base_of<Op, clazz>::value, " must be base of Op");                  \
  static const OpRegister<clazz> g_##clazz##DEVICE##_##_PyBoost_reg(#clazz, #DEVICE, []() { \
    auto op = std::make_shared<clazz##DEVICE>();                                            \
    op->set_device_context(PyBoostUtils::GetDeviceContext(#DEVICE));                        \
    return op;                                                                              \
  });

#define MS_REG_PYBOOST_OP(DEVICE, clazz) MS_REG_PYBOOST_OP_REG(DEVICE, clazz)

#define CREATE_PYBOOST_OP(NAME, DEVICE) \
  mindspore::kernel::pyboost::OpFactory<mindspore::kernel::pyboost::NAME>::Get().Create(#NAME, DEVICE);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_PYBOOST_OP_REGISTER_H_
