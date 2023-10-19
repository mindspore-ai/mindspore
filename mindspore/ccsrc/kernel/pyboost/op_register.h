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
#include <functional>
#include <memory>
#include <string>
#include "ir/scalar.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "ir/tensor.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT Op {
 public:
  Op() = default;
  virtual ~Op() = default;
  virtual void CastInput() = 0;
  void Grad() {}

  const std::vector<tensor::TensorPtr> &outputs() const { return outputs_; }

 protected:
  std::vector<tensor::TensorPtr> outputs_;
};

template <typename T>
class BACKEND_EXPORT OpFactory {
 public:
  using OpCreater = std::function<std::shared_ptr<T>()>;
  static OpFactory<T> &Get();
  void Register(const std::string &device, OpCreater &&func) {
    auto ret = op_creater_.try_emplace(device, func);
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicate op creater for " << device;
    }
  }

  std::shared_ptr<T> Create(const std::string &device);

  bool IsRegistered(const std::string &device) const { return op_creater_.find(device) != op_creater_.end(); }

 private:
  OpFactory() = default;
  ~OpFactory() = default;
  DISABLE_COPY_AND_ASSIGN(OpFactory);
  std::map<std::string, OpCreater> op_creater_;
};

template <typename T>
class OpRegister {
 public:
  using OpCreater = std::function<std::shared_ptr<T>()>;
  OpRegister(const std::string &device, OpCreater &&fun) { OpFactory<T>::Get().Register(device, std::move(fun)); }
  ~OpRegister() = default;
};

#define MS_REG_PYBOOST_OP_REG(DEVICE, clazz)                               \
  static_assert(std::is_base_of<Op, clazz>::value, " must be base of Op"); \
  static const OpRegister<clazz> g_##clazz##DEVICE##_##_PyBoost_reg(       \
    #DEVICE, []() { return std::make_shared<clazz##DEVICE>(); });

#define MS_REG_PYBOOST_OP(DEVICE, clazz) MS_REG_PYBOOST_OP_REG(DEVICE, clazz)
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_PYBOOST_OP_REGISTER_H_
