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
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/pyboost/py_boost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT Op {
 public:
  Op() = default;
  virtual ~Op() = default;
  virtual void CastInput() = 0;
  void set_grad_func(const std::function<void()> &grad_func) { grad_func_ = grad_func; }
  void DoGrad() { grad_func_(); }
  void set_primitive(const PrimitivePtr &primitive) { primitive_ = primitive; }
  const PrimitivePtr &primitive() const { return primitive_; }

  const std::vector<tensor::TensorPtr> &outputs() const { return outputs_; }

  tensor::TensorPtr output(const size_t &idx) {
    if (idx >= outputs_.size()) {
      MS_LOG(EXCEPTION) << "idx is out of bounds, idx:" << idx << ", outputs_.size():" << outputs_.size();
    }
    return outputs_[idx];
  }

  const std::vector<AbstractBasePtr> &input_abs() const { return input_abs_; }
  const AbstractBasePtr &output_abs() const { return output_abs_; }

  template <typename... T>
  inline void InferOutput(T &... args) {
    (input_abs_.emplace_back(args->ToAbstract()), ...);
    auto eval_impl = abstract::GetPrimitiveInferImpl(primitive_);
    output_abs_ = eval_impl->InferShapeAndType(nullptr, primitive_, input_abs_);
    outputs_.clear();
    PyBoostUtils::CreateOutputTensor(output_abs_, &outputs_);
  }

 protected:
  std::vector<tensor::TensorPtr> outputs_;
  std::function<void()> grad_func_;
  PrimitivePtr primitive_;
  // Save abstract for grad.
  std::vector<AbstractBasePtr> input_abs_;
  AbstractBasePtr output_abs_;
};

template <typename T>
class BACKEND_EXPORT OpFactory {
 public:
  using OpCreater = std::function<std::shared_ptr<T>()>;
  static OpFactory<T> &Get();
  void Register(const std::string &name, const std::string &device, OpCreater &&func) {
    MS_LOG(DEBUG) << "Reg for op " << name << " on device " << device;
    auto ret = op_creater_.try_emplace(device, func);
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicate op creater for " << name << " on device " << device;
    }
  }

  std::shared_ptr<T> Create(const std::string &name, const std::string &device);

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
  OpRegister(const std::string &name, const std::string &device, OpCreater &&fun) {
    OpFactory<T>::Get().Register(name, device, std::move(fun));
  }
  ~OpRegister() = default;
};

#define MS_REG_PYBOOST_OP_REG(DEVICE, clazz)                               \
  static_assert(std::is_base_of<Op, clazz>::value, " must be base of Op"); \
  static const OpRegister<clazz> g_##clazz##DEVICE##_##_PyBoost_reg(       \
    #clazz, #DEVICE, []() { return std::make_shared<clazz##DEVICE>(); });

#define MS_REG_PYBOOST_OP(DEVICE, clazz) MS_REG_PYBOOST_OP_REG(DEVICE, clazz)

#define CREATE_PYBOOST_OP(NAME, DEVICE) \
  mindspore::kernel::pyboost::OpFactory<mindspore::kernel::pyboost::NAME>::Get().Create(#NAME, DEVICE);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_FORWARD_PYBOOST_OP_REGISTER_H_
