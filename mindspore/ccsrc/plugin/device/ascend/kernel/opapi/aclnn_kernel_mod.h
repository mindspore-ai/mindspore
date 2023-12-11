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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include "ops/base_operator.h"
#include "ops/op_def.h"
#include "kernel/kernel.h"
#include "plugin/factory/ms_factory.h"
#include "include/common/utils/utils.h"
#include "runtime/pynative/op_runtime_info.h"
#include "transform/acl_ir/acl_convert.h"
#include "transform/acl_ir/op_api_exec.h"
#include "transform/acl_ir/op_api_util.h"

namespace mindspore {
namespace kernel {
using aclTensor = transform::aclTensor;
using aclOpExecutor = transform::aclOpExecutor;
using CallBackFunc = std::function<void()>;
using OpApiUtil = transform::OpApiUtil;

class EmptyKernelTensor {
 public:
  EmptyKernelTensor() { tensor_ = new KernelTensor(); }
  ~EmptyKernelTensor() { delete tensor_; }
  void set_dtype_id(TypeId dtype_id) { tensor_->set_dtype_id(dtype_id); }
  KernelTensor *get() const { return tensor_; }

 private:
  KernelTensor *tensor_;
};

class AclnnKernelMod : public KernelMod {
 public:
  explicit AclnnKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {}
  ~AclnnKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  virtual void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  }
  virtual bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  void ResetDeivceAddress(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {}

  void ParseGenExecutor(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args);

  bool IsNeedUpdateOutputShapeAndSize() override { return false; }
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in aclnn."; }

  void UpdateWorkspace(const std::tuple<uint64_t, aclOpExecutor *, CallBackFunc> &args);

  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace);
  void RunOpSync(void *stream_ptr, const std::vector<KernelTensor *> &workspace);

  void SetDTypes(const std::string &op_name);

 protected:
  template <size_t N, std::size_t... Is>
  auto GetTupleFrontImpl(const std::vector<KernelTensor *> &vecs, std::index_sequence<Is...>) {
    return std::make_tuple(vecs[Is]...);
  }

  template <size_t N>
  auto GetTupleFront(const std::vector<KernelTensor *> &vecs) {
    return GetTupleFrontImpl<N>(vecs, std::make_index_sequence<N>());
  }

  template <typename T, typename... Vecs>
  std::vector<T> ConcatVecs(const std::vector<T> &vec, const Vecs &... vecs) {
    std::vector<T> result = vec;
    (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
    return result;
  }

  template <typename T, typename... Vecs>
  std::vector<T> ConcatVecs(const Vecs &... vecs) {
    static_assert((std::is_same_v<T, typename Vecs::value_type> && ...), "All vectors must have the same type!");
    std::vector<T> result;
    (result.insert(result.end(), vecs.begin(), vecs.end()), ...);
    return result;
  }

  template <size_t N, typename... Ts>
  auto GetKernelTuple(const std::vector<Ts> &... vecs) {
    const auto &new_vec = ConcatVecs(vecs...);
    if (new_vec.size() != N) {
      MS_LOG(EXCEPTION) << op_type_ << "'s config op input and output's size must be same, but get " << N << " with "
                        << new_vec.size();
    }
    const auto &result = GetTupleFront<N>(new_vec);
    return result;
  }

  aclOpExecutor *executor_{nullptr};
  CallBackFunc release_func_{nullptr};
  std::vector<mindspore::ops::OP_DTYPE> inputs_dtypes_;
  std::vector<mindspore::ops::OP_DTYPE> outputs_dtypes_;
  std::string op_type_;
};

using AclnnKernelModPtr = std::shared_ptr<AclnnKernelMod>;
using AclnnKernelModPtrList = std::vector<AclnnKernelModPtr>;

#define REGISTER_ACLNN_CLASS(TYPE)                                                                          \
  template <size_t N>                                                                                       \
  class Aclnn##TYPE##KernelMod : public AclnnKernelMod {                                                    \
   public:                                                                                                  \
    explicit Aclnn##TYPE##KernelMod(std::string &&op_type) : AclnnKernelMod(std::move(op_type)) {}          \
    ~Aclnn##TYPE##KernelMod() = default;                                                                    \
    void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,                                        \
                          const std::vector<KernelTensor *> &outputs) override {                            \
      auto executor_info = GenExecutor(inputs, outputs);                                                    \
      this->UpdateWorkspace(executor_info);                                                                 \
    }                                                                                                       \
    bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,    \
                const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {                    \
      this->ParseGenExecutor(GenExecutor(inputs, outputs));                                                 \
      this->RunOp(stream_ptr, workspace);                                                                   \
      return true;                                                                                          \
    }                                                                                                       \
                                                                                                            \
   private:                                                                                                 \
    template <typename... Ts>                                                                               \
    auto GenExecutor(const std::vector<Ts> &... vecs) {                                                     \
      const auto &op_type = this->op_type_;                                                                 \
      const auto &res_tuple = this->GetKernelTuple<N>(vecs...);                                             \
      auto executor_info =                                                                                  \
        std::apply([&op_type](const auto &... args) { return GEN_EXECUTOR(op_type, args...); }, res_tuple); \
      return executor_info;                                                                                 \
    }                                                                                                       \
  };

#define MS_ACLLNN_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_KERNEL_FACTORY_REG(AclnnKernelMod, NAME, DERIVE_CLASS)
#define MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(NAME, TYPE, N)                    \
  REGISTER_ACLNN_CLASS(NAME)                                                  \
  static const KernelRegistrar<AclnnKernelMod> g_##NAME##_AclnnKernelMod_reg( \
    #NAME, []() { return std::make_shared<Aclnn##NAME##KernelMod<N>>(#TYPE); });
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACLNN_KERNEL_MOD_H_
