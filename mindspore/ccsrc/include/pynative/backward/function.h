/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <type_traits>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "abstract/abstract_value.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/utils/pynative/variable.h"

namespace mindspore::pynative::autograd {
using TensorPtr = tensor::TensorPtr;
using TensorPtrList = std::vector<TensorPtr>;

inline TensorPtrList ToTensorList(const TensorPtr &tensor) { return TensorPtrList{tensor}; }

inline TensorPtrList ToTensorList(const TensorPtrList &tensor_list) { return tensor_list; }

template <typename T>
std::enable_if_t<std::is_same_v<T, TensorPtrList>, T> ToOutputType(const TensorPtrList &tensor_list) {
  return tensor_list;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, TensorPtr>, T> ToOutputType(const TensorPtrList &tensor_list) {
  return tensor_list[0];
}

inline std::string GetFunctionTypeName(const char *name) {
#ifdef _MSC_VER
  return name;
#else
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

PYNATIVE_EXPORT void PrepareForForward();

template <typename X, typename... Args>
using forward_t = decltype(X::Forward(nullptr, std::declval<Args>()...));

template <class T>
struct Function {
  template <typename X = T, typename... Args>
  static auto Apply(Args &&...args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>>;
};

class SavedTensor;
using SavedTensorPtr = std::shared_ptr<SavedTensor>;
using SavedTensorPtrList = std::vector<SavedTensorPtr>;

struct PYNATIVE_EXPORT AutogradContext {
  AutogradContext() = default;

  /// \brief Save tensors for backward computation.
  ///
  /// This function saves a list of tensors that will be used later in the
  /// backward pass. These tensors will be retained in memory until the backward
  /// computation is finished.
  ///
  /// \param to_save A list of tensors to save.
  void SaveForBackward(const TensorPtrList &to_save) { to_save_ = std::move(to_save); }

  /// \brief Retrieve the tensors saved for backward computation.
  ///
  /// This function retrieves the list of tensors that were saved in the
  /// `SaveForBackward` method. These tensors can be used in the backward pass.
  ///
  /// \return The list of saved tensors.
  const TensorPtrList GetSavedTensors() const;

  /// \brief Mark input tensors as "dirty".
  ///
  /// Marking a tensor as "dirty" means that the forward computation has modified
  /// the contents of the tensor in-place. This informs the autograd engine that
  /// it should not rely on the original value of the tensor for gradient computation.
  ///
  /// \param inputs A list of tensors that are modified in-place.
  void MarkDirty(const TensorPtrList &inputs);

  /// \brief Mark output tensors as "non-differentiable".
  ///
  /// This function marks certain output tensors as not requiring gradients.
  /// This is useful for outputs that are detached from the computation graph.
  ///
  /// \param outputs A list of tensors that are non-differentiable.
  void MarkNonDifferentiable(const TensorPtrList &outputs);

  /// \brief Set whether to materialize gradients.
  ///
  /// When set to `false`, the autograd engine will not create zero-filled
  /// gradient tensors for inputs that do not require gradients. This can save
  /// memory in cases where gradients are not needed.
  ///
  /// \param value A boolean flag indicating whether to materialize gradients.
  void SetMaterializeGrads(bool value) { materialize_grads_ = value; }

  /// \brief Check if an input gradient is needed for a given output edge.
  ///
  /// This function determines whether the gradient for a specific tensor input
  /// is required by inspecting the backward graph. Only tensor inputs are
  /// considered, and the index corresponds to the position of tensors among
  /// all inputs (ignoring non-tensor inputs such as integers or other data types).
  /// For example, if the inputs are `[Tensor, int, Tensor]`, the valid indices
  /// for this function are 0 and 1 (corresponding to the two tensors).
  ///
  /// \param tensor_index The index of the tensor input (relative to tensors only).
  /// \return True if the gradient for the input tensor is required, false otherwise.
  bool NeedsInputGrad(size_t tensor_index) const;

  /// \brief Check if a tensor requires gradients.
  ///
  /// This function can be used in both the forward and backward passes to check
  /// whether a given tensor needs to be tracked for gradient computation.
  ///
  /// \param tensor A pointer to the tensor to check.
  /// \return True if the tensor requires gradients, false otherwise.
  bool NeedGrad(const TensorPtr &tensor);

  /// \brief A key-value store for saving arbitrary data during the forward pass.
  ///
  /// This map allows you to save custom data (e.g., scalars, configurations)
  /// during the forward computation that can be accessed during the backward
  /// computation. The data is stored as key-value pairs.
  std::unordered_map<std::string, ValuePtr> saved_data;

  void GenerateSavedNodes();

  friend PYNATIVE_EXPORT void CppFunctionDoGrad(AutogradContext *context, const TensorPtrList &inputs,
                                                TensorPtrList *outputs);

 private:
  TensorPtrList to_save_;
  std::unordered_set<TensorPtr> non_differentiable_;
  std::unordered_set<TensorPtr> dirty_inputs_;
  bool materialize_grads_{true};
  // This is used for avoid cycle reference.
  SavedTensorPtrList saved_tensors_;
  std::weak_ptr<BackwardNode> node_;

  template <class T>
  friend struct CppFunctionNode;
};

template <class T>
struct CppFunctionNode : public BackwardNode {
  explicit CppFunctionNode(string name, size_t output_size = 1) : BackwardNode(std::move(name), output_size) {}
  ~CppFunctionNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  void Release() override;
  void SetContextBackwardNode(const BackwardNodePtr &node) { context_.node_ = node; }
  void SetOutputSize(size_t output_size) { output_size_ = output_size; }
  AutogradContext context_;
  std::vector<bool> is_tensor_input_;
  abstract::AbstractBasePtrList outputs_abstract_;
};

template <class T>
template <typename X, typename... Args>
auto Function<T>::Apply(Args &&...args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>> {
  // create CppFunctionNode
  auto function_name = std::string("Function[") + GetFunctionTypeName(typeid(T).name()) + "]";
  MS_LOG(DEBUG) << function_name << " Begin Apply";
  auto node_ptr = std::make_shared<CppFunctionNode<T>>(function_name);
  // process function input
  TensorPtrList input_vars;
  auto check_is_base_tensor = [&input_vars, &node_ptr](auto &&arg) {
    using ArgType = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<ArgType, TensorPtr>) {
      arg->set_need_pipeline_sync(true);
      (void)input_vars.emplace_back(arg);
      node_ptr->is_tensor_input_.emplace_back(true);
    } else {
      node_ptr->is_tensor_input_.emplace_back(false);
    }
  };
  (check_is_base_tensor(std::forward<Args>(args)), ...);

  // prepare for forward
  PrepareForForward();

  // forward
  MS_LOG(DEBUG) << function_name << " Begin Forward";
  using forward_return_t = forward_t<X, Args...>;
  forward_return_t output;
  {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    output = T::Forward(&(node_ptr->context_), std::forward<Args>(args)...);
  }

  TensorPtrList output_list = ToTensorList(output);
  MS_EXCEPTION_IF_CHECK_FAIL(output_list.size() > 0, "The output list must not be empty.");
  AbstractBasePtrList outputs_abstract;
  outputs_abstract.reserve(output_list.size());
  for (size_t i = 0; i < output_list.size(); i++) {
    AbstractBasePtr abs = output_list[i]->ToAbstract();
    abs->set_value(kValueAny);
    (void)outputs_abstract.emplace_back(abs);
  }

  node_ptr->SetContextBackwardNode(node_ptr);
  node_ptr->outputs_abstract_ = outputs_abstract;
  node_ptr->SetOutputSize(output_list.size());

  // set autograd
  CppFunctionDoGrad(&(node_ptr->context_), input_vars, &output_list);
  return ToOutputType<forward_return_t>(output_list);
}

PYNATIVE_EXPORT TensorPtrList GradPreProcess(const ValuePtrList &grads, const AbstractBasePtrList &outputs_abstract,
                                             bool materialize_grads, const std::string &function_name);

PYNATIVE_EXPORT ValuePtrList GradPostProcess(const TensorPtrList &outputs, std::vector<bool> is_tensor_input,
                                             const std::string &function_name);

template <class T>
ValuePtrList CppFunctionNode<T>::CallBackward(const ValuePtrList &grads) {
  auto grad_in = GradPreProcess(grads, outputs_abstract_, context_.materialize_grads_, name_);
  auto grad_out = T::Backward(&context_, grad_in);
  return GradPostProcess(grad_out, is_tensor_input_, name_);
}

template <class T>
void CppFunctionNode<T>::Release() {
  context_.to_save_.clear();
  context_.saved_data.clear();
  context_.saved_tensors_.clear();
}
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
