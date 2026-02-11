/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "ir/anf.h"
#include "ir/meta_grad_data.h"
#include "ir/tensor.h"
#include "utils/ordered_map.h"
#include "pynative/backward/hook/hook_py.h"
#include "include/utils/pynative/hook.h"

namespace mindspore::pynative::autograd {
class FuncBuilder;
struct GradAttr {
  GradAttr(bool get_all, bool get_by_list, bool sens_param, bool get_by_position, bool weight_param_is_tuple)
      : grad_all_inputs(get_all),
        grad_weights(get_by_list),
        has_sens(sens_param),
        get_by_position(get_by_position),
        weight_param_is_tuple(weight_param_is_tuple) {}

  bool grad_all_inputs;
  bool grad_weights;
  bool has_sens;
  bool get_by_position;
  bool weight_param_is_tuple;
};

class COMMON_EXPORT SavedNode {
 public:
  SavedNode() = default;
  SavedNode(ValuePtr data, std::shared_ptr<BackwardNode> grad_node, bool is_view_inplace, bool is_placeholder)
      : data_(std::move(data)),
        weak_grad_node_(grad_node),
        is_view_inplace_(is_view_inplace),
        is_placeholder_(is_placeholder) {}
  ValuePtr Unwrap(BackwardNodePtr grad_node, bool only_tensor = false);
  static std::shared_ptr<SavedNode> ConstructSavedNode(const ValuePtr &output, bool is_view_inplace = false);

 private:
  ValuePtr data_{nullptr};
  // Because when view inplace happen, inplace output grad node is not it's self,
  // so we need add weak reference for view output.
  std::weak_ptr<BackwardNode> weak_grad_node_{};
  bool is_view_inplace_{false};
  bool is_placeholder_{false};
};
using SavedNodePtr = std::shared_ptr<SavedNode>;

class COMMON_EXPORT TensorMeta {
 public:
  TensorMeta() : is_default_(true) {}
  TensorMeta(std::vector<int64_t> shape, TypePtr dtype, std::vector<int64_t> strides = {}, size_t storage_offset = 0)
      : shape_(std::move(shape)),
        dtype_(std::move(dtype)),
        strides_(std::move(strides)),
        storage_offset_(storage_offset) {}

  [[nodiscard]] const std::vector<int64_t> &shape() const { return shape_; }
  [[nodiscard]] const std::vector<int64_t> &strides() const { return strides_; }
  [[nodiscard]] const TypePtr &dtype() const { return dtype_; }
  [[nodiscard]] size_t storage_offset() const { return storage_offset_; }
  [[nodiscard]] size_t is_default() const { return is_default_; }
  [[nodiscard]] bool IsBroadcastTo(const ShapeVector &shape) const;
  [[nodiscard]] bool IsSameShape(const ShapeVector &shape) const;

 private:
  std::vector<int64_t> shape_{};
  TypePtr dtype_{nullptr};
  std::vector<int64_t> strides_{};
  size_t storage_offset_{0};
  bool is_default_{false};
};

class BackwardNode;
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

class COMMON_EXPORT AutoGradMetaData : public AutoGradMetaInterface {
 public:
  AutoGradMetaData() = default;
  explicit AutoGradMetaData(const InputType input_type) : input_type_(input_type) {}
  explicit AutoGradMetaData(BackwardNodePtr grad_node, const InputType input_type = InputType::kConstant)
      : grad_node_(std::move(grad_node)), input_type_(input_type) {}
  [[nodiscard]] const tensor::TensorPtr &grad() const override { return grad_; }
  void set_grad(const tensor::TensorPtr &grad) override { grad_ = grad; }
  [[nodiscard]] BackwardNodePtr UnsafeGetGradNodeImpl() const override { return grad_node_; }
  void set_grad_node(const BackwardNodePtr &grad_node) override { grad_node_ = grad_node; }
  [[nodiscard]] InputType input_type() const override { return input_type_; }
  void set_input_type(InputType input_type) override { input_type_ = input_type; }
  [[nodiscard]] size_t output_index() const override { return output_index_; }
  void set_output_index(size_t output_index) override { output_index_ = output_index; }
  [[nodiscard]] bool requires_grad() const override;
  void set_requires_grad(bool requires_grad) override { requires_grad_ = requires_grad; }
  [[nodiscard]] bool retains_grad() const override { return retains_grad_; }
  void set_retains_grad(bool retains_grad) override { retains_grad_ = retains_grad; }
  bool is_view() const override { return false; }
  ~AutoGradMetaData() override = default;

 private:
  tensor::TensorPtr grad_{nullptr};
  // grad_node for call grad fn.
  BackwardNodePtr grad_node_{nullptr};
  // Type of grad tensor
  InputType input_type_{InputType::kUnkown};
  // Index of op output tensors.
  size_t output_index_{0};
  // whether tensor requires grad.
  bool requires_grad_{false};
  // whether tensor retains grad.
  bool retains_grad_{false};
};

using AutoGradMetaDataPtr = std::shared_ptr<AutoGradMetaData>;
using Tensor = tensor::Tensor;
using TensorPtr = std::shared_ptr<tensor::Tensor>;
using TensorPtrSet = std::unordered_set<tensor::TensorPtr>;

class ViewInfo {
 public:
  explicit ViewInfo(TensorPtr base) : base_(std::move(base)) {}
  [[nodiscard]] ViewInfo Union() const { return ViewInfo(base_); }
  [[nodiscard]] const tensor::TensorPtr &base() const { return base_; }

 private:
  TensorPtr base_;
};

enum class CreationType {
  // View created in grad mode.
  kDefault = 0,
  // View created in no grad mode.
  kNoGradMode,
  // View created by multi output op.
  kMultiOutput,
  // View created by custom bprop.
  kCustomBprop,
};

class COMMON_EXPORT ViewAutoGradMetaData final : public AutoGradMetaData {
 public:
  ViewAutoGradMetaData(const ViewInfo &&view_info, InputType input_type,
                       CreationType creation_type = CreationType::kDefault)
      : AutoGradMetaData(input_type), view_info_(view_info), creation_type_(creation_type) {}
  [[nodiscard]] const ViewInfo &view_info() const { return view_info_; }
  [[nodiscard]] uint32_t version_attr() const { return version_attr_; }
  void set_version_attr(uint32_t version) { version_attr_ = version; }
  CreationType creation_type() { return creation_type_; }
  void set_creation_type(const CreationType &creation_type) { creation_type_ = creation_type; }
  [[nodiscard]] bool requires_grad() const override;
  bool is_view() const override { return true; }
  ~ViewAutoGradMetaData() override = default;

 private:
  ViewInfo view_info_;
  CreationType creation_type_;
  // We need set version attr in bprop queue to avoid multi thread race.
  uint32_t version_attr_{0};
};
using ViewAutoGradMetaDataPtr = std::shared_ptr<ViewAutoGradMetaData>;

struct Edge {
  /// \brief Constructor.
  /// \param[in] Grad node the grad node represents object need calculate gradient.
  /// \param[in] input_index The input index is variable output index.
  explicit Edge(BackwardNodePtr grad_node, size_t input_index)
      : grad_node(std::move(grad_node)), input_index(input_index) {}
  // Just a placeholder.
  Edge() : grad_node(nullptr), input_index(0) {}
  // Check edge is defined, if is defined, it mean that this edge is effective.
  // We need use undefined edge as placeholder, so that we can known operator input index exactly,
  // for example, when we use copy operator, we will knonw it has two tensor input, and next_edges[0] is self tensor.
  // so that when we use inplace op, we can skip self's edge and update other edges.
  [[nodiscard]] inline bool is_defined() const { return grad_node != nullptr; }
  BackwardNodePtr grad_node;
  size_t input_index;
};

using CppTensorHookList = std::vector<std::unique_ptr<CppTensorBackwardNodePreHook>>;
using PyTensorHookMap = OrderedMap<uint64_t, std::unique_ptr<PyTensorBackwardNodePreHook>>;
using RetainGradHookMap = std::unordered_map<size_t, std::unique_ptr<RetainGradHook>>;

class COMMON_EXPORT BackwardNode : public std::enable_shared_from_this<BackwardNode> {
 public:
  /// \brief Constructor.
  /// \param name
  /// \param output_size
  explicit BackwardNode(string name, size_t output_size = 1) noexcept;
  /// \brief Constructor.
  /// \param[in] name The name represents op name.
  /// \param[in] output_size The output_size is output size for op.
  explicit BackwardNode(string name, uint64_t seq_id, size_t output_size) noexcept;
  /// \brief Destructor.
  virtual ~BackwardNode() = default;
  DISABLE_COPY_AND_ASSIGN(BackwardNode);

  /// \brief CallBackward function is used to calculate gradient of this node.
  /// \param[in] grads Grads is this node output's gradients.
  virtual ValuePtrList CallBackward(const ValuePtrList &grads) { return {}; }

  /// \brief Is node a leaf.
  virtual bool IsLeaf() { return false; }

  /// \brief Postprocess gradients from func to align with next_edges.
  /// \param[in] gradient_value Gradients value is gradients result from func
  /// which need postprocess.
  /// \return Real gradients after postprocess, the size is same as next edges size.
  virtual ValuePtrList PostProcess(const ValuePtrList &gradient_value);

  /// \brief Add python tensor hook.
  /// \param id
  /// \param hook_fn
  void AddPyTensorHook(uint64_t id, std::unique_ptr<PyTensorBackwardNodePreHook> &&hook_fn) {
    if (!py_tensor_pre_hooks_) {
      py_tensor_pre_hooks_ = std::make_unique<PyTensorHookMap>();
    }
    (*py_tensor_pre_hooks_)[id] = std::move(hook_fn);
  }

  /// \brief Remove python tensor hook.
  /// \param id
  void RemovePyTensorHook(uint64_t id) { (void)py_tensor_pre_hooks_->erase(id); }

  /// \brief Add retain grad hook.
  /// \param output_idx
  /// \param hook_fn
  void AddRetainGradHook(size_t output_idx, std::unique_ptr<RetainGradHook> &&hook_fn) {
    if (!retain_grad_hooks_) {
      retain_grad_hooks_ = std::make_unique<RetainGradHookMap>();
    }
    (*retain_grad_hooks_)[output_idx] = std::move(hook_fn);
  }

  /// \brief Pop retain grad hook.
  /// \param output_idx
  /// \return Popped retain gradient hook.
  std::unique_ptr<RetainGradHook> PopRetainGradHook(size_t output_idx) {
    MS_EXCEPTION_IF_NULL(retain_grad_hooks_);
    auto hook_fn = std::move((*retain_grad_hooks_)[output_idx]);
    (void)retain_grad_hooks_->erase(output_idx);
    return hook_fn;
  }

  /// \brief Set Python Pre Hook
  /// \param hook_fn
  void SetPyPreHook(std::unique_ptr<PyBackwardNodePreHook> &&hook_fn) { py_pre_hook_ = std::move(hook_fn); }

  /// \brief Set Python Post Hook
  /// \param hook_fn
  void SetPyPostHook(std::unique_ptr<PyBackwardNodePostHook> &&hook_fn) { py_post_hook_ = std::move(hook_fn); }

  /// \brief Add cpp tensor hook.
  /// \param[in] hook
  /// \return hook index
  unsigned AddCppTensorHook(std::unique_ptr<CppTensorBackwardNodePreHook> &&hook);

  /// \brief Remove cpp tensor hook.
  /// \param[in] idx Cpp tensor hook id.
  void RemoveCppTensorHook(unsigned idx);

  /// check next edges is all not defined.
  /// \return true
  bool IsEmpty() const;

  /// \brief The PostProcess function is used to represent this node's inputs, which can
  /// backpropagation gradients. this interface can be modified.
  /// \return next edges
  std::vector<Edge> &mutable_next_edges() { return next_edges_; }
  /// \brief The PostProcess function is used to represent this node's inputs, which can
  /// backpropagation gradients.
  /// \return next edges
  const std::vector<Edge> &next_edges() const { return next_edges_; }

  /// \brief Set next edge for backward node.
  void set_next_edge(Edge &&edge, size_t i) { next_edges_[i] = std::move(edge); }

  /// \brief Set next edges for backward node.
  void set_next_edges(std::vector<Edge> &&next_edges) { next_edges_ = next_edges; }

  /// \brief Add next edges for backward node.
  void add_next_edge(Edge edge) { (void)next_edges_.emplace_back(std::move(edge)); }

  /// \brief return mutable metadata
  /// \return metadata
  std::vector<TensorMeta> &mutable_metadata() { return metadata_; }

  /// \brief add metadata of grad node.
  void add_output_metadata(const tensor::TensorPtr &output);
  /// \brief name of this Node.
  /// \return name
  const std::string &name() const { return name_; }

  /// \brief Unique id of this Node.
  /// \return unique id
  std::string UniqueId() const { return name_ + "-" + std::to_string(seq_id_); }

  /// \brief Cpp tensor pre hook for backward node.
  /// \return hook list
  const std::unique_ptr<CppTensorHookList> &cpp_tensor_pre_hooks() const { return cpp_tensor_pre_hooks_; }
  /// \brief Python tensor hook for backward node.
  /// \return py_tensor_pre_hooks
  const std::unique_ptr<PyTensorHookMap> &py_tensor_pre_hooks() const noexcept { return py_tensor_pre_hooks_; }

  /// \brief RetainGrad hook for backward node.
  /// \return retain_grad_hooks
  const std::unique_ptr<RetainGradHookMap> &retain_grad_hooks() const noexcept { return retain_grad_hooks_; }

  /// \brief Python pre hook for backward node.
  /// \return py_pre_hook
  const std::unique_ptr<PyBackwardNodePreHook> &py_pre_hook() const noexcept { return py_pre_hook_; }

  /// \brief Python post hook for backward node.
  /// \return py_post_hook
  const std::unique_ptr<PyBackwardNodePostHook> &py_post_hook() const noexcept { return py_post_hook_; }

  /// \brief The sequence number of current node.
  /// \return sequence number
  size_t seq_id() const { return seq_id_; }

  /// \brief The size of node output.
  /// \return output size
  size_t output_size() const { return output_size_; }

  /// \brief Release resource
  /// \return void
  virtual void Release() {}

  /// \brief Generate description str of node.
  /// \return string
  std::string ToString() const;

  /// \brief Create derived backward node with custom deleter.
  /// \return backward node
  template <typename Derived, typename... Args>
  static std::shared_ptr<Derived> Create(Args &&...args) {
    return std::shared_ptr<Derived>(new Derived(std::forward<Args>(args)...), CustomDeleter);
  }

  bool IsReleased() const { return is_released_; }

  void SetReleased(bool released) { is_released_ = released; }

 protected:
  static void CustomDeleter(BackwardNode *grad_node);
  std::vector<Edge> next_edges_;
  std::vector<TensorMeta> metadata_;
  std::string name_;
  size_t seq_id_;
  size_t output_size_;
  bool is_released_{false};

  // Tensor Hook -> RetainGrad Hook -> Capture Grad -> BackwardNode Pre Hook -> CallBackward -> BackwardNode Post Hook
  std::unique_ptr<PyTensorHookMap> py_tensor_pre_hooks_{nullptr};
  std::unique_ptr<RetainGradHookMap> retain_grad_hooks_{nullptr};
  std::unique_ptr<CppTensorHookList> cpp_tensor_pre_hooks_{};
  std::unique_ptr<PyBackwardNodePreHook> py_pre_hook_{nullptr};
  std::unique_ptr<PyBackwardNodePostHook> py_post_hook_{nullptr};
};
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

template <typename T>
bool isa(const BackwardNodePtr &base_ptr) {
  const auto &object = (*base_ptr);
  return typeid(object) == typeid(T);
}

template <typename T>
bool isa(const BackwardNode *base_ptr) {
  const auto &object = (*base_ptr);
  return typeid(object) == typeid(T);
}

class COMMON_EXPORT AutoDiffInterface {
 public:
  [[nodiscard]] virtual bool IsInExecGraph(const BackwardNodePtr &node) const = 0;
  virtual void AddNodeToExecGraph(const BackwardNodePtr &node) = 0;
  virtual size_t CurrentAutoDiffEngineId() = 0;
};
using AutoDiffInterfacePtr = std::shared_ptr<AutoDiffInterface>;

class COMMON_EXPORT AutoDiffGuard {
 public:
  explicit AutoDiffGuard(const AutoDiffInterfacePtr &auto_diff);
  ~AutoDiffGuard();

 private:
  AutoDiffInterfacePtr prev_auto_diff_engine_;
};

namespace impl {
COMMON_EXPORT void SetTensorGradMetaData(const TensorPtr &tensor, const BackwardNodePtr &grad_node, size_t index);
COMMON_EXPORT void SetVariable(const ValuePtrList &flatten_outs, const BackwardNodePtr &grad_node);
COMMON_EXPORT AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::Tensor &tensor);
COMMON_EXPORT ViewAutoGradMetaDataPtr GetViewAutogradMetaImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT BackwardNodePtr GetUnsafeGradNodeImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT bool RequiresGrad(const tensor::TensorPtr &tensor);
COMMON_EXPORT AutoDiffInterfacePtr CurrentAutoDiffEngine();
}  // namespace impl
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_
