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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_

#include <vector>
#include <string>
#include "ir/anf.h"
#include "ir/func_graph.h"

namespace mindspore::pynative::autograd {
using TensorPtrList = tensor::TensorPtrList;
class Variable;
struct Edge {
  /// \brief Constructor.
  ///
  /// \param[in] variable The variable represents object need gradient.
  /// \param[in] input_index The input index is variable output index.
  explicit Edge(std::shared_ptr<Variable> variable, size_t input_index)
      : variable(std::move(variable)), input_index(input_index) {}
  std::shared_ptr<Variable> variable;
  size_t input_index;
};

class BackwardNode {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] name The name represents op name.
  /// \param[in] output_size The output_size is output size for op.
  explicit BackwardNode(string name, size_t output_size = 1) : name_(std::move(name)), output_size_(output_size) {}
  /// \brief Destructor.
  virtual ~BackwardNode() = default;
  /// \brief CallBackward function is used to calculate gradient of this node.
  ///
  /// \param[in] grads Grads is this node output's gradients.
  virtual TensorPtrList CallBackward(const TensorPtrList &grads) { return {}; }
  /// \brief Collect next edges of this node. The inputs should be flatten.
  /// \param[in] inputs Inputs is op input.
  virtual void UpdateNextEdges(const std::vector<ValuePtr> &inputs);
  /// \brief Postprocess gradients from func to align with next_edges.
  /// \param[in] gradient_value Gradients value is gradients result from func
  /// which need postprocess.
  /// \return Real gradients after postprocess, the size is same as next edges size.
  virtual TensorPtrList PostProcess(const ValuePtrList &gradient_value);
  /// \brief The PostProcess function is used to represent this node's inputs, which can
  /// backpropagation gradients.
  ///
  /// \return next edges
  const std::vector<Edge> &next_edges() const { return next_edges_; }
  /// \brief The gradient_index function is used to represent index of inputs,
  /// which need calculate gradient.
  ///
  /// \return gradient index
  const std::vector<size_t> &gradient_index() const { return gradient_index_; }
  /// \brief name of this Node.
  ///
  /// \return name
  const std::string &name() { return name_; }
  /// \brief The size of node output.
  ///
  /// \return output size
  size_t output_size() const { return output_size_; }

 protected:
  std::vector<Edge> next_edges_;
  std::vector<size_t> gradient_index_;
  std::string name_;
  size_t output_size_;
};
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

// Variable represent a parameter or output object of an op.
class Variable {
 public:
  /// \brief Constructor.
  ///
  Variable() = default;
  /// \brief Constructor.
  ///
  /// \param fn, Backward function.
  /// \param is_leaf, The variable is leaf or not.
  explicit Variable(BackwardNodePtr fn, bool is_leaf = false) : fn_(std::move(fn)), is_leaf_(is_leaf) {}
  /// \brief Backward function.
  ///
  /// \return fn
  const BackwardNodePtr &fn() const { return fn_; }
  /// \brief Gradients of the variable if variable is left node, nullptr if not left node.
  ///
  const tensor::TensorPtr &grad() const { return grad_; }
  /// \brief Set gradients of the leaf variable.
  ///
  /// \param grad
  void set_grad(const tensor::TensorPtr &grad) { grad_ = grad; }
  /// \brief Name of fake op.
  ///
  /// \return fake_prim_name
  const string &fake_prim_name() const { return fake_prim_name_; }
  /// \brief Set name of fake op.
  ///
  /// \param fake_prim_name
  void set_fake_prim_name(const string &fake_prim_name) { fake_prim_name_ = fake_prim_name; }
  /// \brief Flag to judge whether the op is fake op.
  ///
  bool is_fake_bprop() const { return is_fake_bprop_; }
  /// \brief Set fake bprop.
  ///
  /// \param is_fake_bprop
  void set_is_fake_bprop(bool is_fake_bprop) { is_fake_bprop_ = is_fake_bprop; }
  /// \brief Flag to judge whether the variable is need propagate.
  ///
  /// \return True if the variable need propagate, false if not.
  bool is_need_propagate() const { return is_need_propagate_; }
  /// \brief Set need propagate.
  ///
  void set_is_need_propagate(bool is_need_grad) { is_need_propagate_ = is_need_grad; }
  /// \brief Flag to judge whether the variable is need grad.
  ///
  /// \return is need grad
  bool is_need_grad() const { return is_need_grad_; }
  /// \brief Set need grad.
  ///
  /// \param is_need_grad
  void set_is_need_grad(bool is_need_grad) { is_need_grad_ = is_need_grad; }
  /// \brief Judge whether the variable is left node.
  ///
  /// \return True if variable is leaf, false if not.
  bool is_leaf() const { return is_leaf_; }
  /// \brief Debug info.
  ///
  /// \return debug info.
  std::string ToString();

 private:
  // Abstract bprop function
  BackwardNodePtr fn_;
  // Grad for this variable, only leaf node has grad.
  tensor::TensorPtr grad_;
  // If node has not bprop, we record its prim name
  std::string fake_prim_name_;
  // Record this node is a fake bprop
  bool is_fake_bprop_{false};
  // Flag to judge need to propagrate
  bool is_need_propagate_{false};
  // Flag to judge variable whether need grad
  bool is_need_grad_{true};
  // Flag the variable is a leaf in bprop.
  bool is_leaf_{false};
};
using VariablePtr = std::shared_ptr<Variable>;
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_
