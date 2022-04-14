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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_NODE_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_NODE_H_

#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <set>
#include "src/expression/export.h"
#include "inner/model_generated.h"
#include "src/expression/param.h"
#include "src/expression/expr.h"
#include "src/tensor.h"
#include "nnacl/op_base.h"

namespace mindspore {
class NodeImpl;
namespace schema {
struct TensorT;
struct CNodeT;
}  // namespace schema

namespace lite {
class Node {
 public:
  const std::string kGradName = "Gradients";
  explicit Node(const std::string name) : opParam_(nullptr), name_(name) { expr_.emplace_back(this); }
  virtual ~Node();
  Node() : Node("") {}
  explicit Node(Node *node) : Node(*node) {}
  EXPR *create(std::string name) {
    name_ = name;
    return &expr_[0];
  }
  virtual std::vector<EXPR *> operator()(const std::vector<EXPR *> &inputs) {
    auto x = construct(inputs);
    return x;
  }
  virtual std::vector<EXPR *> operator()(const std::initializer_list<EXPR *> &&inputs) {
    std::vector<EXPR *> vec = inputs;
    auto x = construct(vec);
    return x;
  }
  virtual std::vector<EXPR *> operator()(const std::initializer_list<EXPR *> &inputs) {
    std::vector<EXPR *> vec = inputs;
    auto x = construct(vec);
    return x;
  }
  void set_primitive(schema::PrimitiveType primitive) {
    primitive_ = primitive;
    if (OpParam() != nullptr) opParam_->type_ = primitive_;
  }
  schema::PrimitiveType primitive() { return primitive_; }
  virtual std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs);
  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }
  virtual void update_name(std::string name) { set_name(name + "/" + name_); }
  size_t Load(std::string file_name, size_t offset = 0) { return offset; }
  OpParameter *OpParam() const { return opParam_.get(); }
  virtual void Add(Node *node) {}
  virtual std::vector<EXPR *> Clone(EXPR *grad, EXPR *weight) { return {}; }
  void SetOpParam(std::shared_ptr<OpParameter> opParam) { opParam_ = opParam; }
  void SetOpParam(void *opParam) { opParam_.reset(reinterpret_cast<OpParameter *>(opParam), free); }
  static std::string UniqueName(const std::string &name) { return name + "-" + std::to_string(name_id++); }
  static std::string UniqueName(std::string &&name) { return name + "-" + std::to_string(name_id++); }
  template <typename T>
  int CloneOpParam(std::shared_ptr<OpParameter> opParam) {
    auto t = reinterpret_cast<T *>(opParam.get());
    auto obj = new (std::nothrow) T(*t);  // copy content
    if (obj == nullptr) {
      MS_LOG(ERROR) << "Cannot allocate obj";
      return RET_ERROR;
    }
    opParam_.reset(reinterpret_cast<OpParameter *>(obj));
    return RET_OK;
  }
  template <typename T>
  int CloneOpParam(OpParameter *opParam) {
    auto t = reinterpret_cast<T *>(opParam);
    auto obj = new (std::nothrow) T(*t);  // copy content
    if (obj == nullptr) {
      MS_LOG(ERROR) << "Cannot allocate obj";
      return RET_ERROR;
    }
    opParam_.reset(reinterpret_cast<OpParameter *>(obj));
    return RET_OK;
  }
  virtual Param *weight() { return nullptr; }
  EXPR *expr(int i) { return &expr_[i]; }
  EXPR *expr() { return expr(0); }
  std::vector<EXPR *> inputs() { return expr()[0].params(); }
  size_t InputsNum() { return expr()[0].params().size(); }
  size_t OutputsNum() { return expr_.size(); }
  EXPR *input(int idx) { return expr()[0].params().at(idx); }
  EXPR *output(int idx) { return expr(idx); }
  EXPR *CreateWeights(std::vector<int> dims, TypeId data_type, int format, Param::Mode mode, std::string name);
  Node *CreateConstTensor(int index, std::vector<int> dims, TypeId data_type, int format, std::string name,
                          const void *data);
  virtual std::vector<EXPR *> Grad(EXPR *expr);
  virtual Param *data() { return nullptr; }
  bool IsLearn(Node *node) { return learnable_.find(node) != learnable_.end(); }
  virtual void SetLearn() {}
  virtual std::set<Node *> trainable_params() { return learnable_; }
  std::vector<int> &dims() { return expr()->dims(); }
  std::vector<int> &dims(int i) { return expr(i)->dims(); }
  // export
  int MakeEntry(ExportSession *session);
  void PushOp(Node *n) { ops_.push_back(n); }
  virtual void AddNetOutput(std::vector<EXPR *> *output) {}
  int SetOutputs(int num);
  std::shared_ptr<OpParameter> opParam_;
  void set_impl(std::shared_ptr<NodeImpl> impl) { impl_ = impl; }

 protected:
  std::vector<EXPR> expr_;   // hold outputs
  std::vector<Node *> ops_;  // all nodes or subnets
  int InferShape();
  void AddLearn(Node *node) { learnable_.insert(node); }
  void AssignLearn(std::set<Node *> &&learn) { learnable_ = learn; }

  std::unique_ptr<schema::CNodeT> CreateCNode(std::vector<uint32_t> inputIndex, std::vector<uint32_t> outputIndex);
  virtual int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode);
  std::unique_ptr<schema::TensorT> CreateTensor(std::string name, int type, int data_type,
                                                const std::vector<int32_t> dims, int format,
                                                const std::vector<uint8_t> &data);

 private:
  int CreateTensorFromExpr(const std::vector<EXPR *> &expr, std::vector<Tensor *> *tensors, bool is_input = false);
  void FreeAllTensors(std::vector<Tensor *> *tensors);
  static int name_id;
  std::set<Node *> learnable_;  // set of nodes with learnable parameters
  std::string name_;
  schema::PrimitiveType primitive_;
  std::shared_ptr<NodeImpl> impl_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_NODE_H_
