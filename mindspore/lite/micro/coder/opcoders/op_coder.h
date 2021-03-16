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
#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODER_H_

#include <vector>
#include <set>
#include <string>
#include <memory>
#include "coder/context.h"
#include "coder/graph.h"
#include "coder/allocator/allocator.h"
#include "include/errorcode.h"
#include "src/lite_kernel.h"
#include "securec/include/securec.h"
#include "coder/opcoders/op_coder_register.h"
#include "coder/log.h"

namespace mindspore::lite::micro {
constexpr int kPrecision = 19;

class OperatorCoder {
 public:
  OperatorCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const Model::Node *node, size_t node_index, Target target)
      : input_tensors_(in_tensors),
        output_tensors_(out_tensors),
        target_(target),
        node_(node),
        node_index_(node_index) {
    allocator_ = MemoryAllocator::GetInstance();
    input_tensor_ = input_tensors_.at(kInputIndex);
    output_tensor_ = output_tensors_.at(kOutputIndex);
  }

  std::string name() const { return node_->name_; }

  void set_input_tensor_indices(const std::vector<uint32_t> &input_indices);
  void set_output_tensor_indices(const std::vector<uint32_t> &output_indices);

  const std::vector<uint32_t> input_tensor_indices() const;
  const std::vector<uint32_t> output_tensor_indices() const;

  const std::vector<Tensor *> input_tensors() const;
  const std::vector<Tensor *> output_tensors() const;

  void AddInputOp(OperatorCoder *op) { input_ops_.push_back(op); }
  void AddOutputOp(OperatorCoder *op) { output_ops_.push_back(op); }
  const std::vector<OperatorCoder *> input_ops() const { return input_ops_; }
  const std::vector<OperatorCoder *> output_ops() const { return output_ops_; }

  void set_type(int type) { type_ = type; }
  const int type() const { return type_; }

  size_t node_index() const;

  void set_parameter(OpParameter *parameter);

  const Model::Node *node() const { return this->node_; }

  void AddInitialParameters(Tensor *parameter) { initial_parameters_.push_back(parameter); }

  const std::vector<Tensor *> initial_parameters() const { return initial_parameters_; }

  // context
  virtual int Prepare(CoderContext *const context) = 0;

  virtual int DoCode(CoderContext *const context) = 0;

  virtual ~OperatorCoder();

  void set_thread_num(int thread_num);

 protected:
  std::vector<Tensor *> input_tensors_;
  std::vector<Tensor *> output_tensors_;
  Target target_{kTargetUnknown};
  const Model::Node *node_{nullptr};
  Tensor *input_tensor_{nullptr};
  Tensor *output_tensor_{nullptr};

  OpParameter *parameter_{nullptr};

  MemoryAllocator *allocator_{nullptr};

  bool support_parallel_{false};
  int thread_num_{1};

 private:
  size_t node_index_{0};
  std::vector<uint32_t> input_tensor_indices_;
  std::vector<uint32_t> output_tensor_indices_;

  std::vector<OperatorCoder *> input_ops_;
  std::vector<OperatorCoder *> output_ops_;
  std::vector<Tensor *> initial_parameters_;
  int type_{schema::PrimitiveType_NONE};
};

// a template func for normal op_coder creator
template <typename T>
std::unique_ptr<OperatorCoder> CPUOpCoderCreator(const std::vector<Tensor *> &in_tensors,
                                                 const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                                 size_t node_index, Target target) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is null";
    return nullptr;
  }
  std::unique_ptr<T> coder = std::make_unique<T>(in_tensors, out_tensors, node, node_index, target);
  return coder;
}
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODER_H_
