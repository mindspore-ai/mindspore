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

#define CODE_PARALLEL_FUNC(func) code << "ParallelLaunch(THREAD_POOL_DEFAULT, " << func << ", &args, thread_num);\n"

class OperatorCoder {
 public:
  OperatorCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const Model::Node *node, size_t node_index, Target target)
      : input_tensors_(in_tensors),
        output_tensors_(out_tensors),
        node_(node),
        target_(target),
        node_index_(node_index) {
    allocator_ = MemoryAllocator::GetInstance();
    // vectors checked not empty in OpCoderBuilder::build
    input_tensor_ = input_tensors_.at(kInputIndex);
    output_tensor_ = output_tensors_.at(kOutputIndex);
  }
  std::string ID() const { return node_->name_; }

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

  size_t node_index() const;

  void set_parameter(OpParameter *parameter);
  const PrimitiveC *primitive() const { return node_->primitive_; }

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
  const Model::Node *node_{nullptr};
  Target target_{kTargetUnknown};
  Tensor *input_tensor_{nullptr};
  Tensor *output_tensor_{nullptr};

  OpParameter *parameter_{nullptr};

  MemoryAllocator *allocator_{nullptr};

  std::string thread_num_s_{"1"};
  int thread_num_{1};

 private:
  size_t node_index_{0};
  std::vector<uint32_t> input_tensor_indices_;
  std::vector<uint32_t> output_tensor_indices_;

  std::vector<OperatorCoder *> input_ops_;
  std::vector<OperatorCoder *> output_ops_;
  std::vector<Tensor *> initial_parameters_;
};

// a template func for normal op_coder creator
template <typename T>
std::unique_ptr<OperatorCoder> CPUOpCoderCreator(const std::vector<Tensor *> &in_tensors,
                                                 const std::vector<Tensor *> &out_tensors, const Model::Node *node,
                                                 size_t node_index, Target target) {
  std::unique_ptr<T> coder = std::make_unique<T>(in_tensors, out_tensors, node, node_index, target);
  return coder;
}
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODER_H_
