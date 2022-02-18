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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_REDUCE_BASE_CODER_H
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_REDUCE_BASE_CODER_H

#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore::lite::micro {
class ReduceBaseCoder : public OperatorCoder {
 public:
  ReduceBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ReduceBaseCoder() override = default;

  virtual int Init();

 private:
  int CheckInputsOutputs() const;
  int CheckParameters();

 protected:
  int axes_[MAX_SHAPE_SIZE]{};
  int num_axes_{0};
  int mode_{0};
  bool reduce_to_end_{false};

 protected:
  void CalculateTmpBufferSize();
  void CalculateInnerOuterSize();
  std::vector<size_t> buffer_sizes_;
  std::vector<int> outer_sizes_;
  std::vector<int> inner_sizes_;
  std::vector<int> axis_sizes_;
  int outer_size_{0};
  int inner_size_{0};
  int axis_size_{0};
  virtual int ReSize();
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_REDUCE_BASE_CODER_H
