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
#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_BASE_CODER_H
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_BASE_CODER_H
#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/fp32/strided_slice_fp32.h"
namespace mindspore::lite::micro {
class StridedSliceBaseCoder final : public OperatorCoder {
 public:
  StridedSliceBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                        const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~StridedSliceBaseCoder() override = default;

  int Prepare(CoderContext *context) override;

  int DoCode(CoderContext *context) override;

 private:
  int ReSize();
  int InitFastRunParam();
  bool MatchFastPattern();
  int DoNormalCode(CoderContext *ctx);
  int DoFastCode(CoderContext *ctx);

 private:
  StridedSliceParameter *strided_slice_parameter_{nullptr};
  int split_axis_{-1};
  int inner_{1};
  int outer_{1};
  int cal_num_per_thread_{1};
  size_t inner_size_{1};
  bool fast_run_{false};
  bool parallel_on_split_axis_{false};
  bool parallel_on_outer_{false};
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_STRIDED_SLICE_BASE_CODER_H
