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

#ifndef MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_ELIMINATE_TRANSPOSE_H_
#define MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_ELIMINATE_TRANSPOSE_H_

#include <vector>
#include "src/litert/pass/format_pass/format_pass.h"
#include "src/litert/pass/format_pass/transpose_strategy.h"

namespace mindspore::lite::pass {
class EliminateTranspose : public FormatPass {
 public:
  explicit EliminateTranspose(Format format) : FormatPass(format) {}
  virtual ~EliminateTranspose() = default;
  int RunPass(kernel::SubGraphKernel *graph, std::vector<lite::Tensor *> *tensors);

 private:
  int DoubleTransposeFusion(kernel::SubGraphKernel *subgraph);
  int EliminateForSingleKernel(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *all_tensors);
  int HorizontalTransposeFusionPass(kernel::SubGraphKernel *subgraph);
  int RepeteTransposeFusionPass(kernel::SubGraphKernel *subgraph, std::vector<lite::Tensor *> *all_tensors);

  TransposeStrategy transpose_strategy_;
  Format format_;
  int max_pass_count_ = 10;
  bool graph_changed_ = true;
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_ELIMINATE_TRANSPOSE_H_
