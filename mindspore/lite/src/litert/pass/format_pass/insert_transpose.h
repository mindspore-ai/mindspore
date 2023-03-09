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

#ifndef MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_INSERT_TRANSPOSE_H_
#define MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_INSERT_TRANSPOSE_H_

#include <vector>
#include <string>
#include "src/litert/pass/format_pass/format_pass.h"
#include "src/litert/pass/format_pass/pass_utils.h"
#include "src/litert/kernel_exec.h"

namespace mindspore::lite::pass {
class InsertTranspose : public FormatPass {
 public:
  explicit InsertTranspose(Format format) : FormatPass(format) {}
  virtual ~InsertTranspose() = default;
  int RunPass(kernel::SubGraphKernel *graph, std::vector<lite::Tensor *> *tensors);

 private:
  int TransposeConstData(kernel::KernelExec *kernel, size_t index);
};
}  // namespace mindspore::lite::pass
#endif  // MINDSPORE_LITE_SRC_LITERT_PASS_FORMAT_PASS_INSERT_TRANSPOSE_H_
