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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_RESIZE_FP16_CODER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_RESIZE_FP16_CODER_H_

#include "coder/opcoders/nnacl/fp32/resize_fp32_coder.h"
#include <vector>
#include <algorithm>
#include <string>
#include "include/errorcode.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/kernel/cpu/fp16/resize_fp16.h"
#include "nnacl/base/cast_base.h"

namespace mindspore::lite::micro::nnacl {
class ResizeFP16Coder : public ResizeFP32Coder {
 public:
  ResizeFP16Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const LiteGraph::Node *node, size_t node_index, Target target)
      : ResizeFP32Coder(in_tensors, out_tensors, node, node_index, target) {}
  ~ResizeFP16Coder() override { FreeTmpBuffer(); };
  int DoCode(CoderContext *const context) override;

 private:
  int DataTypeLen() override;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_FP16_RESIZE_FP16_CODER_H_
