/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
#define MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_

#include <vector>
#include "src/lite_kernel.h"

namespace mindspore::kernel {
class OpenCLKernel : public LiteKernel {
 public:
  explicit OpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                        const std::vector<lite::tensor::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs, nullptr, nullptr) {}

  virtual int Init() { return -1; }
  virtual int Prepare() { return -1; }
  virtual int InferShape() { return -1; }
  virtual int ReSize() { return -1; }
  virtual int Run() { return -1; }
  virtual int GetImageSize(size_t idx, std::vector<size_t>* img_size) { return -1; }
  virtual int GetGlobalSize(size_t idx, std::vector<size_t>* global_size) { return -1; }
  virtual int GetLocalSize(size_t idx, const std::vector<size_t>& global_size,
                           std::vector<size_t>* local_size) { return -1; }
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
