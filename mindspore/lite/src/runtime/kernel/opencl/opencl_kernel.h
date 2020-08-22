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
#include "include/errorcode.h"

namespace mindspore::kernel {

enum class OpenCLMemType { BUF, IMG };

struct OpenCLToFormatParameter {
  OpParameter op_parameter;
  schema::Format src_format{schema::Format_NHWC};
  schema::Format dst_format{schema::Format_NHWC4};
  OpenCLMemType out_mem_type{OpenCLMemType::IMG};
};

class OpenCLKernel : public LiteKernel {
 public:
  explicit OpenCLKernel(OpParameter *parameter, const std::vector<lite::tensor::Tensor *> &inputs,
                        const std::vector<lite::tensor::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs, nullptr, nullptr) {}

  virtual int Init() { return RET_ERROR; }
  virtual int Prepare() { return RET_ERROR; }
  virtual int InferShape() { return RET_ERROR; }
  virtual int ReSize() { return RET_ERROR; }
  virtual int Run() { return RET_ERROR; }
  virtual int GetImageSize(size_t idx, std::vector<size_t> *img_size) { return RET_ERROR; }
  virtual int GetGlobalSize(size_t idx, std::vector<size_t> *global_size) { return RET_ERROR; }
  virtual int GetLocalSize(size_t idx, const std::vector<size_t> &global_size, std::vector<size_t> *local_size) {
    return RET_ERROR;
  }
  OpenCLMemType GetMemType() { return out_mem_type_; }
  void SetMemType(OpenCLMemType mem_type) { out_mem_type_ = mem_type; }
  schema::Format GetInOriFormat() { return in_ori_format_; }
  schema::Format GetOutOriFormat() { return out_ori_format_; }

 protected:
  OpenCLMemType out_mem_type_{OpenCLMemType::IMG};
  schema::Format in_ori_format_{schema::Format_NHWC};
  schema::Format out_ori_format_{schema::Format_NHWC4};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
