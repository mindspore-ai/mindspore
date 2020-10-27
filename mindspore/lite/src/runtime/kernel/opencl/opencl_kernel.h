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
#include "src/runtime/opencl/opencl_runtime.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

enum class OpenCLMemType { BUF, IMG };

struct OpenCLToFormatParameter {
  OpParameter op_parameter{};
  schema::Format src_format{schema::Format::Format_NHWC};
  schema::Format dst_format{schema::Format::Format_NHWC4};
  OpenCLMemType out_mem_type{OpenCLMemType::IMG};
};

struct Image2DInfo {
  explicit Image2DInfo(const lite::Tensor *tensor) {
    if (tensor) {
      auto shape = tensor->shape();
      if (shape.size() == 1) {
        N = shape[0];
      } else if (shape.size() == 2) {
        N = shape[0];
        C = shape[1];
      } else if (shape.size() == 3) {
        N = shape[0];
        W = shape[1];
        C = shape[2];
      } else if (shape.size() == 4) {
        N = shape[0];
        H = shape[1];
        W = shape[2];
        C = shape[3];
      } else if (shape.size() >= 5) {
        MS_LOG(ERROR) << "GPU dont't support Tensor with dim=" << shape.size();
      }
      FLT_size = tensor->data_type() == kNumberTypeFloat16 ? sizeof(cl_half) : sizeof(cl_float);
    } else {
      FLT_size = sizeof(cl_float);
    }

    FLT4_size = FLT_size * 4;
    Slice = UP_DIV(C, C4NUM);
    if (W * Slice <= MAX_IMAGE2D_SIZE) {
      height = N * H;
      width = W * Slice;
    } else {
      height = W;
      width = N * H * Slice;
    }

    auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
    int alignment = runtime_wrapper.GetInstance()->GetImagePitchAlignment();
    row_pitch = (width + alignment - 1) / alignment * alignment * FLT4_size;

    ElementsNum = N * H * W * C;
    ElementsC4Num = N * H * W * Slice * C4NUM;
    OriginSize = ElementsNum * FLT_size;
    Image2DSize = height * width * FLT4_size;
  }

  size_t N{1};
  size_t H{1};
  size_t W{1};
  size_t C{1};
  size_t Slice{};
  size_t width{};
  size_t height{};
  size_t FLT_size{};
  size_t FLT4_size{};
  size_t row_pitch{};
  size_t ElementsNum{};
  size_t ElementsC4Num{};
  size_t OriginSize{};
  size_t Image2DSize{};
};

class OpenCLKernel : public LiteKernel {
 public:
  OpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs, nullptr, nullptr) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
  }
  ~OpenCLKernel() override = default;

  int Init() override { return RET_ERROR; }
  int PreProcess() override { return RET_ERROR; }
  int ReSize() override { return RET_ERROR; }
  int Run() override { return RET_ERROR; }

  virtual int InitBuffer() { return RET_OK; }
  virtual int GetGlobalSize(size_t idx, std::vector<size_t> *global_size) { return RET_ERROR; }
  virtual int GetLocalSize(size_t idx, const std::vector<size_t> &global_size, std::vector<size_t> *local_size) {
    return RET_ERROR;
  }
  int GetImageSize(size_t idx, std::vector<size_t> *img_size) {
    if (idx >= out_tensors_.size()) {
      return RET_ERROR;
    }
    auto img_info = Image2DInfo(out_tensors_[idx]);
    size_t img_dtype = ocl_runtime_->GetFp16Enable() ? CL_HALF_FLOAT : CL_FLOAT;
    *img_size = {img_info.width, img_info.height, img_dtype};
    return RET_OK;
  }

  OpenCLMemType GetMemType() { return out_mem_type_; }
  void SetMemType(OpenCLMemType mem_type) { out_mem_type_ = mem_type; }

 protected:
  lite::opencl::OpenCLRuntime *ocl_runtime_;
  OpenCLMemType out_mem_type_{OpenCLMemType::IMG};
  std::vector<size_t> global_size_;
  std::vector<size_t> local_size_;

 private:
  lite::opencl::OpenCLRuntimeWrapper ocl_runtime_wrap_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
