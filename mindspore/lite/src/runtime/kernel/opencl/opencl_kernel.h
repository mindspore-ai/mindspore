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

struct OpenCLToFormatParameter {
  OpParameter op_parameter{};
  schema::Format src_format{schema::Format::Format_NHWC};
  schema::Format dst_format{schema::Format::Format_NHWC4};
  lite::opencl::MemType out_mem_type{lite::opencl::MemType::IMG};
};

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(const SrcT *src, DstT *dst, int src_num) {
  auto *N = dst;
  auto *H = dst + 1;
  auto *W = dst + 2;
  auto *C = dst + 3;
  if (src_num == 1) {
    *N = src[0];
  } else if (src_num == 2) {
    *N = src[0];
    *C = src[1];
  } else if (src_num == 3) {
    *N = src[0];
    *W = src[1];
    *C = src[2];
  } else if (src_num == 4) {
    *N = src[0];
    *H = src[1];
    *W = src[2];
    *C = src[3];
  } else if (src_num >= 5) {
    MS_LOG(ERROR) << "GPU doesn't support ndim>=" << src_num;
  }
}

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(const SrcT *src, DstT *dst, int src_num, DstT default_value) {
  for (int i = 0; i < 4; ++i) {
    dst[i] = default_value;
  }
  Broadcast2GpuShape(src, dst, src_num);
}

struct Image2DInfo {
  explicit Image2DInfo(const lite::Tensor *tensor) {
    if (tensor == nullptr) {
      return;
    }
    auto shape = tensor->shape();
    OriDim = shape.size();
    if (OriDim == 1) {
      N = shape[0];
    } else if (OriDim == 2) {
      N = shape[0];
      C = shape[1];
    } else if (OriDim == 3) {
      N = shape[0];
      W = shape[1];
      C = shape[2];
    } else if (OriDim == 4) {
      N = shape[0];
      H = shape[1];
      W = shape[2];
      C = shape[3];
    } else if (OriDim >= 5) {
      MS_LOG(ERROR) << "GPU doesn't support Tensor with ndim>=" << OriDim;
    }
    Slice = UP_DIV(C, C4NUM);

    FLT_size = tensor->data_type() == kNumberTypeFloat16 ? sizeof(cl_half) : sizeof(cl_float);
    FLT4_size = FLT_size * 4;
    if (W * Slice <= MAX_IMAGE2D_SIZE) {
      height = N * H;
      width = W * Slice;
    } else {
      height = W;
      width = N * H * Slice;
    }

    ElementsNum = N * H * W * C;
    ElementsC4Num = N * H * W * Slice * C4NUM;
    OriginSize = ElementsNum * FLT_size;
    Image2DSize = height * width * FLT4_size;
  }

  size_t RowPitch() const {
    auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
    int alignment = runtime_wrapper.GetInstance()->GetImagePitchAlignment();
    size_t row_pitch = UP_ROUND(width, alignment) * FLT4_size;
    return row_pitch;
  }

  int AlignAxis(int oriAxis) const {
    if (OriDim == 0) {
      return 0;
    }
    int no_neg_axis = (oriAxis + OriDim) % OriDim;
    if (no_neg_axis == 0) {
      return 0;
    }
    return no_neg_axis + 4 - OriDim;
  }

  size_t N{1};
  size_t H{1};
  size_t W{1};
  size_t C{1};
  size_t Slice{};
  size_t width{};
  size_t height{};
  size_t FLT_size{4};
  size_t FLT4_size{16};
  size_t ElementsNum{};
  size_t ElementsC4Num{};
  size_t OriginSize{};
  size_t Image2DSize{};
  size_t OriDim{};
};

class OpenCLKernel : public LiteKernel {
 public:
  OpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs, nullptr, nullptr) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
  }
  ~OpenCLKernel() override = default;
  int AlignGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local) {
    std::vector<size_t> internal_global_ws = global;
    for (size_t i = 0; i < local.size(); ++i) {
      internal_global_ws[i] = UP_ROUND(global[i], local[i]);
    }

    MS_LOG(DEBUG) << "global size: " << global.size() << ", local size: " << local.size();
    for (size_t i = 0; i < global.size(); i++) {
      MS_LOG(DEBUG) << "global[" << i << "] = " << global[i];
    }
    for (size_t i = 0; i < local.size(); i++) {
      MS_LOG(DEBUG) << "local[" << i << "] = " << local[i];
    }

    if (global.size() == 1) {
      global_range_ = cl::NDRange(internal_global_ws[0]);
      if (!local.empty()) {
        local_range_ = cl::NDRange(local[0]);
      }
    } else if (global.size() == 2) {
      global_range_ = cl::NDRange(internal_global_ws[0], internal_global_ws[1]);
      if (!local.empty()) {
        local_range_ = cl::NDRange(local[0], local[1]);
      }
    } else if (global.size() == 3) {
      global_range_ = cl::NDRange(internal_global_ws[0], internal_global_ws[1], internal_global_ws[2]);
      if (!local.empty()) {
        local_range_ = cl::NDRange(local[0], local[1], local[2]);
      }
    } else {
      MS_LOG(ERROR) << "Not supported NDRange!";
      return RET_ERROR;
    }
    return RET_OK;
  }

  int Init() override { return RET_ERROR; }  // !!!To be deleted
  int Prepare() override { return RET_OK; }
  int PreProcess() override { return RET_ERROR; }
  int ReSize() override { return RET_ERROR; }
  int Run() override { return RET_ERROR; }

  virtual int CheckSpecs() { return RET_ERROR; }
  virtual int InitWeights() { return RET_OK; }
  virtual void SetConstArgs() {}
  virtual void SetGlobalLocal() {}
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

  lite::opencl::MemType GetMemType() { return out_mem_type_; }
  void SetMemType(lite::opencl::MemType mem_type) { out_mem_type_ = mem_type; }

 protected:
  lite::opencl::OpenCLRuntime *ocl_runtime_;
  lite::opencl::MemType out_mem_type_{lite::opencl::MemType::IMG};
  cl::NDRange global_range_{cl::NullRange};
  cl::NDRange local_range_{cl::NullRange};
  std::vector<size_t> global_size_;  // !!!To be deleted
  std::vector<size_t> local_size_;   // !!!To be deleted

 private:
  lite::opencl::OpenCLRuntimeWrapper ocl_runtime_wrap_;
};
template <class T>
kernel::LiteKernel *OpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                        const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                        const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) T(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Check " << opParameter->name_ << " specification failed!";
    return nullptr;
  }
  return kernel;
}
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
