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
#define MAX_PROFILING_TIME_MILLI_SECOND 10 * 1000  // 10 seconds

#include <vector>
#include <set>
#include <map>
#include <string>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "src/runtime/gpu/opencl/opencl_runtime.h"
#include "mindspore/lite/src/dequant.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/resize_parameter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

struct OpenCLToFormatParameter {
  OpParameter op_parameter{};
  lite::opencl::MemType out_mem_type{lite::opencl::MemType::IMG};
};

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num) {
  MS_ASSERT(dst);
  if (src == nullptr || src_num <= 0) {
    return;
  }
  auto *N = dst;
  auto *H = dst + 1;
  auto *W = dst + 2;
  auto *C = dst + 3;
  if (src_num == 1) {
    *C = src[0];
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
  } else if (src_num > 4) {
    MS_LOG(ERROR) << "GPU doesn't support ndim>=" << src_num;
  }
}

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num, DstT default_value) {
  MS_ASSERT(dst);
  for (int i = 0; i < 4; ++i) {
    dst[i] = default_value;
  }
  if (src == nullptr || src_num <= 0) {
    return;
  }
  Broadcast2GpuShape(dst, src, src_num);
}

struct GpuTensorInfo {
  GpuTensorInfo() = default;
  explicit GpuTensorInfo(const lite::Tensor *tensor) {
    auto ocl_runtime_wrap_ = lite::opencl::OpenCLRuntimeWrapper();
    if (tensor == nullptr) {
      return;
    }
    auto shape_ori = tensor->shape();
    NDim = shape_ori.size();
    cl_int4 shape;
    Broadcast2GpuShape(shape.s, shape_ori.data(), shape_ori.size(), 1);
    N = shape.s[0];
    H = shape.s[1];
    W = shape.s[2];
    C = shape.s[3];
    MS_ASSERT(N > 0);
    MS_ASSERT(H > 0);
    MS_ASSERT(W > 0);
    MS_ASSERT(C > 0);
    Slice = UP_DIV(C, C4NUM);

    FLT_size = tensor->data_type() == kNumberTypeFloat16 ? sizeof(cl_half) : sizeof(cl_float);
    FLT4_size = FLT_size * 4;
    if (W * Slice <= ocl_runtime_wrap_.GetInstance()->GetMaxImage2DWidth()) {
      height = N * H;
      width = W * Slice;
    } else {
      height = N * H * W;
      width = Slice;
      if (height > ocl_runtime_wrap_.GetInstance()->GetMaxImage2DHeight()) {
        height = -1;
        width = -1;
      }
    }

    ElementsNum = N * H * W * C;
    ElementsC4Num = N * H * W * Slice * C4NUM;
    OriginSize = ElementsNum * FLT_size;
    Image2DSize = height * width * FLT4_size;
  }

  size_t RowPitch() const {
    auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
    int alignment = runtime_wrapper.GetInstance()->GetImagePitchAlignment();
    MS_ASSERT(alignment);
    size_t row_pitch = UP_ROUND(width, alignment) * FLT4_size;
    return row_pitch;
  }

  int AlignAxis(int oriAxis) const {
    if (NDim == 0 || NDim == 1) {
      return 3;
    }
    int no_neg_axis = static_cast<int>((oriAxis + NDim) % NDim);
    if (no_neg_axis == 0) {
      return 0;
    }
    return static_cast<int>(no_neg_axis + 4 - NDim);
  }

  bool IsImageSizeValid() { return width > 0 && height > 0; }

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
  size_t NDim{};
};

struct BaseTuningParameter {
  std::vector<size_t> local_size;
  friend std::ostream &operator<<(std::ostream &ostrm, const BaseTuningParameter &a) {
    ostrm << "LocalSize:";
    for (auto i : a.local_size) {
      ostrm << i << ",";
    }
    return ostrm;
  }
};
class OpenCLKernel : public LiteKernel {
 public:
  OpenCLKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
  }
  ~OpenCLKernel() override = default;
  int AlignGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local);

  int Prepare() override { return RET_OK; }
  int PreProcess() override;
  int PostProcess() override;
  int ReSize() override;
  int Run() override { return RET_ERROR; }

  virtual int CheckSpecs();
  virtual int InitWeights() { return RET_OK; }
  virtual void SetConstArgs() {}
  virtual void SetGlobalLocal() {}
  virtual int GetGlobalSize(size_t idx, std::vector<size_t> *global_size) { return RET_ERROR; }
  virtual int GetLocalSize(size_t idx, const std::vector<size_t> &global_size, std::vector<size_t> *local_size) {
    return RET_ERROR;
  }
  virtual std::vector<BaseTuningParameter> GenerateTuningParam();
  virtual int AssignTuningParam(const BaseTuningParameter &param);
  virtual int Tune();

  int GetImageSize(size_t idx, lite::opencl::ImageSize *img_size);
  void PrintOutput(int print_num = 10, const std::string &out_file = "");
  lite::opencl::MemType GetMemType() { return out_mem_type_; }
  void SetMemType(lite::opencl::MemType mem_type) { out_mem_type_ = mem_type; }
  OpParameter *GetParameter() { return op_parameter_; }
  virtual double GetProfilingTimeMs();
  int DequantWeight();
  void FreeDequantedWeight();
  virtual int InferShape();

 protected:
  static std::set<size_t> GenerateLocalByGlobal(size_t global_i);

  virtual std::string Key() {
    std::string key = type_str();
    key += "_global";
    for (auto i : global_size_) {
      key += "_" + std::to_string(i);
    }
    return key;
  }

 protected:
  lite::opencl::OpenCLRuntime *ocl_runtime_;
  lite::opencl::MemType out_mem_type_{lite::opencl::MemType::IMG};
  cl::NDRange global_range_{cl::NullRange};
  cl::NDRange local_range_{cl::NullRange};
  std::vector<size_t> global_size_;
  std::vector<size_t> local_size_;
  cl::Kernel kernel_;
  cl::Event event_;
  void *restore_quant_data_{nullptr};
  bool dequant_flag_{false};

 private:
  lite::opencl::OpenCLRuntimeWrapper ocl_runtime_wrap_;
  static inline std::map<std::string, BaseTuningParameter> tuned_param_cache_;
};
template <class T>
kernel::LiteKernel *OpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                        const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) T(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  if (!opParameter->infer_flag_) {
    MS_LOG(WARNING) << "kernel don't infer shape yet!";
    return kernel;
  }
  auto ret = kernel->CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  ret = kernel->OpenCLKernel::CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
