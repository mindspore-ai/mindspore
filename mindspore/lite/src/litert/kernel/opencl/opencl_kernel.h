/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_OPENCL_KERNEL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_OPENCL_KERNEL_H_
#define MAX_PROFILING_TIME_MILLI_SECOND 10 * 1000  // 10 seconds
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <string>
#include <cfloat>
#include "src/litert/lite_kernel.h"
#include "src/litert/kernel_exec.h"
#include "include/errorcode.h"
#include "src/litert/kernel/gpu/opencl/opencl_runtime.h"
#include "mindspore/lite/src/litert/tensor_category.h"
#include "src/litert/kernel/opencl/utils.h"
#include "nnacl/resize_parameter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
constexpr int INPUT_TENSOR_SIZE_1 = 1;
constexpr int INPUT_TENSOR_SIZE_2 = 2;
constexpr int INPUT_TENSOR_SIZE_3 = 3;
constexpr int INPUT_TENSOR_SIZE_4 = 4;
constexpr int INPUT_TENSOR_SIZE_5 = 5;
constexpr int INPUT_TENSOR_SIZE_6 = 6;
constexpr int INPUT_TENSOR_SIZE_16 = 16;
constexpr int OUTPUT_TENSOR_SIZE_1 = 1;
constexpr int OUTPUT_TENSOR_SIZE_2 = 2;
constexpr int OUTPUT_TENSOR_SIZE_3 = 3;
constexpr int OUTPUT_TENSOR_SIZE_4 = 4;

struct OpenCLToFormatParameter {
  OpParameter op_parameter{};
  lite::opencl::MemType out_mem_type{lite::opencl::MemType::IMG};
};

struct OpenGLTexture2DToOpenCLParameter {
  OpParameter op_parameter{};
  lite::opencl::MemType out_mem_type{lite::opencl::MemType::IMG};
};

template <typename SrcT, typename DstT>
int Broadcast2GpuShape(const SrcT *src, int src_num, DstT *dst, int dsc_num) {
  if (src == nullptr || src_num <= 0 || dst == nullptr || dsc_num < src_num) {
    MS_LOG(WARNING) << "Broadcast2GpuShape invalid input";
    return RET_ERROR;
  }

  if (src_num == DIMENSION_1D) {  // 1 1 1 C
    dst[kNHWC_C] = src[DIMENSION_0D];
  } else if (src_num == DIMENSION_2D) {  // N 1 1 C
    dst[kNHWC_N] = src[DIMENSION_0D];
    dst[kNHWC_C] = src[DIMENSION_1D];
  } else if (src_num == DIMENSION_3D) {  // N 1 W C
    dst[kNHWC_N] = src[DIMENSION_0D];
    dst[kNHWC_W] = src[DIMENSION_1D];
    dst[kNHWC_C] = src[DIMENSION_2D];
  } else if (src_num == DIMENSION_4D) {  // N H W C
    dst[kNHWC_N] = src[DIMENSION_0D];
    dst[kNHWC_H] = src[DIMENSION_1D];
    dst[kNHWC_W] = src[DIMENSION_2D];
    dst[kNHWC_C] = src[DIMENSION_3D];
  } else if (src_num == DIMENSION_5D) {  // N D H W C
    dst[kNDHWC_N] = src[DIMENSION_0D];
    dst[kNDHWC_D] = src[DIMENSION_1D];
    dst[kNDHWC_H] = src[DIMENSION_2D];
    dst[kNDHWC_W] = src[DIMENSION_3D];
    dst[kNDHWC_C] = src[DIMENSION_4D];
  } else if (src_num > DIMENSION_5D) {
    MS_LOG(WARNING) << "GPU doesn't support ndim>=" << src_num;
    return RET_ERROR;
  }

  return RET_OK;
}

template <typename SrcT, typename DstT>
int Broadcast2GpuShape(const SrcT *src, int src_num, DstT *dst, int dsc_num, DstT default_value) {
  if (dst == nullptr || dsc_num <= 0) {
    MS_LOG(WARNING) << "Broadcast2GpuShape invalid input";
    return RET_ERROR;
  }
  for (int i = 0; i < dsc_num; ++i) {
    dst[i] = default_value;
  }
  if (src == nullptr || src_num <= 0) {
    return RET_OK;
  }

  return Broadcast2GpuShape(src, src_num, dst, dsc_num);
}

int CpuAxis2GpuAxis(size_t ndim, int cpu_axis, int *gpu_axis);

struct GpuTensorInfo {
  GpuTensorInfo() = default;
  explicit GpuTensorInfo(const lite::Tensor *tensor) {
    auto ocl_runtime_wrap_ = lite::opencl::OpenCLRuntimeInnerWrapper();
    if (tensor == nullptr) {
      return;
    }
    auto shape_ori = tensor->shape();
    NDim = shape_ori.size();
    std::vector<size_t> shape_gpu(DIMENSION_5D);
    (void)Broadcast2GpuShape(shape_ori.data(), NDim, shape_gpu.data(), DIMENSION_5D, (size_t)1);
    if (NDim == DIMENSION_5D) {
      N = shape_gpu[kNDHWC_N];
      D = shape_gpu[kNDHWC_D];
      H = shape_gpu[kNDHWC_H];
      W = shape_gpu[kNDHWC_W];
      C = shape_gpu[kNDHWC_C];
    } else {
      N = shape_gpu[kNHWC_N];
      H = shape_gpu[kNHWC_H];
      W = shape_gpu[kNHWC_W];
      C = shape_gpu[kNHWC_C];
    }

    MS_ASSERT(N > 0);
    MS_ASSERT(D > 0);
    MS_ASSERT(H > 0);
    MS_ASSERT(W > 0);
    MS_ASSERT(C > 0);
    Slice = UP_DIV(C, C4NUM);

    FLT_size = tensor->data_type() == kNumberTypeFloat16 ? sizeof(cl_half) : sizeof(cl_float);
    FLT4_size = FLT_size * C4NUM;
    if (W * Slice <= ocl_runtime_wrap_.GetInstance()->GetMaxImage2DWidth()) {
      height = N * D * H;
      width = W * Slice;
    } else {
      height = N * D * H * W;
      width = Slice;
      if (height > ocl_runtime_wrap_.GetInstance()->GetMaxImage2DHeight()) {
        height = -1;
        width = -1;
      }
    }

    ElementsNum = N * D * H * W * C;
    ElementsC4Num = N * D * H * W * Slice * C4NUM;
    OriginSize = ElementsNum * FLT_size;
    Image2DSize = height * width * FLT4_size;
  }

  static std::unique_ptr<GpuTensorInfo> CreateGpuTensorInfo(const lite::Tensor *tensor) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "CreateGpuTensorInfo func's input tensor is nullptr";
      return nullptr;
    }

    auto gpu_tensor = std::make_unique<GpuTensorInfo>();
    auto ocl_runtime_wrap_ = lite::opencl::OpenCLRuntimeInnerWrapper();

    auto shape_ori = tensor->shape();
    gpu_tensor->NDim = shape_ori.size();
    std::vector<size_t> shape_gpu(DIMENSION_5D);
    auto ret = Broadcast2GpuShape(shape_ori.data(), gpu_tensor->NDim, shape_gpu.data(), DIMENSION_5D, (size_t)1);
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "CreateGpuTensorInfo Broadcast2GpuShape failed";
      return nullptr;
    }

    if (gpu_tensor->NDim == DIMENSION_5D) {
      gpu_tensor->N = shape_gpu[kNDHWC_N];
      gpu_tensor->D = shape_gpu[kNDHWC_D];
      gpu_tensor->H = shape_gpu[kNDHWC_H];
      gpu_tensor->W = shape_gpu[kNDHWC_W];
      gpu_tensor->C = shape_gpu[kNDHWC_C];
    } else {
      gpu_tensor->N = shape_gpu[kNHWC_N];
      gpu_tensor->H = shape_gpu[kNHWC_H];
      gpu_tensor->W = shape_gpu[kNHWC_W];
      gpu_tensor->C = shape_gpu[kNHWC_C];
    }

    MS_ASSERT(gpu_tensor->N > 0);
    MS_ASSERT(gpu_tensor->D > 0);
    MS_ASSERT(gpu_tensor->H > 0);
    MS_ASSERT(gpu_tensor->W > 0);
    MS_ASSERT(gpu_tensor->C > 0);
    gpu_tensor->Slice = UP_DIV(gpu_tensor->C, C4NUM);

    gpu_tensor->FLT_size = tensor->data_type() == kNumberTypeFloat16 ? sizeof(cl_half) : sizeof(cl_float);
    gpu_tensor->FLT4_size = gpu_tensor->FLT_size * C4NUM;
    if (gpu_tensor->W * gpu_tensor->Slice <= ocl_runtime_wrap_.GetInstance()->GetMaxImage2DWidth()) {
      gpu_tensor->height = gpu_tensor->N * gpu_tensor->D * gpu_tensor->H;
      gpu_tensor->width = gpu_tensor->W * gpu_tensor->Slice;
    } else {
      gpu_tensor->height = gpu_tensor->N * gpu_tensor->D * gpu_tensor->H * gpu_tensor->W;
      gpu_tensor->width = gpu_tensor->Slice;
      if (gpu_tensor->height > ocl_runtime_wrap_.GetInstance()->GetMaxImage2DHeight()) {
        gpu_tensor->height = -1;
        gpu_tensor->width = -1;
      }
    }

    gpu_tensor->ElementsNum = gpu_tensor->N * gpu_tensor->D * gpu_tensor->H * gpu_tensor->W * gpu_tensor->C;
    gpu_tensor->ElementsC4Num =
      gpu_tensor->N * gpu_tensor->D * gpu_tensor->H * gpu_tensor->W * gpu_tensor->Slice * C4NUM;
    gpu_tensor->OriginSize = gpu_tensor->ElementsNum * gpu_tensor->FLT_size;
    gpu_tensor->Image2DSize = gpu_tensor->height * gpu_tensor->width * gpu_tensor->FLT4_size;

    return gpu_tensor;
  }

  size_t RowPitch() const {
    auto runtime_wrapper = lite::opencl::OpenCLRuntimeInnerWrapper();
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
    return static_cast<int>(no_neg_axis + C4NUM - NDim);
  }

  bool IsImageSizeValid() { return width > 0 && height > 0; }

  size_t N{1};
  size_t D{1};
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
  void AlignGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local);

  int Prepare() override { return RET_OK; }
  int PreProcess() override;
  int ReSize() override;
  int Run() override { return RET_ERROR; }
  int PostProcess() override {
    if (is_oversize_kernel_) {
      return FreeInWorkTensor();
    }
    return RET_OK;
  }

  bool MallocDataDone();
  std::string OpenCLKernelHeader();

  virtual int CheckSpecs();
  virtual int CheckSpecsWithoutShape() { return RET_OK; }
  virtual int InitWeights() { return RET_OK; }
  virtual int SetConstArgs() { return RET_OK; }
  virtual int SetGlobalLocal() { return RET_OK; }
  virtual int GetGlobalSize(size_t idx, std::vector<size_t> *global_size) { return RET_ERROR; }
  virtual int GetLocalSize(size_t idx, const std::vector<size_t> &global_size, std::vector<size_t> *local_size) {
    return RET_ERROR;
  }
  virtual std::vector<BaseTuningParameter> GenerateTuningParam();
  virtual int AssignTuningParam(const BaseTuningParameter &param);
  virtual int Tune();
  virtual int StoreConstData() { return RET_OK; }
  virtual std::string DumpCode() { return "No source code generated!"; }

  int GetImageSize(size_t idx, lite::opencl::ImageSize *img_size);
  void PrintOutput(int print_num = 10, const std::string &out_file = "");
  lite::opencl::MemType GetMemType() { return out_mem_type_; }
  void SetMemType(lite::opencl::MemType mem_type) { out_mem_type_ = mem_type; }
  OpParameter *GetParameter() { return op_parameter_; }
  virtual double GetProfilingTimeMs();
  virtual int InferShape();

 protected:
  void PrintShape(lite::Tensor *output_tensor);
  static std::set<size_t> GenerateLocalByGlobal(size_t global_i);

  virtual std::string Key() {
    std::string key = schema::EnumNamePrimitiveType(type());
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
  bool is_oversize_kernel_{false};

 private:
  lite::opencl::OpenCLRuntimeInnerWrapper ocl_runtime_wrap_;
  static inline std::map<std::string, BaseTuningParameter> tuned_param_cache_;
};

template <class T>
kernel::LiteKernel *OpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                        const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) T(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }

  auto ret = kernel->CheckSpecsWithoutShape();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(WARNING) << "Check " << opParameter->name_ << " specification Without shape failed!";
    delete kernel;
    return nullptr;
  }

  auto shape = outputs.front()->shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "don't infer shape yet!";
    return kernel;
  }
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "don't support output shape has zero.";
    return nullptr;
  }
  ret = kernel->CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(WARNING) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  ret = kernel->OpenCLKernel::CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(WARNING) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  ret = reinterpret_cast<OpenCLKernel *>(kernel)->StoreConstData();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(WARNING) << "Store " << opParameter->name_ << " const data failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_OPENCL_OPENCL_KERNEL_H_
