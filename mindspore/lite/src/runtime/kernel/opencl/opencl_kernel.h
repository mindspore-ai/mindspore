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
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num) {
  MS_ASSERT(dst);
  MS_ASSERT(src);
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
  } else if (src_num > 4) {
    MS_LOG(ERROR) << "GPU doesn't support ndim>=" << src_num;
  }
}

template <typename SrcT, typename DstT>
void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num, DstT default_value) {
  MS_ASSERT(dst);
  MS_ASSERT(src);
  for (int i = 0; i < 4; ++i) {
    dst[i] = default_value;
  }
  Broadcast2GpuShape(dst, src, src_num);
}

struct GpuTensorInfo {
  explicit GpuTensorInfo(const lite::Tensor *tensor) {
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
    MS_ASSERT(alignment);
    size_t row_pitch = UP_ROUND(width, alignment) * FLT4_size;
    return row_pitch;
  }

  int AlignAxis(int oriAxis) const {
    if (NDim == 0) {
      return 0;
    }
    int no_neg_axis = static_cast<int>((oriAxis + NDim) % NDim);
    if (no_neg_axis == 0) {
      return 0;
    }
    return static_cast<int>(no_neg_axis + 4 - NDim);
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
               const std::vector<lite::Tensor *> &outputs)
      : LiteKernel(parameter, inputs, outputs, nullptr, nullptr) {
    ocl_runtime_ = ocl_runtime_wrap_.GetInstance();
  }
  ~OpenCLKernel() override = default;
  int AlignGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local) {
    std::vector<size_t> internal_global_ws = global;
    for (size_t i = 0; i < local.size(); ++i) {
      internal_global_ws.at(i) = UP_ROUND(global.at(i), local.at(i));
    }

    MS_LOG(DEBUG) << "global size: " << global.size() << ", local size: " << local.size();
    for (size_t i = 0; i < global.size(); i++) {
      MS_LOG(DEBUG) << "global[" << i << "] = " << global.at(i);
    }
    for (size_t i = 0; i < local.size(); i++) {
      MS_LOG(DEBUG) << "local[" << i << "] = " << local.at(i);
    }
    if (local.empty()) {
      local_range_ = cl::NullRange;
    }
    if (global.size() == 1) {
      global_range_ = cl::NDRange(internal_global_ws.at(0));
      if (!local.empty()) {
        local_range_ = cl::NDRange(local.at(0));
      }
    } else if (global.size() == 2) {
      global_range_ = cl::NDRange(internal_global_ws.at(0), internal_global_ws.at(1));
      if (!local.empty()) {
        local_range_ = cl::NDRange(local.at(0), local.at(1));
      }
    } else if (global.size() == 3) {
      global_range_ = cl::NDRange(internal_global_ws.at(0), internal_global_ws.at(1), internal_global_ws.at(2));
      if (!local.empty()) {
        local_range_ = cl::NDRange(local.at(0), local.at(1), local.at(2));
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
    MS_ASSERT(img_size);
    if (idx >= out_tensors_.size()) {
      return RET_ERROR;
    }
    auto img_info = GpuTensorInfo(out_tensors_[idx]);
    size_t img_dtype = ocl_runtime_->GetFp16Enable() ? CL_HALF_FLOAT : CL_FLOAT;
    *img_size = {img_info.width, img_info.height, img_dtype};
    return RET_OK;
  }

  lite::opencl::MemType GetMemType() { return out_mem_type_; }
  void SetMemType(lite::opencl::MemType mem_type) { out_mem_type_ = mem_type; }

  virtual std::vector<BaseTuningParameter> GenerateTuningParam() {
    size_t ndim = global_size_.size();
    std::vector<BaseTuningParameter> tuning_params = {};
    if (ndim == 0) {
      MS_LOG(ERROR) << "Generate tuning param failed, global_size_ is null.";
      return tuning_params;
    }
    BaseTuningParameter default_tuning_param = BaseTuningParameter();
    tuning_params.push_back(default_tuning_param);
    std::vector<size_t> max_work_items = ocl_runtime_->GetWorkItemSize();
    size_t max_workgroup_size = ocl_runtime_->GetMaxWorkGroupSize(kernel_);
    size_t MIN_WORKGROUP_SIZE = 8;
    std::set<size_t> candidate_x = GenerateLocalByGlobal(global_size_[0]);
    std::set<size_t> candidate_y = {1};
    std::set<size_t> candidate_z = {1};
    if (ndim > 1) {
      candidate_y = GenerateLocalByGlobal(global_size_[1]);
    }
    if (ndim > 2) {
      candidate_z = GenerateLocalByGlobal(global_size_[2]);
    }
    for (auto x : candidate_x) {
      if (x <= max_work_items[0]) {
        for (auto y : candidate_y) {
          if (y <= max_work_items[1]) {
            for (auto z : candidate_z) {
              auto group_size = x * y * z;
              if (z <= max_work_items[2] && group_size <= max_workgroup_size && group_size >= MIN_WORKGROUP_SIZE) {
                BaseTuningParameter tuning_param = BaseTuningParameter();
                tuning_param.local_size = {x, y, z};
                tuning_params.push_back(tuning_param);
              }
            }
          }
        }
      }
    }
    return tuning_params;
  }

  virtual int AssignTuningParam(const BaseTuningParameter param) {
    std::vector<size_t> local_size_tmp = param.local_size;
    if (local_size_tmp.size() > global_size_.size()) {
      local_size_tmp = std::vector<size_t>(local_size_tmp.begin(), local_size_tmp.begin() + global_size_.size());
    }
    AlignGlobalLocal(global_size_, local_size_tmp);
    return RET_OK;
  }

  virtual int Tune() {
    if (!ocl_runtime_->isProfiling()) {
      MS_LOG(WARNING) << "Tuning mode require opencl runtime profiling.";
      return RET_OK;
    }
    lite::opencl::TuningMode mode = ocl_runtime_->GetTuningMode();
    if (mode == lite::opencl::TuningMode::DEFAULT) {
      return RET_OK;
    }
    static const std::set<int> FAST_MODE_OPS = {schema::PrimitiveType_Conv2D, schema::PrimitiveType_DepthwiseConv2D,
                                                schema::PrimitiveType_DeConv2D};
    if (mode == lite::opencl::TuningMode::FAST && FAST_MODE_OPS.find(op_parameter_->type_) == FAST_MODE_OPS.end()) {
      return RET_OK;
    }
    auto tuning_params = GenerateTuningParam();
    if (tuning_params.empty()) {
      MS_LOG(WARNING) << "Tuning param size is 0.";
      return RET_OK;
    }
    int index = -1;
    double min_time = MAX_PROFILING_TIME_MILLI_SECOND;
    for (int i = 0; i < tuning_params.size(); i++) {
      AssignTuningParam(tuning_params[i]);
      auto ret = Run();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Tuning " << name() << " failed for tuning param " << tuning_params[i];
        return ret;
      }
      double current_time = GetProfilingTimeMs();
      MS_LOG(DEBUG) << "Tuning " << name() << " param (" << tuning_params[i] << ") exectime " << current_time << "ms";
      if (current_time < min_time) {
        min_time = current_time;
        index = i;
      }
    }
    if (index != -1) {
      MS_LOG(INFO) << "Tuning " << name() << " result: param (" << tuning_params[index] << ") exectime " << min_time
                   << "ms";
      AssignTuningParam(tuning_params[index]);
    } else {
      MS_LOG(WARNING) << "Cannot find suitable param.";
    }
    return RET_OK;
  }

  double GetProfilingTimeMs() {
    if (!ocl_runtime_->isProfiling()) {
      return MAX_PROFILING_TIME_MILLI_SECOND;
    }
    cl_ulong time_start;
    cl_ulong time_end;
    event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
    cl_ulong time_ns = time_end - time_start;
    return static_cast<double>(time_ns) * 1e-6;
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
  static std::set<size_t> GenerateLocalByGlobal(size_t global_i) {
    std::set<size_t> local_ = {};
    int index = 1;
    while (index <= global_i) {
      local_.insert(index);
      index *= 2;
    }
    for (size_t i = 1; i <= 16; i++) {
      if (global_i % i == 0) {
        local_.insert(i);
      }
    }
    return local_;
  }

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
    MS_LOG(ERROR) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_OPENCL_KERNEL_H_
