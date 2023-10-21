/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <set>
#include <string>
#include <map>
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/opencl/kernel/matmul.h"
#include "src/litert/kernel/opencl/kernel/strassen.h"
#include "src/litert/kernel/opencl/cl/matmul.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
bool IsUseStrassenMatmul(const std::vector<lite::Tensor *> &in_tensors_) {
  if (in_tensors_.at(0)->shape().size() == DIMENSION_2D) {
    auto shape0 = in_tensors_.at(0)->shape();
    auto shape1 = in_tensors_.at(1)->shape();
    if (in_tensors_.at(1)->IsConst() && (shape0[0] == shape0[1]) && (shape1[0] == shape1[1]) &&
        (shape0[0] == shape1[0]) && (shape0[0] % 8 == 0)) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

int MatMulOpenCLKernel::CheckSpecs() {
  if (!(in_tensors_.size() == INPUT_TENSOR_SIZE_2 || in_tensors_.size() == INPUT_TENSOR_SIZE_3) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeA = param->a_transpose_;
  if (transposeA) {
    MS_LOG(WARNING) << "matmul only support a_transpose_=false yet.";
    return RET_ERROR;
  }
  transposeB = param->b_transpose_;

  act_weight_ = !in_tensors_.at(1)->IsConst();
  bool is_const = in_tensors_.at(1)->category() == lite::Category::CONST_TENSOR ||
                  in_tensors_.at(1)->category() == lite::Category::CONST_SCALAR;
  if (is_const && stored_weight_) {
    act_weight_ = false;
  }

  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  if (in_tensors_[0]->shape().size() != out_tensors_[0]->shape().size() ||
      in_tensors_[0]->shape().size() < DIMENSION_2D || in_tensors_[0]->shape().size() > DIMENSION_4D) {
    MS_LOG(WARNING) << "matmul only support input shape size= 2, 3 or 4.";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulOpenCLKernel::Prepare() {
  std::string kernel_name = "MatMul";
  if (act_weight_) {
    if (transposeB) {
      kernel_name = "MatMulActWeightTransposeB";
    } else {
      kernel_name = "MatMulActWeight";
    }
  }
  dims = in_tensors_[0]->shape().size();
  for (int i = 0; i < dims; i++) {
    inShape[MAX_DIMS - dims + i] = in_tensors_[0]->shape()[i];
    outShape[MAX_DIMS - dims + i] = out_tensors_[0]->shape()[i];
  }
  std::map<int, std::string> dims2str = {{2, "_2d"}, {3, "_4d"}, {4, "_4d"}};
  kernel_name += dims2str[dims];
  std::string source = GetActDefines() + matmul_source;
  const std::string program_name = "MatMul";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  (void)SetGlobalLocal();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

#ifdef ENABLE_FP16
int MatMulOpenCLKernel::PadWeight(std::vector<int> weight_shape_4d, int ci, int co) {
  auto allocator = ocl_runtime_->GetAllocator();
  int a = weight_shape_4d[0];
  int b = weight_shape_4d[1];
  int ci4 = UP_DIV(ci, C4NUM);
  int co4 = UP_DIV(co, C4NUM);
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  padWeight_ = allocator->Malloc(a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size, lite::opencl::MemType::BUF);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  auto padWeightFp32 = reinterpret_cast<float *>(padWeight_);
  auto padWeightFp16 = reinterpret_cast<float16_t *>(padWeight_);
  memset(padWeight_, 0x00, a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  void *src_data = stored_weight_ == nullptr ? in_tensors_.at(kWeightIndex)->data() : stored_weight_;
  auto originWeightFp32 = reinterpret_cast<float *>(src_data);
  auto originWeightFp16 = reinterpret_cast<float16_t *>(src_data);
  bool isModelFp16 = in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16;
  // pad weight
  // ABCICO -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, ABCOCI -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int aa = 0; aa < a; aa++) {
    for (int bb = 0; bb < b; bb++) {
      int baseAB = (aa * b + bb) * ci * CO_;
      for (int i = 0; i < ci4; ++i) {
        for (int j = 0; j < co4; ++j) {
          for (int k = 0; k < C4NUM; ++k) {
            for (int l = 0; l < C4NUM; ++l) {
              int src_ci = i * C4NUM + l;
              int src_co = j * C4NUM + k;
              if (src_ci < ci && src_co < CO_) {
                int originId = baseAB + src_ci * CO_ + src_co;
                if (transposeB) {
                  originId = baseAB + src_co * ci + src_ci;
                }
                if (enable_fp16_) {
                  if (!isModelFp16) {
                    padWeightFp16[index++] = originWeightFp32[originId];
                  } else {
                    padWeightFp16[index++] = originWeightFp16[originId];
                  }
                } else {
                  if (!isModelFp16) {
                    padWeightFp32[index++] = originWeightFp32[originId];
                  } else {
                    padWeightFp32[index++] = originWeightFp16[originId];
                  }
                }
              } else {
                index++;
              }
            }
          }
        }
      }
    }
  }
  return RET_OK;
}
#else
int MatMulOpenCLKernel::PadWeight(std::vector<int> weight_shape_4d, int ci, int co) {
  auto allocator = ocl_runtime_->GetAllocator();
  int a = weight_shape_4d[0];
  int b = weight_shape_4d[1];
  int ci4 = UP_DIV(ci, C4NUM);
  int co4 = UP_DIV(co, C4NUM);
  size_t dtype_size = sizeof(float);
  padWeight_ = allocator->Malloc(a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size, lite::opencl::MemType::BUF);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  if (padWeight_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  auto padWeight = reinterpret_cast<float *>(padWeight_);
  memset(padWeight_, 0x00, a * b * ci4 * co4 * C4NUM * C4NUM * dtype_size);
  void *src_data = stored_weight_ == nullptr ? in_tensors_.at(kWeightIndex)->data() : stored_weight_;
  auto originWeight = reinterpret_cast<float *>(src_data);
  // pad weight
  // ABCICO -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  // if tranposeB, ABCOCI -> AB(CI4)(CO4)(4 from CO)(4 from CI)
  int index = 0;
  for (int aa = 0; aa < a; aa++) {
    for (int bb = 0; bb < b; bb++) {
      int baseAB = (aa * b + bb) * ci * CO_;
      for (int i = 0; i < ci4; ++i) {
        for (int j = 0; j < co4; ++j) {
          for (int k = 0; k < C4NUM; ++k) {
            for (int l = 0; l < C4NUM; ++l) {
              int src_ci = i * C4NUM + l;
              int src_co = j * C4NUM + k;
              if (src_ci < ci && src_co < CO_) {
                int originId = baseAB + src_ci * CO_ + src_co;
                if (transposeB) {
                  originId = baseAB + src_co * ci + src_ci;
                }
                padWeight[index++] = originWeight[originId];
              } else {
                index++;
              }
            }
          }
        }
      }
    }
  }
  return RET_OK;
}
#endif

int MatMulOpenCLKernel::InitWeights() {
  if (!in_tensors_[1]->IsConst()) {
    return InitBias();
  }
  // ABMCI @ ABCICO = ABMCO
  auto allocator = ocl_runtime_->GetAllocator();
  auto weight_shape = in_tensors_[1]->shape();
  int weight_ndim = weight_shape.size();
  std::vector<int> weight_shape_4d(MAX_DIMS, 1);
  for (int i = 0; i < weight_ndim; i++) {
    weight_shape_4d[MAX_DIMS - weight_ndim + i] = weight_shape[i];
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  transposeB = param->b_transpose_;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  int ci;
  if (transposeB) {
    ci = weight_shape_4d[3];
    CO_ = weight_shape_4d[2];
  } else {
    ci = weight_shape_4d[2];
    CO_ = weight_shape_4d[3];
  }

  PadWeight(weight_shape_4d, ci, CO_);

  if (allocator->UnmapBuffer(padWeight_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_weight_);
  return InitBias();
}

#ifdef ENABLE_FP16
int MatMulOpenCLKernel::InitBias() {
  // pad FC Bias
  auto allocator = ocl_runtime_->GetAllocator();
  int co4 = UP_DIV(CO_, C4NUM);
  size_t dtype_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  lite::opencl::ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    void *src_data = stored_bias_ == nullptr ? in_tensors_.at(kBiasIndex)->data() : stored_bias_;
    if (in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat32 && enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(src_data)[i];
      }
    } else if (in_tensors_[kBiasIndex]->data_type() == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < CO_; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(src_data)[i];
      }
    } else {
      memcpy(bias_, src_data, CO_ * dtype_size);
    }
  }
  if (allocator->UnmapBuffer(bias_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_bias_);
  return RET_OK;
}
#else
int MatMulOpenCLKernel::InitBias() {
  // pad FC Bias
  auto allocator = ocl_runtime_->GetAllocator();
  int co4 = UP_DIV(CO_, C4NUM);
  size_t dtype_size = sizeof(float);
  size_t im_dst_x, im_dst_y;
  im_dst_x = co4;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  lite::opencl::ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Malloc failed.";
    return RET_ERROR;
  }
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  if (bias_ == nullptr) {
    MS_LOG(ERROR) << "Map Buffer failed.";
    return RET_ERROR;
  }
  memset(bias_, 0x00, co4 * C4NUM * dtype_size);
  if (in_tensors_.size() == INPUT_TENSOR_SIZE_3) {
    void *src_data = stored_bias_ == nullptr ? in_tensors_.at(kBiasIndex)->data() : stored_bias_;
    memcpy(bias_, src_data, CO_ * dtype_size);
  }
  if (allocator->UnmapBuffer(bias_) != RET_OK) {
    MS_LOG(ERROR) << "UnmapBuffer failed.";
    return RET_ERROR;
  }
  FreeStoredData(stored_bias_);
  return RET_OK;
}
#endif

int MatMulOpenCLKernel::SetGlobalLocal() {
  // local size should be less than MAX_GROUP_SIZE
  local_size_ = {32, 4, 1};
  global_size_ = {1, 1, 1};
  global_size_ = {UP_DIV(static_cast<size_t>(outShape[3]), C4NUM),
                  4 * static_cast<size_t>(outShape[0]) * static_cast<size_t>(outShape[1]),
                  static_cast<size_t>(outShape[2])};
  AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int MatMulOpenCLKernel::SetConstArgs() {
  int arg_count = 2;
  cl_int4 in_shape = {inShape[0], inShape[1], inShape[2], inShape[3]};
  cl_int4 out_shape = {outShape[0], outShape[1], outShape[2], outShape[3]};
  if (act_weight_) {
    arg_count++;
  } else {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, padWeight_, true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, bias_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  auto param = reinterpret_cast<MatMulParameter *>(op_parameter_);
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, param->act_type_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_count = 0;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (act_weight_) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_count++, in_tensors_[1]->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulOpenCLKernel::StoreConstData() {
  if (!InferShapeDone()) {
    stored_weight_ = StoreTensorData(in_tensors_.at(kWeightIndex));
    if (stored_weight_ == nullptr) {
      MS_LOG(ERROR) << "Store weight failed.";
      return RET_ERROR;
    }
    if (in_tensors_.size() > kBiasIndex) {
      stored_bias_ = StoreTensorData(in_tensors_.at(kBiasIndex));
      if (stored_bias_ == nullptr) {
        MS_LOG(ERROR) << "Store bias failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

kernel::LiteKernel *OpenCLMatMulKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  kernel::OpenCLKernel *kernel = nullptr;
  auto shape = outputs.front()->shape();
  bool infer_shape_done = std::find(shape.begin(), shape.end(), -1) == shape.end();
  if (infer_shape_done && IsUseStrassenMatmul(inputs)) {
    MS_LOG(DEBUG) << "use_matmul_strassen";
    kernel = new (std::nothrow) StrassenOpenCLKernel(opParameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) MatMulOpenCLKernel(opParameter, inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(WARNING) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  if (!infer_shape_done) {
    MS_LOG(WARNING) << "kernel don't infer shape yet!";
    auto ret = reinterpret_cast<MatMulOpenCLKernel *>(kernel)->StoreConstData();
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(WARNING) << "Store " << opParameter->name_ << " const data failed!";
      delete kernel;
      return nullptr;
    }
    return kernel;
  }
  if (kernel->CheckSpecs() != RET_OK || kernel->OpenCLKernel::CheckSpecs() != RET_OK) {
    MS_LOG(WARNING) << "Check " << opParameter->name_ << " specification failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MatMulFusion, OpenCLMatMulKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MatMulFusion, OpenCLMatMulKernelCreator)
}  // namespace mindspore::kernel
