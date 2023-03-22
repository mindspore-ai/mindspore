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

#include "src/litert/infer_manager.h"
#include "src/litert/kernel/opencl/opencl_kernel.h"
#include "src/litert/weight_decoder.h"
#include "src/common/file_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;

namespace mindspore::kernel {
int CpuAxis2GpuAxis(size_t ndim, int cpu_axis, int *gpu_axis) {
  static const std::vector<std::vector<int>> kCpuAxis2GpuAxisMapTable = {
    // For 1D tensor, map the cpu axis [0] to gpu axis [kNHWC_C].
    {kNHWC_C},
    // For 2D tensor, map the cpu axis [0, 1] to gpu axis [kNHWC_N, kNHWC_C].
    {kNHWC_N, kNHWC_C},
    // For 3D tensor, map the cpu axis [0, 1, 2] to gpu axis [kNHWC_N, kNHWC_W, kNHWC_C].
    {kNHWC_N, kNHWC_W, kNHWC_C},
    // For 4D tensor, map the cpu axis [0, 1, 2, 3] to gpu axis [kNHWC_N, kNHWC_H, kNHWC_W, kNHWC_C].
    {kNHWC_N, kNHWC_H, kNHWC_W, kNHWC_C},
    // For 5D tensor, map the cpu axis [0, 1, 2, 3, 4] to gpu axis [kNDHWC_N, kNDHWC_D, kNDHWC_H, kNDHWC_W, kNDHWC_C].
    {kNDHWC_N, kNDHWC_D, kNDHWC_H, kNDHWC_W, kNDHWC_C},
  };
  if (gpu_axis == nullptr) {
    MS_LOG(WARNING) << "Input parameter gpu axis is null";
    return RET_ERROR;
  }

  if ((ndim == 0) || (ndim > kCpuAxis2GpuAxisMapTable.size())) {
    MS_LOG(WARNING) << "Only support ndim of 1D...5D, bad input ndim: " << ndim;
    return RET_ERROR;
  }

  const auto &axis_map = kCpuAxis2GpuAxisMapTable[ndim - 1];
  if ((cpu_axis < 0) || (static_cast<size_t>(cpu_axis) >= axis_map.size())) {
    MS_LOG(WARNING) << "Input cpu axis: " << cpu_axis << " is out of range [0," << axis_map.size() << "]";
    return RET_ERROR;
  }
  *gpu_axis = axis_map[cpu_axis];
  return RET_OK;
}

void OpenCLKernel::AlignGlobalLocal(const std::vector<size_t> &global, const std::vector<size_t> &local) {
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
  } else if (global.size() >= 3) {
    global_range_ = cl::NDRange(internal_global_ws.at(0), internal_global_ws.at(1), internal_global_ws.at(2));
    if (!local.empty()) {
      local_range_ = cl::NDRange(local.at(0), local.at(1), local.at(2));
    }
  }
}

int OpenCLKernel::GetImageSize(size_t idx, lite::opencl::ImageSize *img_size) {
  MS_ASSERT(img_size);
  if (idx >= out_tensors_.size()) {
    return RET_ERROR;
  }
  auto img_info = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_[idx]);
  size_t img_dtype = CL_FLOAT;
  switch (out_tensors_[idx]->data_type()) {
    case kNumberTypeFloat32: {
      img_dtype = CL_FLOAT;
      break;
    }
    case kNumberTypeInt32: {
      img_dtype = CL_SIGNED_INT32;
      break;
    }
    case kNumberTypeFloat16: {
      img_dtype = CL_HALF_FLOAT;
      break;
    }
    case kNumberTypeInt8: {
      img_dtype = CL_SIGNED_INT8;
      break;
    }
    default: {
      MS_LOG(WARNING) << "Unsupported data_type " << out_tensors_[idx]->data_type();
      return RET_ERROR;
    }
  }
  *img_size = {img_info->width, img_info->height, img_dtype};
  return RET_OK;
}

void OpenCLKernel::PrintShape(lite::Tensor *output_tensor) {
  printf("shape=(");
  auto shape = output_tensor->shape();
  for (size_t i = 0; i < shape.size(); ++i) {
    printf("%4d", shape[i]);
    if (i + 1 < shape.size()) {
      printf(",");
    }
  }
  printf(") ");
}

void OpenCLKernel::PrintOutput(int print_num, const std::string &out_file) {
  printf("%-30s ", name().c_str());
  if (out_tensors().empty()) {
    return;
  }
  auto *tensor = out_tensors()[0];
  auto mem_type = GetMemType();
  if (tensor == nullptr || tensor->data() == nullptr) {
    return;
  }

  auto img_info = GpuTensorInfo::CreateGpuTensorInfo(tensor);
  auto size = mem_type == lite::opencl::MemType::BUF ? img_info->OriginSize : img_info->Image2DSize;
  std::vector<char> data(size);
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeInnerWrapper();
  auto runtime = runtime_wrapper.GetInstance();
  auto allocator = runtime->GetAllocator();
  if (!runtime->SyncCommandQueue()) {
    MS_LOG(ERROR) << "SyncCommandQueue failed.";
  }
  if (mem_type == lite::opencl::MemType::BUF) {
    if (allocator->MapBuffer(tensor->data(), CL_MAP_READ, nullptr, true) == nullptr) {
      MS_LOG(ERROR) << "Map Buffer failed.";
    }
    memcpy(data.data(), tensor->data(), img_info->OriginSize);
    if (allocator->UnmapBuffer(tensor->data()) != RET_OK) {
      MS_LOG(ERROR) << "UnmapBuffer failed.";
    }
  } else {
    runtime->ReadImage(tensor->data(), data.data());
  }

  PrintShape(tensor);

  auto total_num = mem_type == lite::opencl::MemType::BUF ? img_info->ElementsNum : img_info->ElementsC4Num;
  for (int i = 0; i < print_num && i < static_cast<int>(total_num); ++i) {
#ifdef ENABLE_FP16
    if (tensor->data_type() == kNumberTypeInt32) {
      printf("%d %7d | ", i, reinterpret_cast<int32_t *>(data.data())[i]);
    } else if (tensor->data_type() == kNumberTypeFloat16) {
      printf("%d %7.3f | ", i, reinterpret_cast<float16_t *>(data.data())[i]);
    } else if (tensor->data_type() == kNumberTypeFloat32) {
      printf("%d %7.3f | ", i, reinterpret_cast<float *>(data.data())[i]);
    } else if (tensor->data_type() == kNumberTypeInt8) {
      printf("%d %7d | ", i, static_cast<int>(reinterpret_cast<int8_t *>(data.data())[i]));
    }
#else
    printf("%d %7.3f | ", i, reinterpret_cast<float *>(data.data())[i]);
#endif
  }
  printf("\n");

  if (!out_file.empty()) {
    (void)lite::WriteToBin(out_file, data.data(), data.size());
  }
}

std::string OpenCLKernel::OpenCLKernelHeader() {
  std::stringstream header;
  header << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
            "__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
            "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n";
  if (this->registry_data_type_ == kNumberTypeFloat32) {
    header << "#define DTYPE float\n"
              "#define DTYPE4 float4\n"
              "#define WRITE_IMAGE write_imagef\n"
              "#define READ_IMAGE read_imagef\n";
  } else if (this->registry_data_type_ == kNumberTypeFloat16) {
    header << "#define DTYPE half\n"
              "#define DTYPE4 half4\n"
              "#define WRITE_IMAGE write_imageh\n"
              "#define READ_IMAGE read_imageh\n";
  } else if (this->registry_data_type_ == kNumberTypeInt32) {
    header << "#define DTYPE int\n"
              "#define DTYPE4 int4\n"
              "#define WRITE_IMAGE write_imagei\n"
              "#define READ_IMAGE read_imagei\n";
  } else {
    MS_LOG(ERROR) << "Unsupported data type: " << this->registry_data_type_;
    return "";
  }
  return header.str();
}

bool OpenCLKernel::MallocDataDone() {
  for (auto &out_tensor : out_tensors_) {
    if (out_tensor->data() == nullptr) {
      return false;
    }
    auto allocator = out_tensor->allocator();
    if (allocator == nullptr) {
      return false;
    }
    lite::opencl::MemType memType;
    auto buffer = reinterpret_cast<mindspore::lite::opencl::OpenCLAllocator *>(allocator.get())
                    ->GetOpenclMemPtr(out_tensor->data(), &memType);
    if ((buffer == nullptr) || (memType != lite::opencl::MemType::IMG)) {
      return false;
    }
  }
  return true;
}

int OpenCLKernel::PreProcess() {
  if (MallocDataDone()) {
    return RET_OK;
  }
  int ret = ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    auto *output = out_tensors_.at(i);
    CHECK_NULL_RETURN(output);
    CHECK_NULL_RETURN(output->allocator());
    if (GetMemType() == lite::opencl::MemType::IMG) {
      ImageSize img_size;
      ret = GetImageSize(i, &img_size);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "GetImageSize failed";
        return ret;
      }
      auto data_ptr =
        output->allocator()->Malloc(img_size.width, img_size.height, static_cast<enum DataType>(output->data_type()));
      if (data_ptr == nullptr) {
        MS_LOG(ERROR) << "Malloc data failed";
        return RET_ERROR;
      }
      output->set_data(data_ptr);
    } else {
      ret = output->MallocData();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "MallocData failed";
        return ret;
      }
    }
    output->ResetRefCount();
  }
  is_oversize_kernel_ = ocl_runtime_->GetAllocator()->IsOverSize();
  return RET_OK;
}

int OpenCLKernel::InferShape() {
  if (InferShapeDone()) {
    return RET_OK;
  }
  auto ret = lite::KernelInferShape(in_tensors_, out_tensors_, op_parameter_);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "InferShape failed, type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type()));
    return ret;
  }
  return RET_OK;
}

int OpenCLKernel::ReSize() {
  if (InferShapeDone()) {
    return RET_OK;
  }
  auto ret = InferShape();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CheckSpecs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReSize failed for check kernel specs!";
    return ret;
  }
  ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReSize failed for kernel prepare!";
    return ret;
  }
  return RET_OK;
}

std::vector<BaseTuningParameter> OpenCLKernel::GenerateTuningParam() {
  size_t ndim = global_size_.size();
  std::vector<BaseTuningParameter> tuning_params = {};
  if (ndim == 0) {
    MS_LOG(ERROR) << "Generate tuning param failed, global_size_ is null.";
    return tuning_params;
  }
  BaseTuningParameter default_tuning_param = BaseTuningParameter();
  default_tuning_param.local_size = local_size_;
  tuning_params.push_back(default_tuning_param);
  std::vector<size_t> max_work_items = ocl_runtime_->GetWorkItemSize();
  size_t max_workgroup_size = ocl_runtime_->GetMaxWorkGroupSize(kernel_);
  const size_t MIN_WORKGROUP_SIZE = 8;
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

int OpenCLKernel::AssignTuningParam(const BaseTuningParameter &param) {
  std::vector<size_t> local_size_tmp = param.local_size;
  if (local_size_tmp.size() > global_size_.size()) {
    local_size_tmp = std::vector<size_t>(local_size_tmp.begin(), local_size_tmp.begin() + global_size_.size());
  }
  AlignGlobalLocal(global_size_, local_size_tmp);
  return RET_OK;
}

int OpenCLKernel::Tune() {
  if (!ocl_runtime_->isProfiling()) {
    MS_LOG(WARNING) << "Tuning mode require opencl runtime profiling.";
    return RET_OK;
  }
  lite::opencl::TuningMode mode = ocl_runtime_->GetTuningMode();
  if (mode == lite::opencl::TuningMode::DEFAULT) {
    return RET_OK;
  }
  static const std::set<int> FAST_MODE_OPS = {schema::PrimitiveType_Conv2DFusion,
                                              schema::PrimitiveType_Conv2dTransposeFusion};
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
  for (size_t i = 0; i < tuning_params.size(); i++) {
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

double OpenCLKernel::GetProfilingTimeMs() {
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

std::set<size_t> OpenCLKernel::GenerateLocalByGlobal(size_t global_i) {
  std::set<size_t> local_ = {};
  int index = 1;
  while (index <= static_cast<int>(global_i)) {
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

int OpenCLKernel::CheckSpecs() {
  if (out_mem_type_ == lite::opencl::MemType::IMG) {
    if (!GpuTensorInfo::CreateGpuTensorInfo(out_tensors_[0])->IsImageSizeValid()) {
      return RET_ERROR;
    }
  }
  if (in_tensors_.size() > 0) {
    if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16 &&
        in_tensors_[0]->data_type() != kNumberTypeInt32 && in_tensors_[0]->data_type() != kNumberTypeInt8) {
      MS_LOG(WARNING) << "Unsupported data type: " << in_tensors_[0]->data_type();
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
