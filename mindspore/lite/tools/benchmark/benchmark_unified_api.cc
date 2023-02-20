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

#include "tools/benchmark/benchmark_unified_api.h"
#define __STDC_FORMAT_MACROS
#include <cinttypes>
#undef __STDC_FORMAT_MACROS
#include <algorithm>
#include <utility>
#include <functional>
#include <iomanip>
#include <limits>
#include "src/common/common.h"
#include "src/tensor.h"
#include "tools/common/string_util.h"
#include "nnacl/nnacl_common.h"
#ifdef ENABLE_ARM64
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <unistd.h>
#endif
#ifdef SUPPORT_NNIE
#include "include/hi_common.h"
#include "include/hi_comm_vb.h"
#include "include/mpi_sys.h"
#include "include/mpi_vb.h"
#endif
#ifdef PARALLEL_INFERENCE
#include <thread>
#include "src/common/config_file.h"
#endif

namespace mindspore {
constexpr size_t kDataToStringMaxNum = 40;
constexpr int kPrintDataNum = 20;
constexpr int kFrequencyDefault = 3;
constexpr int kPercentageDivisor = 100;
constexpr int kDumpInputsAndOutputs = 0;
constexpr int kDumpOutputs = 2;
#ifdef PARALLEL_INFERENCE
constexpr int kMaxRequestNum = 200;
#endif
namespace lite {
int BenchmarkUnifiedApi::GenerateGLTexture(std::map<std::string, GLuint> *input_gl_texture) {
  for (auto tensor : ms_inputs_for_api_) {
    float *input_data = reinterpret_cast<float *>(malloc(tensor.DataSize()));
    if (input_data == nullptr) {
      MS_LOG(ERROR) << "new input_data failed";
      return RET_ERROR;
    }
    int status = GenerateRandomData(tensor.DataSize(), input_data, static_cast<int>(tensor.DataType()));
    if (status != RET_OK) {
      free(input_data);
      std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
      MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
      return status;
    }
    status = FillGLTextureToTensor(input_gl_texture, &tensor, tensor.Name(), input_data);
    free(input_data);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Fill GLTexture to input tensor" << status;
      return status;
    }
  }

  return RET_OK;
}

int BenchmarkUnifiedApi::FillGLTextureToTensor(std::map<std::string, GLuint> *gl_texture, mindspore::MSTensor *tensor,
                                               std::string name, void *data) {
  MS_CHECK_TRUE_MSG(gl_texture != nullptr, RET_ERROR, "gl_texture is nullptr");
  MS_CHECK_TRUE_MSG(tensor != nullptr, RET_ERROR, "tensor is nullptr");

  auto image_id = 0;

  int width = 1, height = 1, channel = 1;
  if (tensor->Shape().size() == DIMENSION_2D) {
    height = tensor->Shape()[kNHWC_N];
    channel = tensor->Shape()[kNHWC_H];
  } else if (tensor->Shape().size() == DIMENSION_3D) {
    width = tensor->Shape()[kNHWC_H];
    height = tensor->Shape()[kNHWC_N];
    channel = tensor->Shape()[kNHWC_C];
  } else if (tensor->Shape().size() == DIMENSION_4D) {
    width = tensor->Shape()[kNHWC_W];
    height = tensor->Shape()[kNHWC_H];
    channel = tensor->Shape()[kNHWC_C];
  } else {
    MS_LOG(ERROR) << "the tensor shape is not support";
    return RET_ERROR;
  }

  if (data == nullptr) {
    image_id = gl_runtime_.GLCreateTexture(width, height, channel);
  } else {
    image_id = gl_runtime_.CopyHostToDeviceTexture(data, width, height, channel);
  }

  if (image_id != GL_NONE) {
    gl_texture->insert(std::pair<std::string, GLuint>(name, image_id));
  } else {
    MS_LOG(ERROR) << "glMemPool CopyHostToDeviceTexture failed";
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::LoadAndBindGLTexture() {
  std::map<std::string, GLuint> input_gl_texture;
  std::map<std::string, GLuint> output_gl_texture;

  if (flags_->in_data_file_.empty()) {
    auto status = GenerateGLTexture(&input_gl_texture);
    if (status != RET_OK) {
      std::cerr << "Generate input GLTexture error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input GLTexture error " << status;
      return status;
    }
  } else {
    auto status = ReadGLTextureFile(&input_gl_texture);
    if (status != RET_OK) {
      std::cerr << "ReadGLTextureFile error, " << status << std::endl;
      MS_LOG(ERROR) << "ReadGLTextureFile error, " << status;
      return status;
    }
  }

  for (auto &tensor : ms_outputs_for_api_) {
    auto status = FillGLTextureToTensor(&output_gl_texture, &tensor, tensor.Name());
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Fill GLTexture to output tensor" << status;
      return status;
    }
  }

  auto status = ms_model_.BindGLTexture2DMemory(input_gl_texture, &output_gl_texture);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "BindGLTexture2DMemory failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::ReadGLTextureFile(std::map<std::string, GLuint> *input_gl_texture) {
  if (ms_inputs_for_api_.empty()) {
    return RET_OK;
  }
  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < flags_->input_data_list_.size(); i++) {
      auto tensor = ms_inputs_for_api_.at(i);
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      auto tensor_data_size = tensor.DataSize();
      if (size != tensor_data_size) {
        std::cerr << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                  << std::endl;
        MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
        delete[] bin_buf;
        return RET_ERROR;
      }

      auto status = FillGLTextureToTensor(input_gl_texture, &tensor, tensor.Name(), bin_buf);
      delete[] bin_buf;
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Fill GLTexture to input tensor" << status;
        return status;
      }
    }
  }

  return RET_OK;
}

int BenchmarkUnifiedApi::LoadInput() {
  if (flags_->enable_gl_texture_ == true) {
    if (lite::BenchmarkUnifiedApi::LoadAndBindGLTexture() != RET_OK) {
      MS_LOG(ERROR) << "Generate input GLTexture error";
      return RET_ERROR;
    }
    return RET_OK;
  }

  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData();
    if (status != RET_OK) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != RET_OK) {
      std::cerr << "ReadInputFile error, " << status << std::endl;
      MS_LOG(ERROR) << "ReadInputFile error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::GenerateInputData() {
#ifdef PARALLEL_INFERENCE
  if (flags_->enable_parallel_predict_) {
    std::vector<void *> inputs;
    for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
      auto tensor_name = ms_inputs_for_api_[i].Name();
      size_t size;
      if (ms_inputs_for_api_[i].DataType() == static_cast<enum DataType>(kNumberTypeFloat32)) {
        size = sizeof(float);
      } else if (ms_inputs_for_api_[i].DataType() == static_cast<enum DataType>(kNumberTypeInt32)) {
        size = sizeof(int32_t);
      } else {
        MS_LOG(ERROR) << "not support in model pool.";
        return RET_ERROR;
      }
      for (size_t j = 0; j < flags_->resize_dims_[i].size(); j++) {
        size *= flags_->resize_dims_[i][j];
      }
      void *input_data = new (std::nothrow) char[size];
      if (input_data == nullptr) {
        MS_LOG(ERROR) << "new input_data failed";
        for (auto &data : inputs) {
          auto buf = static_cast<char *>(data);
          delete[] buf;
          data = nullptr;
        }
        return RET_ERROR;
      }
      inputs.push_back(input_data);
      int status = GenerateRandomData(size, input_data, static_cast<int>(ms_inputs_for_api_[i].DataType()));
      if (status != RET_OK) {
        MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
        for (auto &data : inputs) {
          auto buf = static_cast<char *>(data);
          delete[] buf;
          data = nullptr;
        }
        return status;
      }
    }
    all_inputs_data_.push_back(inputs);
    return RET_OK;
  }
#endif
  for (auto &tensor : ms_inputs_for_api_) {
    if (static_cast<int>(tensor.DataType()) == kObjectTypeString) {
      MSTensor *input = MSTensor::StringsToTensor(tensor.Name(), {"you're the best."});
      if (input == nullptr) {
        std::cerr << "StringsToTensor failed" << std::endl;
        MS_LOG(ERROR) << "StringsToTensor failed";
        return RET_ERROR;
      }
      tensor = *input;
      delete input;
    } else {
      auto input_data = tensor.MutableData();
      if (input_data == nullptr) {
        MS_LOG(ERROR) << "MallocData for inTensor failed";
        return RET_ERROR;
      }
      int status = GenerateRandomData(tensor.DataSize(), input_data, static_cast<int>(tensor.DataType()));
      if (status != RET_OK) {
        std::cerr << "GenerateRandomData for inTensor failed: " << status << std::endl;
        MS_LOG(ERROR) << "GenerateRandomData for inTensor failed:" << status;
        return status;
      }
    }
  }
  return RET_OK;
}

void BenchmarkUnifiedApi::UpdateConfigInfo() {
#define WIPE_DEEP_CONFIG_ENV '0'
#define WIPE_DEEP_CONFIG_VOCAB_SIZE "100"
#define WIPE_DEEP_CONFIG_DEVICE_CACHE_SIZE "40"

  auto env = std::getenv("BENCHMARK_UPDATE_CONFIG_ENV");
  if (env == nullptr) {
    return;
  }
  if (env[0] == WIPE_DEEP_CONFIG_ENV) {
    ms_model_.UpdateConfig(kMSCacheSection, std::make_pair(kMSCacheVocabSizeKey, WIPE_DEEP_CONFIG_VOCAB_SIZE));
    ms_model_.UpdateConfig(kMSCacheSection, std::make_pair(kMSCacheDeviceSizeKey, WIPE_DEEP_CONFIG_DEVICE_CACHE_SIZE));
  }
  return;
}

int BenchmarkUnifiedApi::ReadInputFile() {
#ifdef PARALLEL_INFERENCE
  if (flags_->enable_parallel_predict_) {
    std::vector<void *> inputs;
    for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        for (auto &data : inputs) {
          auto buf = static_cast<char *>(data);
          delete[] buf;
          data = nullptr;
        }
        return RET_ERROR;
      }
      inputs.push_back(bin_buf);
    }
    all_inputs_data_.push_back(inputs);
    return RET_OK;
  }
#endif
  if (ms_inputs_for_api_.empty()) {
    return RET_OK;
  }

  if (this->flags_->in_data_type_ == kImage) {
    MS_LOG(ERROR) << "Not supported image input";
    return RET_ERROR;
  } else {
    for (size_t i = 0; i < flags_->input_data_list_.size(); i++) {
      auto &cur_tensor = ms_inputs_for_api_.at(i);
      size_t size;
      char *bin_buf = ReadFile(flags_->input_data_list_[i].c_str(), &size);
      if (bin_buf == nullptr) {
        MS_LOG(ERROR) << "ReadFile return nullptr";
        return RET_ERROR;
      }
      if (static_cast<int>(cur_tensor.DataType()) == kObjectTypeString) {
        std::string str(bin_buf, size);
        MSTensor *input = MSTensor::StringsToTensor(cur_tensor.Name(), {str});
        if (input == nullptr) {
          std::cerr << "StringsToTensor failed" << std::endl;
          MS_LOG(ERROR) << "StringsToTensor failed";
          delete[] bin_buf;
          return RET_ERROR;
        }
        cur_tensor = *input;
      } else {
        auto tensor_data_size = cur_tensor.DataSize();
        if (size != tensor_data_size) {
          std::cerr << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size
                    << std::endl;
          MS_LOG(ERROR) << "Input binary file size error, required: " << tensor_data_size << ", in fact: " << size;
          delete[] bin_buf;
          return RET_ERROR;
        }
        auto input_data = cur_tensor.MutableData();
        if (input_data == nullptr) {
          MS_LOG(ERROR) << "input_data is nullptr.";
          delete[] bin_buf;
          return RET_ERROR;
        }
        memcpy(input_data, bin_buf, tensor_data_size);
      }
      delete[] bin_buf;
    }
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::GetDataTypeByTensorName(const std::string &tensor_name) {
#ifdef PARALLEL_INFERENCE
  for (auto tensor : ms_outputs_for_api_) {
    auto name = tensor.Name();
    if (name == tensor_name) {
      return static_cast<int>(tensor.DataType());
    }
  }
  MS_LOG(ERROR) << "not find tensor name : " << tensor_name << " in model output.";
  return static_cast<int>(DataType::kTypeUnknown);
#endif
  return static_cast<int>(ms_model_.GetOutputByTensorName(tensor_name).DataType());
}

void BenchmarkUnifiedApi::UpdateDistributionName(const std::shared_ptr<mindspore::Context> &context,
                                                 std::string *name) {
  if (flags_->device_ != "GPU") {
    return;
  }

  if (name->size() == 0) {
    return;
  }

  if (context->MutableDeviceInfo().size() == 0) {
    return;
  }

  auto device_info = context->MutableDeviceInfo().front();
  GPUDeviceInfo *gpu_info = reinterpret_cast<GPUDeviceInfo *>(device_info.get());
  auto rank_id = gpu_info->GetRankID();
  if (rank_id == 0) {
    return;
  }
  gpu_info->SetDeviceID(rank_id);

  /* model file & benchmark data file: include .mindir
   config file :  include .config */
  auto replace_pos = name->find(".mindir");
  if (replace_pos == std::string::npos) {
    replace_pos = name->find(".config");
  }

  if (replace_pos == std::string::npos) {
    return;
  }

  *name = name->replace(replace_pos, sizeof('.'), std::to_string(rank_id) + ".");

  MS_LOG(INFO) << "Update distribution info: " << *name;
  std::cout << "Update distribution info: " << *name << std::endl;
  return;
}

int BenchmarkUnifiedApi::InitMSContext(const std::shared_ptr<mindspore::Context> &context) {
  context->SetThreadNum(flags_->num_threads_);
  context->SetThreadAffinity(flags_->cpu_bind_mode_);
  context->SetInterOpParallelNum(flags_->inter_op_parallel_num_);
  if (!flags_->core_list_.empty()) {
    context->SetThreadAffinity(flags_->core_list_);
  }
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
  if (flags_->delegate_mode_ == "CoreML") {
    context->SetBuiltInDelegate(kCoreML);
  } else if (flags_->delegate_mode_ == "NNAPI") {
    context->SetBuiltInDelegate(kNNAPI);
  }
  context->SetEnableParallel(flags_->enable_parallel_);
#endif

  auto &device_list = context->MutableDeviceInfo();

  if (flags_->device_ == "GPU") {
    std::shared_ptr<GPUDeviceInfo> gpu_device_info = std::make_shared<GPUDeviceInfo>();
    gpu_device_info->SetEnableFP16(flags_->enable_fp16_);
    uint32_t device_id = 0;
    auto device_id_env = std::getenv("GPU_DEVICE_ID");
    if (device_id_env != nullptr) {
      try {
        device_id = static_cast<uint32_t>(std::stoul(device_id_env));
      } catch (std::invalid_argument &e) {
        MS_LOG(WARNING) << "Invalid device id env:" << device_id_env << ". Set default device id 0.";
      }
      MS_LOG(INFO) << "GPU device_id = " << device_id;
    }
    gpu_device_info->SetDeviceID(device_id);
    if (flags_->enable_gl_texture_) {
      gpu_device_info->SetEnableGLTexture(flags_->enable_gl_texture_);

      auto gl_context = eglGetCurrentContext();
      gpu_device_info->SetGLContext(gl_context);

      auto gl_display = eglGetCurrentDisplay();
      gpu_device_info->SetGLDisplay(gl_display);
    } else {
      gpu_device_info->SetProvider("tensorrt");
      gpu_device_info->SetAllocator(nullptr);
    }
    device_list.push_back(gpu_device_info);
  }

  if (flags_->device_ == "NPU") {
    std::shared_ptr<KirinNPUDeviceInfo> npu_device_info = std::make_shared<KirinNPUDeviceInfo>();
    npu_device_info->SetEnableFP16(flags_->enable_fp16_);
    npu_device_info->SetFrequency(kFrequencyDefault);
    device_list.push_back(npu_device_info);
  }

  if (flags_->device_ == "Ascend310" || flags_->device_ == "Ascend310P") {
    uint32_t device_id = 0;
    auto device_id_env = std::getenv("ASCEND_DEVICE_ID");
    if (device_id_env != nullptr) {
      try {
        device_id = static_cast<uint32_t>(std::stoul(device_id_env));
      } catch (std::invalid_argument &e) {
        MS_LOG(WARNING) << "Invalid device id env:" << device_id_env << ". Set default device id 0.";
      }
      MS_LOG(INFO) << "Ascend device_id = " << device_id;
    }
    std::shared_ptr<AscendDeviceInfo> ascend_device_info = std::make_shared<AscendDeviceInfo>();
    ascend_device_info->SetDeviceID(device_id);
    auto back_policy_env = std::getenv("ASCEND_BACK_POLICY");
    if (back_policy_env != nullptr) {
      ascend_device_info->SetProvider(back_policy_env);
    }
    device_list.push_back(ascend_device_info);
  }

  // CPU priority is behind GPU and NPU
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(flags_->enable_fp16_);
  device_info->SetProvider(flags_->provider_);
  device_list.push_back(device_info);

  return RET_OK;
}
#ifdef PARALLEL_INFERENCE
int BenchmarkUnifiedApi::CompareOutputForModelPool(std::vector<mindspore::MSTensor> *outputs) {
  if (outputs->empty()) {
    MS_LOG(ERROR) << "outputs is empty.";
    return RET_ERROR;
  }
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  // check the output tensor name.
  for (size_t i = 0; i < outputs->size(); i++) {
    std::string tensor_name = outputs->at(i).Name();
    mindspore::MSTensor tensor = outputs->at(i);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Get tensor failed, tensor name: " << tensor_name;
      return RET_ERROR;
    }
    constexpr float kParallelRelative = 1e-7;
    constexpr float kParallelAbsolute = 1e-10;
    int ret = CompareDataGetTotalBiasAndSize(tensor_name, &tensor, &total_bias, &total_size, kParallelRelative,
                                             kParallelAbsolute);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Error in CompareData";
      std::cerr << "Error in CompareData" << std::endl;
      std::cout << "=======================================================" << std::endl << std::endl;
      return ret;
    }
  }
  float mean_bias;
  if (total_size != 0) {
    mean_bias = ((total_bias / float_t(total_size)) * kPercentageDivisor);
  } else {
    mean_bias = 0;
  }

  std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%" << std::endl;
  std::cout << "=======================================================" << std::endl << std::endl;

  if (mean_bias > this->flags_->accuracy_threshold_) {
    MS_LOG(ERROR) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
    std::cerr << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}
#endif

void Convert2Float32(float *__restrict out, const uint16_t in) {
  uint32_t t1;
  uint32_t t2;
  uint32_t t3;

  t1 = in & 0x7fffu;
  t2 = in & 0x8000u;
  t3 = in & 0x7c00u;

  t1 <<= 13u;
  t2 <<= 16u;

  t1 += 0x38000000;

  t1 = (t3 == 0 ? 0 : t1);

  t1 |= t2;

  *(out) = static_cast<float>(t1);
}

int BenchmarkUnifiedApi::CompareOutput() {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_bias = 0;
  int total_size = 0;
  // check the output tensor name.
  if (this->benchmark_tensor_names_ != ms_model_.GetOutputTensorNames()) {
    MS_LOG(ERROR) << "The output tensor name is wrong.";
    return RET_ERROR;
  }
  for (const auto &calib_tensor : benchmark_data_) {
    std::string tensor_name = calib_tensor.first;
    mindspore::MSTensor tensor = ms_model_.GetOutputByTensorName(tensor_name);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Get tensor failed, tensor name: " << tensor_name;
      return RET_ERROR;
    }
    int ret;
    if (static_cast<int>(tensor.DataType()) == kObjectTypeString) {
      std::vector<std::string> output_strings = MSTensor::TensorToStrings(tensor);
      ret = CompareStringData(tensor_name, calib_tensor.second->strings_data, output_strings);
    } else {
      if (flags_->enable_gl_texture_) {
        auto *gltexture_id = reinterpret_cast<GLuint *>(tensor.MutableData());
        if (gltexture_id == nullptr) {
          MS_LOG(ERROR) << "get gltexture_id failed";
          return RET_ERROR;
        }
        auto tmp = gl_runtime_.CopyDeviceTextureToHost(*gltexture_id);
        if (tmp == nullptr) {
          MS_LOG(ERROR) << "CopyDeviceTextureToHost failed";
          return RET_ERROR;
        }
        float *hostptr = reinterpret_cast<float *>(tmp);

        auto tensor_shape = tensor.Shape();
        auto data_len =
          std::accumulate(tensor_shape.begin(), tensor_shape.end(), sizeof(float), std::multiplies<size_t>());
        auto *new_tensor = new (std::nothrow)
          MSTensor(tensor_name, mindspore::DataType::kNumberTypeFloat32, tensor_shape, hostptr, data_len);
        MS_CHECK_TRUE_MSG(new_tensor != nullptr, RET_ERROR, "new tensor failed");
        if (new_tensor->MutableData() == nullptr) {
          MS_LOG(ERROR) << "CopyDeviceTextureToHost failed";
          delete new_tensor;
          return RET_ERROR;
        }
        ret = CompareDataGetTotalBiasAndSize(tensor_name, new_tensor, &total_bias, &total_size);
        delete new_tensor;
      } else {
        ret = CompareDataGetTotalBiasAndSize(tensor_name, &tensor, &total_bias, &total_size);
      }
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Error in CompareData";
      std::cerr << "Error in CompareData" << std::endl;
      std::cout << "=======================================================" << std::endl << std::endl;
      return ret;
    }
  }
  float mean_bias;
  if (total_size != 0) {
    mean_bias = ((total_bias / float_t(total_size)) * kPercentageDivisor);
  } else {
    mean_bias = 0;
  }

  std::cout << "Mean bias of all nodes/tensors: " << mean_bias << "%" << std::endl;
  std::cout << "=======================================================" << std::endl << std::endl;

  if (mean_bias > this->flags_->accuracy_threshold_) {
    MS_LOG(ERROR) << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%";
    std::cerr << "Mean bias of all nodes/tensors is too big: " << mean_bias << "%" << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::CompareOutputByCosineDistance(float cosine_distance_threshold) {
  std::cout << "================ Comparing Output data ================" << std::endl;
  float total_cosine_distance = 0;
  int total_size = 0;
  // check the output tensor name.
  if (this->benchmark_tensor_names_ != ms_model_.GetOutputTensorNames()) {
    MS_LOG(ERROR) << "The output tensor name is wrong.";
    return RET_ERROR;
  }
  for (const auto &calib_tensor : benchmark_data_) {
    std::string tensor_name = calib_tensor.first;
    mindspore::MSTensor tensor = ms_model_.GetOutputByTensorName(tensor_name);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Get tensor failed, tensor name: " << tensor_name;
      return RET_ERROR;
    }
    int ret;
    if (static_cast<int>(tensor.DataType()) == kObjectTypeString) {
      std::vector<std::string> output_strings = MSTensor::TensorToStrings(tensor);
      ret = CompareStringData(tensor_name, calib_tensor.second->strings_data, output_strings);
    } else {
      ret = CompareDataGetTotalCosineDistanceAndSize(tensor_name, &tensor, &total_cosine_distance, &total_size);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Error in CompareData";
      std::cerr << "Error in CompareData" << std::endl;
      std::cout << "=======================================================" << std::endl << std::endl;
      return ret;
    }
  }
  float mean_cosine_distance;
  if (total_size != 0) {
    mean_cosine_distance = total_cosine_distance / float_t(total_size);
  } else {
    mean_cosine_distance = CosineErrMaxVal;
  }
  mean_cosine_distance = 1 - mean_cosine_distance;
  std::cout << "Cosine distance of all nodes/tensors: " << std::setprecision(std::numeric_limits<double>::digits10)
            << mean_cosine_distance << std::endl;
  std::cout << "=======================================================" << std::endl << std::endl;

  if (mean_cosine_distance < cosine_distance_threshold) {
    MS_LOG(ERROR) << "cosine distance of all nodes/tensors is too small: " << mean_cosine_distance;
    std::cerr << "Mean cosine distance of all nodes/tensors is too small: " << mean_cosine_distance << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::CompareDataGetTotalBiasAndSize(const std::string &name, mindspore::MSTensor *tensor,
                                                        float *total_bias, int *total_size, float relative_tolerance,
                                                        float absolute_tolerance) {
  float bias = 0;
  auto mutableData = tensor->MutableData();
  if (mutableData == nullptr) {
    MS_LOG(ERROR) << "mutableData is nullptr.";
    return RET_ERROR;
  }
  switch (static_cast<int>(tensor->DataType())) {
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32: {
      bias = CompareData<float, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      bias = CompareData<int8_t, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeUInt8: {
      bias = CompareData<uint8_t, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      bias = CompareData<int32_t, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      bias = CompareData<int16_t, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeBool: {
      bias = CompareData<bool, int64_t>(name, tensor->Shape(), mutableData, relative_tolerance, absolute_tolerance);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      size_t shapeSize = 1;
      for (int64_t dim : tensor->Shape()) {
        if (dim <= 0) {
          MS_LOG(ERROR) << "The shape of output " << name << " should be great than 0 after inference, got "
                        << tensor->Shape();
          return RET_ERROR;
        }
        MS_CHECK_FALSE_MSG(SIZE_MUL_OVERFLOW(shapeSize, static_cast<size_t>(dim)), RET_ERROR, "mul overflow");
        shapeSize *= static_cast<size_t>(dim);
      }
      auto *floatArr = new float[shapeSize];
      for (size_t i = 0; i < shapeSize; ++i) {
        uint16_t tmpInt = reinterpret_cast<uint16_t *>(mutableData)[i];
        floatArr[i] = ShortToFloat32(tmpInt);
      }
      bias = CompareData<float, int64_t>(name, tensor->Shape(), floatArr);
      delete[] floatArr;
      break;
    }
    default:
      MS_LOG(ERROR) << "Datatype " << static_cast<int>(tensor->DataType()) << " is not supported.";
      return RET_ERROR;
  }
  if (bias < 0) {
    MS_LOG(ERROR) << "CompareData failed, name: " << name;
    return RET_ERROR;
  }
  *total_bias += bias;
  *total_size += 1;
  return RET_OK;
}
int BenchmarkUnifiedApi::CompareDataGetTotalCosineDistanceAndSize(const std::string &name, mindspore::MSTensor *tensor,
                                                                  float *total_cosine_distance, int *total_size) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr.";
    return RET_ERROR;
  }
  if (total_cosine_distance == nullptr) {
    MS_LOG(ERROR) << "total_cosine_distance is nullptr.";
    return RET_ERROR;
  }
  if (total_size == nullptr) {
    MS_LOG(ERROR) << "total_size is nullptr.";
    return RET_ERROR;
  }
  float bias = 0;
  auto mutableData = tensor->MutableData();
  if (mutableData == nullptr) {
    MS_LOG(ERROR) << "mutableData is nullptr.";
    return RET_ERROR;
  }
  int res = RET_OK;
  switch (static_cast<int>(tensor->DataType())) {
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32: {
      res = CompareDatabyCosineDistance<float>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    case TypeId::kNumberTypeFloat16: {
      size_t shapeSize = 1;
      for (int64_t dim : tensor->Shape()) {
        if (dim <= 0) {
          MS_LOG(ERROR) << "Invalid shape.";
          return RET_ERROR;
        }
        MS_CHECK_FALSE_MSG(SIZE_MUL_OVERFLOW(shapeSize, static_cast<size_t>(dim)), RET_ERROR, "mul overflow");
        shapeSize *= static_cast<size_t>(dim);
      }
      float *floatArr = new float[shapeSize];
      for (size_t i = 0; i < shapeSize; ++i) {
        uint16_t tmpInt = reinterpret_cast<uint16_t *>(mutableData)[i];
        Convert2Float32(&floatArr[i], tmpInt);
        reinterpret_cast<float *>(mutableData)[i] = floatArr[i];
      }
      delete[] floatArr;
      bias = CompareData<float, int64_t>(name, tensor->Shape(), mutableData);
      break;
    }
    case TypeId::kNumberTypeInt8: {
      res = CompareDatabyCosineDistance<int8_t>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    case TypeId::kNumberTypeUInt8: {
      res = CompareDatabyCosineDistance<uint8_t>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    case TypeId::kNumberTypeInt32: {
      res = CompareDatabyCosineDistance<int32_t>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    case TypeId::kNumberTypeInt16: {
      res = CompareDatabyCosineDistance<int16_t>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    case TypeId::kNumberTypeBool: {
      res = CompareDatabyCosineDistance<bool>(name, tensor->Shape(), mutableData, &bias);
      break;
    }
    default:
      MS_LOG(ERROR) << "Datatype " << static_cast<int>(tensor->DataType()) << " is not supported.";
      return RET_ERROR;
  }
  if (res != RET_OK) {
    MS_LOG(ERROR) << "CompareData failed, name: " << name;
    return RET_ERROR;
  }
  *total_cosine_distance += 1 - bias;
  *total_size += 1;
  return RET_OK;
}

int BenchmarkUnifiedApi::MarkPerformance() {
  MS_LOG(INFO) << "Running warm up loops...";
  std::cout << "Running warm up loops..." << std::endl;
  std::vector<MSTensor> outputs;

  for (int i = 0; i < flags_->warm_up_loop_count_; i++) {
    auto status = ms_model_.Predict(ms_inputs_for_api_, &outputs);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Inference error ";
      std::cerr << "Inference error " << std::endl;
      return RET_ERROR;
    }
  }

  MS_LOG(INFO) << "Running benchmark loops...";
  std::cout << "Running benchmark loops..." << std::endl;
  uint64_t time_min = 1000000;
  uint64_t time_max = 0;
  uint64_t time_avg = 0;

  for (int i = 0; i < flags_->loop_count_; i++) {
    auto inputs = ms_model_.GetInputs();
    for (auto tensor : inputs) {
      tensor.MutableData();  // prepare data
    }
    auto start = GetTimeUs();
    auto status = ms_model_.Predict(ms_inputs_for_api_, &outputs, ms_before_call_back_, ms_after_call_back_);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Inference error ";
      std::cerr << "Inference error ";
      return RET_ERROR;
    }

    auto end = GetTimeUs();
    auto time = end - start;
    time_min = std::min(time_min, time);
    time_max = std::max(time_max, time);
    time_avg += time;
  }

  if (flags_->time_profiling_) {
    const std::vector<std::string> per_op_name = {"opName", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    const std::vector<std::string> per_op_type = {"opType", "avg(ms)", "percent", "calledTimes", "opTotalTime"};
    (void)PrintResult(per_op_name, op_times_by_name_);
    (void)PrintResult(per_op_type, op_times_by_type_);
#ifdef ENABLE_ARM64
  } else if (flags_->perf_profiling_) {
    if (flags_->perf_event_ == "CACHE") {
      const std::vector<std::string> per_op_name = {"opName", "cache ref(k)", "cache ref(%)", "miss(k)", "miss(%)"};
      const std::vector<std::string> per_op_type = {"opType", "cache ref(k)", "cache ref(%)", "miss(k)", "miss(%)"};
      (void)PrintPerfResult(per_op_name, op_perf_by_name_);
      (void)PrintPerfResult(per_op_type, op_perf_by_type_);
    } else if (flags_->perf_event_ == "STALL") {
      const std::vector<std::string> per_op_name = {"opName", "frontend(k)", "frontend(%)", "backendend(k)",
                                                    "backendend(%)"};
      const std::vector<std::string> per_op_type = {"opType", "frontend(k)", "frontend(%)", "backendend(k)",
                                                    "backendend(%)"};
      (void)PrintPerfResult(per_op_name, op_perf_by_name_);
      (void)PrintPerfResult(per_op_type, op_perf_by_type_);
    } else {
      const std::vector<std::string> per_op_name = {"opName", "cycles(k)", "cycles(%)", "ins(k)", "ins(%)"};
      const std::vector<std::string> per_op_type = {"opType", "cycles(k)", "cycles(%)", "ins(k)", "ins(%)"};
      (void)PrintPerfResult(per_op_name, op_perf_by_name_);
      (void)PrintPerfResult(per_op_type, op_perf_by_type_);
    }
#endif
  }

  if (flags_->loop_count_ > 0) {
    time_avg /= static_cast<size_t>(flags_->loop_count_);
    MS_LOG(INFO) << "Model = " << flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str()
                 << ", NumThreads = " << flags_->num_threads_ << ", MinRunTime = " << time_min / kFloatMSEC
                 << ", MaxRuntime = " << time_max / kFloatMSEC << ", AvgRunTime = " << time_avg / kFloatMSEC;
    printf("Model = %s, NumThreads = %d, MinRunTime = %f ms, MaxRuntime = %f ms, AvgRunTime = %f ms\n",
           flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1).c_str(), flags_->num_threads_,
           time_min / kFloatMSEC, time_max / kFloatMSEC, time_avg / kFloatMSEC);
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::MarkAccuracy() {
  MS_LOG(INFO) << "MarkAccuracy";
  std::cout << "MarkAccuracy" << std::endl;

  int status = 0;
  if (flags_->enable_gl_texture_) {
    for (auto in_tensor : ms_inputs_for_api_) {
      auto *input = reinterpret_cast<GLuint *>(in_tensor.MutableData());
      if (input == nullptr) {
        MS_LOG(ERROR) << "get input data failed";
        return RET_ERROR;
      }
      float *hostptr = reinterpret_cast<float *>(gl_runtime_.CopyDeviceTextureToHost(*input));
      size_t print_num = 20;
      gl_runtime_.PrintImage2DData(hostptr, 1, 1, print_num);
    }
  } else {
    status = PrintInputData();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "PrintInputData error " << status;
      std::cerr << "PrintInputData error " << status << std::endl;
      return status;
    }
  }
  std::vector<MSTensor> outputs;
  auto ret = ms_model_.Predict(ms_inputs_for_api_, &outputs, ms_before_call_back_, ms_after_call_back_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Inference error ";
    std::cerr << "Inference error " << std::endl;
    return RET_ERROR;
  }
  status = ReadCalibData();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Read calib data error " << status;
    std::cerr << "Read calib data error " << status << std::endl;
    return status;
  }
  status = CompareOutput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compare output error " << status;
    std::cerr << "Compare output error " << status << std::endl;
    return status;
  }
  if (this->flags_->cosine_distance_threshold_ >= -1) {
    status = CompareOutputByCosineDistance(this->flags_->cosine_distance_threshold_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Compare output error by consine distance " << status;
      std::cerr << "Compare output error by consine distance" << status << std::endl;
      return status;
    }
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::PrintInputData() {
  for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
    mindspore::MSTensor input = ms_inputs_for_api_[i];
    auto tensor_data_type = static_cast<int>(input.DataType());

    std::cout << "InData " << i << ": ";
    if (tensor_data_type == TypeId::kNumberTypeFloat16) {
      MS_LOG(INFO) << "DataType: " << TypeId::kNumberTypeFloat16;
      continue;
    }
    if (tensor_data_type == TypeId::kObjectTypeString) {
      std::vector<std::string> output_strings = MSTensor::TensorToStrings(input);
      size_t print_num = std::min(output_strings.size(), static_cast<size_t>(20));
      for (size_t j = 0; j < print_num; j++) {
        std::cout << output_strings[j] << std::endl;
      }
      continue;
    }
    size_t print_num = std::min(static_cast<int>(input.ElementNum()), kPrintDataNum);
    const void *in_data = input.MutableData();
    if (in_data == nullptr) {
      MS_LOG(ERROR) << "in_data is nullptr.";
      return RET_ERROR;
    }

    for (size_t j = 0; j < print_num; j++) {
      if (tensor_data_type == TypeId::kNumberTypeFloat32 || tensor_data_type == TypeId::kNumberTypeFloat) {
        std::cout << static_cast<const float *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt8) {
        std::cout << static_cast<const int8_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeUInt8) {
        std::cout << static_cast<const uint8_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt32) {
        std::cout << static_cast<const int32_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeInt64) {
        std::cout << static_cast<const int64_t *>(in_data)[j] << " ";
      } else if (tensor_data_type == TypeId::kNumberTypeBool) {
        std::cout << static_cast<const bool *>(in_data)[j] << " ";
      } else {
        MS_LOG(ERROR) << "Datatype: " << tensor_data_type << " is not supported.";
        return RET_ERROR;
      }
    }
    std::cout << std::endl;
  }
  return RET_OK;
}
#ifdef PARALLEL_INFERENCE
void BenchmarkUnifiedApi::ModelParallelRunnerWarmUp(int index) {
  auto in = model_runner_.GetInputs();
  for (size_t i = 0; i < in.size(); i++) {
    in[i].SetShape(resize_dims_[i]);
    in[i].SetData(all_inputs_data_[index][i], false);
  }
  auto warm_up_start = GetTimeUs();
  std::vector<MSTensor> output;
  auto ret = model_runner_.Predict(in, &output);
  for (size_t j = 0; j < in.size(); j++) {
    in[j].SetData(nullptr);
  }
  if (ret != kSuccess) {
    model_parallel_runner_ret_failed_ = true;
    MS_LOG(ERROR) << "model pool predict failed.";
    return;
  }
  auto warm_up_end = GetTimeUs();
  std::cout << "warm up index: " << index << " | time: " << (warm_up_end - warm_up_start) / kFloatMSEC << " ms\n";
}

void BenchmarkUnifiedApi::ModelParallelRunnerRun(int task_num, int parallel_idx) {
  for (int i = 0; i < task_num || task_num == -1; i++) {
    while (!runner_run_start_) {
      continue;
    }
    int idx = parallel_idx + flags_->warm_up_loop_count_;
    auto in = model_runner_.GetInputs();
    if (idx >= static_cast<int>(all_inputs_data_.size())) {
      MS_LOG(ERROR) << "idx is to big :" << idx;
      return;
    }
    auto in_data = all_inputs_data_[idx];
    for (size_t tensor_index = 0; tensor_index < in.size(); tensor_index++) {
      in.at(tensor_index).SetShape(resize_dims_.at(tensor_index));
      in.at(tensor_index).SetData(all_inputs_data_.at(idx)[tensor_index], false);
    }
    auto predict_start = GetTimeUs();
    std::vector<MSTensor> output;
    auto ret = model_runner_.Predict(in, &output);
    if (ret != kSuccess) {
      model_parallel_runner_ret_failed_ = true;
      MS_LOG(ERROR) << "model pool predict failed.";
      for (auto &item : in) {
        item.SetData(nullptr);
      }
      return;
    }
    auto predict_end = GetTimeUs();
    std::cout << "parallel index: " << parallel_idx << " | task index: " << i
              << " | predict time: " << (predict_end - predict_start) / kFloatMSEC << " ms\n";
    for (size_t j = 0; j < in.size(); j++) {
      in[j].SetData(nullptr);
    }
    if (!flags_->benchmark_data_file_.empty()) {
      auto status = CompareOutputForModelPool(&output);
      if (status != RET_OK) {
        model_parallel_runner_ret_failed_ = true;
        MS_LOG(ERROR) << "Compare output error " << status;
        return;
      }
    }
  }
}

int BenchmarkUnifiedApi::AddConfigInfo(const std::shared_ptr<RunnerConfig> &runner_config) {
  if (!flags_->config_file_.empty()) {
    runner_config->SetConfigPath(flags_->config_file_);
  }
  std::map<std::string, std::string> config;
  if (flags_->enable_shared_thread_pool_) {
    config[kEnableSharedThreadPoolKey] = "true";
    if (!flags_->thread_num_limit_per_worker_.empty()) {
      config[kThreadNumLimitPerWorkerKey] = flags_->thread_num_limit_per_worker_;
    }
    if (!flags_->thread_num_remaining_per_worker_.empty()) {
      config[kThreadNumRemainingPerWorkerKey] = flags_->thread_num_remaining_per_worker_;
    }
  } else {
    config[kEnableSharedThreadPoolKey] = "false";
  }
  runner_config->SetConfigInfo(kSharedThreadPoolSection, config);
  return RET_OK;
}

int BenchmarkUnifiedApi::ParallelInference(std::shared_ptr<mindspore::Context> context) {
  if (flags_->warm_up_loop_count_ > kMaxRequestNum || flags_->parallel_num_ > kMaxRequestNum) {
    MS_LOG(WARNING) << "in parallel predict warm up loop count should less than" << kMaxRequestNum;
  }

  (void)std::transform(flags_->resize_dims_.begin(), flags_->resize_dims_.end(), std::back_inserter(resize_dims_),
                       [&](auto &shapes) { return this->ConverterToInt64Vector<int>(shapes); });

  // model runner init
  auto runner_config = std::make_shared<RunnerConfig>();
  runner_config->SetContext(context);
  runner_config->SetWorkersNum(flags_->workers_num_);
  auto status = AddConfigInfo(runner_config);
  MS_CHECK_FALSE_MSG(status != kSuccess, RET_ERROR, "add config info for parallel predict failed.");
  auto model_init_start = GetTimeUs();
  auto ret = model_runner_.Init(flags_->model_file_, runner_config);
  MS_CHECK_FALSE_MSG(ret != kSuccess, RET_ERROR, "model pool init failed.");
  auto model_init_end = GetTimeUs();

  // load data
  ms_inputs_for_api_ = model_runner_.GetInputs();
  MS_CHECK_FALSE_MSG(ms_inputs_for_api_.empty(), RET_ERROR, "model pool input is empty.");
  ms_outputs_for_api_ = model_runner_.GetOutputs();
  MS_CHECK_FALSE_MSG(ms_outputs_for_api_.empty(), RET_ERROR, "model pool output is empty.");
  for (int i = 0; i < flags_->parallel_num_ + flags_->warm_up_loop_count_; i++) {
    status = LoadInput();
    MS_CHECK_FALSE_MSG(status != RET_OK, status, "Generate input data error");
    std::vector<MSTensor> output;
    all_outputs_.push_back(output);
  }
  if (!flags_->benchmark_data_file_.empty()) {
    for (size_t i = 0; i < ms_inputs_for_api_.size(); i++) {
      auto &tensor = ms_inputs_for_api_[i];
      tensor.SetShape(resize_dims_[i]);
      tensor.SetData(all_inputs_data_[0][i], false);
    }
    status = PrintInputData();
    MS_CHECK_FALSE_MSG(status != RET_OK, status, "PrintInputData error ");
    status = ReadCalibData();
    MS_CHECK_FALSE_MSG(status != RET_OK, status, "ReadCalibData error ");
  }

  // warm up
  std::vector<std::thread> model_thread_warm_up;
  for (int i = 0; i < flags_->warm_up_loop_count_; i++) {
    model_thread_warm_up.push_back(std::thread(&BenchmarkUnifiedApi::ModelParallelRunnerWarmUp, this, i));
  }
  for (auto &warm_up_thread : model_thread_warm_up) {
    warm_up_thread.join();
  }
  if (model_parallel_runner_ret_failed_) {
    return RET_ERROR;
  }
  std::cout << "=============== end warm up ===============\n";
  // do loop count
  std::vector<std::thread> model_thread_run;
  for (int parallel_num_idx = 0; parallel_num_idx < flags_->parallel_num_; parallel_num_idx++) {
    model_thread_run.push_back(
      std::thread(&BenchmarkUnifiedApi::ModelParallelRunnerRun, this, flags_->parallel_task_num_, parallel_num_idx));
  }
  auto start_run_time = lite::GetTimeUs();
  runner_run_start_ = true;
  for (auto &run_thread : model_thread_run) {
    run_thread.join();
  }
  auto end_run_time = lite::GetTimeUs();
  if (model_parallel_runner_ret_failed_) {
    return RET_ERROR;
  }
  std::cout << "=================================" << std::endl;
  std::cout << "parallel predict init time: " << (model_init_end - model_init_start) / kFloatMSEC << " ms\n";
  std::cout << "parallel predict all run time: " << (end_run_time - start_run_time) / kFloatMSEC << " ms\n";
  std::cout << "=================================" << std::endl;
  return RET_OK;
}
#endif

int BenchmarkUnifiedApi::CompileGraph(mindspore::ModelType model_type, const std::shared_ptr<Context> &context,
                                      const std::string &model_name) {
  Key dec_key;
  if (!flags_->decrypt_key_str_.empty()) {
    dec_key.len = lite::Hex2ByteArray(flags_->decrypt_key_str_, dec_key.key, kEncMaxLen);
    if (dec_key.len == 0) {
      MS_LOG(ERROR) << "dec_key.len == 0";
      return RET_INPUT_PARAM_INVALID;
    }
    flags_->decrypt_key_str_.clear();
  }
  Status ret;
  if (flags_->crypto_lib_path_.empty()) {
    ret = ms_model_.Build(flags_->model_file_, model_type, context);
  } else {
    ret =
      ms_model_.Build(flags_->model_file_, model_type, context, dec_key, flags_->dec_mode_, flags_->crypto_lib_path_);
  }
  memset(dec_key.key, 0, kEncMaxLen);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ms_model_.Build failed while running ", model_name.c_str();
    std::cout << "ms_model_.Build failed while running ", model_name.c_str();
    return RET_ERROR;
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::RunBenchmark() {
  auto start_prepare_time = GetTimeUs();

  if (flags_->enable_gl_texture_) {
    if (!gl_runtime_.Init()) {
      MS_LOG(ERROR) << "opengl runtime init failed ";
      std::cerr << "opengl runtime init failed ";
      return RET_ERROR;
    }
  }

  // Load graph
  std::string model_name = flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1);
  auto iter = ModelTypeMap.find(flags_->model_type_);
  if (iter == ModelTypeMap.end()) {
    MS_LOG(ERROR) << "model_type " << flags_->model_type_ << " is invalid.";
    std::cerr << "model_type " << flags_->model_type_ << " is invalid.";
    return RET_ERROR;
  }
  mindspore::ModelType model_type = iter->second;

  MS_LOG(INFO) << "start unified benchmark run";
  std::cout << "start unified benchmark run" << std::endl;

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running " << model_name.c_str();
    std::cerr << "New context failed while running " << model_name.c_str() << std::endl;
    return RET_ERROR;
  }

  auto status = InitMSContext(context);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "InitMSContext failed while running " << model_name.c_str();
    std::cout << "InitMSContext failed while running " << model_name.c_str();
    return RET_ERROR;
  }

  (void)UpdateDistributionName(context, &flags_->model_file_);
  (void)UpdateDistributionName(context, &flags_->benchmark_data_file_);
  (void)UpdateDistributionName(context, &flags_->config_file_);

  if (!flags_->config_file_.empty()) {
    auto config_ret = ms_model_.LoadConfig(flags_->config_file_);
    if (config_ret != kSuccess) {
      MS_LOG(ERROR) << "ms_model_.LoadConfig failed while running ", model_name.c_str();
      std::cout << "ms_model_.LoadConfig failed while running ", model_name.c_str();
    }
  }

  UpdateConfigInfo();
#ifdef PARALLEL_INFERENCE
  if (flags_->enable_parallel_predict_) {
    MS_CHECK_FALSE_MSG(flags_->resize_dims_.empty(), RET_ERROR, "use parallel predict, inputShapes can not use empty.");
    status = ParallelInference(context);
    MS_CHECK_FALSE_MSG(status != RET_OK, RET_ERROR, "run model pool failed.");
    return RET_OK;
  }
#endif

  status = CompileGraph(model_type, context, model_name);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Compile graph failed.";
    return status;
  }
  if (!flags_->resize_dims_.empty()) {
    std::vector<std::vector<int64_t>> resize_dims;
    (void)std::transform(flags_->resize_dims_.begin(), flags_->resize_dims_.end(), std::back_inserter(resize_dims),
                         [&](auto &shapes) { return this->ConverterToInt64Vector<int>(shapes); });

    auto ret = ms_model_.Resize(ms_model_.GetInputs(), resize_dims);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Input tensor resize failed.";
      std::cout << "Input tensor resize failed.";
      return RET_ERROR;
    }
  }

  ms_inputs_for_api_ = ms_model_.GetInputs();
  ms_outputs_for_api_ = ms_model_.GetOutputs();
  auto end_prepare_time = GetTimeUs();
  MS_LOG(INFO) << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kFloatMSEC) << " ms";
  std::cout << "PrepareTime = " << ((end_prepare_time - start_prepare_time) / kFloatMSEC) << " ms" << std::endl;

  // Load input
  MS_LOG(INFO) << "start generate input data";
  status = LoadInput();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Generate input data error";
    return status;
  }
  if (!flags_->benchmark_data_file_.empty()) {
    status = MarkAccuracy();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkAccuracy error: " << status;
      std::cout << "Run MarkAccuracy error: " << status << std::endl;
      return status;
    }
  } else {
    status = MarkPerformance();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run MarkPerformance error: " << status;
      std::cout << "Run MarkPerformance error: " << status << std::endl;
      return status;
    }
  }
  if (flags_->dump_tensor_data_) {
    std::cout << "Dumped file is saved to : " + dump_file_output_dir_ << std::endl;
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::InitTimeProfilingCallbackParameter() {
  if (flags_->inter_op_parallel_num_ > 1) {
    // before callback
    ms_before_call_back_ = [&, this](const std::vector<mindspore::MSTensor> &before_inputs,
                                     const std::vector<mindspore::MSTensor> &before_outputs,
                                     const MSCallBackParam &call_param) {
      if (before_inputs.empty()) {
        MS_LOG(INFO) << "The num of beforeInputs is empty";
      }
      if (before_outputs.empty()) {
        MS_LOG(INFO) << "The num of beforeOutputs is empty";
      }
      {
        std::lock_guard<std::mutex> _l(op_times_mutex_);
        if (op_times_by_type_.find(call_param.node_type) == op_times_by_type_.end()) {
          op_times_by_type_.insert(std::make_pair(call_param.node_type, std::make_pair(0, 0.0f)));
        }
        if (op_times_by_name_.find(call_param.node_name) == op_times_by_name_.end()) {
          op_times_by_name_.insert(std::make_pair(call_param.node_name, std::make_pair(0, 0.0f)));
        }
        op_start_times_by_name_[call_param.node_name] = GetTimeUs();
        op_call_times_total_++;
      }
      return true;
    };

    // after callback
    ms_after_call_back_ = [&, this](const std::vector<mindspore::MSTensor> &after_inputs,
                                    const std::vector<mindspore::MSTensor> &after_outputs,
                                    const MSCallBackParam &call_param) {
      uint64_t opEnd = GetTimeUs();

      if (after_inputs.empty()) {
        MS_LOG(INFO) << "The num of after inputs is empty";
      }
      if (after_outputs.empty()) {
        MS_LOG(INFO) << "The num of after outputs is empty";
      }
      {
        std::lock_guard<std::mutex> _l(op_times_mutex_);
        float cost = static_cast<float>(opEnd - op_start_times_by_name_[call_param.node_name]) / kFloatMSEC;
        if (flags_->device_ == "GPU") {
          cost = static_cast<float>(call_param.execute_time);
        }
        op_cost_total_ += cost;
        op_times_by_type_[call_param.node_type].first++;
        op_times_by_type_[call_param.node_type].second += cost;
        op_times_by_name_[call_param.node_name].first++;
        op_times_by_name_[call_param.node_name].second += cost;
      }
      return true;
    };
  } else {
    // before callback
    ms_before_call_back_ = [&, this](const std::vector<mindspore::MSTensor> &before_inputs,
                                     const std::vector<mindspore::MSTensor> &before_outputs,
                                     const MSCallBackParam &call_param) {
      if (before_inputs.empty()) {
        MS_LOG(INFO) << "The num of beforeInputs is empty";
      }
      if (before_outputs.empty()) {
        MS_LOG(INFO) << "The num of beforeOutputs is empty";
      }
      if (op_times_by_type_.find(call_param.node_type) == op_times_by_type_.end()) {
        op_times_by_type_.insert(std::make_pair(call_param.node_type, std::make_pair(0, 0.0f)));
      }
      if (op_times_by_name_.find(call_param.node_name) == op_times_by_name_.end()) {
        op_times_by_name_.insert(std::make_pair(call_param.node_name, std::make_pair(0, 0.0f)));
      }

      op_call_times_total_++;
      op_begin_ = GetTimeUs();
      return true;
    };

    // after callback
    ms_after_call_back_ = [&, this](const std::vector<mindspore::MSTensor> &after_inputs,
                                    const std::vector<mindspore::MSTensor> &after_outputs,
                                    const MSCallBackParam &call_param) {
      uint64_t opEnd = GetTimeUs();

      if (after_inputs.empty()) {
        MS_LOG(INFO) << "The num of after inputs is empty";
      }
      if (after_outputs.empty()) {
        MS_LOG(INFO) << "The num of after outputs is empty";
      }

      float cost = static_cast<float>(opEnd - op_begin_) / kFloatMSEC;
      if (flags_->device_ == "GPU") {
        cost = static_cast<float>(call_param.execute_time);
      }
      op_cost_total_ += cost;
      op_times_by_type_[call_param.node_type].first++;
      op_times_by_type_[call_param.node_type].second += cost;
      op_times_by_name_[call_param.node_name].first++;
      op_times_by_name_[call_param.node_name].second += cost;
      return true;
    };
  }
  return RET_OK;
}

int BenchmarkUnifiedApi::InitPerfProfilingCallbackParameter() {
#ifndef ENABLE_ARM64
  MS_LOG(ERROR) << "Only support perf_profiling on arm64.";
  return RET_ERROR;
#else
  struct perf_event_attr pe, pe2;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  memset(&pe2, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe2.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
  pe2.size = sizeof(struct perf_event_attr);
  pe.disabled = 1;
  pe2.disabled = 1;
  pe.exclude_kernel = 1;   // don't count kernel
  pe2.exclude_kernel = 1;  // don't count kernel
  pe.exclude_hv = 1;       // don't count hypervisor
  pe2.exclude_hv = 1;      // don't count hypervisor
  pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  pe2.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  if (flags_->perf_event_ == "CACHE") {
    pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
    pe2.config = PERF_COUNT_HW_CACHE_MISSES;
  } else if (flags_->perf_event_ == "STALL") {
    pe.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
    pe2.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
  } else {
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe2.config = PERF_COUNT_HW_INSTRUCTIONS;
  }
  perf_fd = syscall(__NR_perf_event_open, pe, 0, -1, -1, 0);
  if (perf_fd == -1) {
    MS_LOG(ERROR) << "Failed to open perf event " << pe.config;
    return RET_ERROR;
  }
  perf_fd2 = syscall(__NR_perf_event_open, pe2, 0, -1, perf_fd, 0);
  if (perf_fd2 == -1) {
    MS_LOG(ERROR) << "Failed to open perf event " << pe2.config;
    return RET_ERROR;
  }
  struct PerfCount zero;
  zero.value[0] = 0;
  zero.value[1] = 0;
  // before callback
  ms_before_call_back_ = [&](const std::vector<mindspore::MSTensor> &before_inputs,
                             const std::vector<mindspore::MSTensor> &before_outputs,
                             const MSCallBackParam &call_param) {
    if (before_inputs.empty()) {
      MS_LOG(INFO) << "The num of beforeInputs is empty";
    }
    if (before_outputs.empty()) {
      MS_LOG(INFO) << "The num of beforeOutputs is empty";
    }
    if (op_perf_by_type_.find(call_param.node_type) == op_perf_by_type_.end()) {
      op_perf_by_type_.insert(std::make_pair(call_param.node_type, std::make_pair(0, zero)));
    }
    if (op_perf_by_name_.find(call_param.node_name) == op_perf_by_name_.end()) {
      op_perf_by_name_.insert(std::make_pair(call_param.node_name, std::make_pair(0, zero)));
    }

    op_call_times_total_++;
    ioctl(perf_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    return true;
  };

  // after callback
  ms_after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                            const std::vector<mindspore::MSTensor> &after_outputs, const MSCallBackParam &call_param) {
    struct PerfResult res;
    ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    if (read(perf_fd, &res, sizeof(struct PerfResult)) == -1) {
      MS_LOG(ERROR) << "Failed to read perf_fd";
      return false;
    }

    if (after_inputs.empty()) {
      MS_LOG(INFO) << "The num of after inputs is empty";
    }
    if (after_outputs.empty()) {
      MS_LOG(INFO) << "The num of after outputs is empty";
    }
    float cost1 = static_cast<float>(res.values[0].value);
    float cost2 = static_cast<float>(res.values[1].value);
    op_cost_total_ += cost1;
    op_cost2_total_ += cost2;
    op_perf_by_type_[call_param.node_type].first++;
    op_perf_by_type_[call_param.node_type].second.value[0] += cost1;
    op_perf_by_type_[call_param.node_type].second.value[1] += cost2;
    op_perf_by_name_[call_param.node_name].first++;
    op_perf_by_name_[call_param.node_name].second.value[0] += cost1;
    op_perf_by_name_[call_param.node_name].second.value[1] += cost2;
    return true;
  };
#endif
  return RET_OK;
}

namespace {
template <typename T>
std::string DataToString(void *data, size_t data_number) {
  if (data == nullptr) {
    return "Data of tensor is nullptr";
  }
  std::ostringstream oss;
  auto casted_data = static_cast<T *>(data);
  for (size_t i = 0; i < kDataToStringMaxNum && i < data_number; i++) {
    oss << " " << casted_data[i];
  }
  return oss.str();
}

std::string DumpMSTensor(mindspore::MSTensor *tensor) {
  if (tensor == nullptr) {
    return "Tensor is nullptr";
  }
  std::ostringstream oss;
  oss << " DataType: " << static_cast<int>(tensor->DataType());
  oss << " Shape:";
  for (auto &dim : tensor->Shape()) {
    oss << " " << dim;
  }
  oss << std::endl << " Data:";
  switch (static_cast<int>(tensor->DataType())) {
    case kNumberTypeFloat32: {
      oss << DataToString<float>(tensor->MutableData(), tensor->ElementNum());
    } break;
    case kNumberTypeFloat16: {
      oss << DataToString<int16_t>(tensor->MutableData(), tensor->ElementNum());
    } break;
    case kNumberTypeInt32: {
      oss << DataToString<int32_t>(tensor->MutableData(), tensor->ElementNum());
    } break;
    case kNumberTypeInt16: {
      oss << DataToString<int16_t>(tensor->MutableData(), tensor->ElementNum());
    } break;
    case kNumberTypeInt8: {
      oss << DataToString<int8_t>(tensor->MutableData(), tensor->ElementNum());
    } break;
    default:
      oss << "Unsupported data type to print";
      break;
  }
  return oss.str();
}
#ifndef BENCHMARK_CLIP_JSON
std::string GenerateOutputFileName(mindspore::MSTensor *tensor, const std::string &op_name,
                                   const std::string &file_type, const size_t &idx) {
  std::string file_name = op_name;
  auto pos = file_name.find_first_of('/');
  while (pos != std::string::npos) {
    file_name.replace(pos, 1, ".");
    pos = file_name.find_first_of('/');
  }
  file_name += "_" + file_type + "_" + std::to_string(idx) + "_shape_";
  for (const auto &dim : tensor->Shape()) {
    file_name += std::to_string(dim) + "_";
  }
  if (kTypeIdMap.find(static_cast<int>(tensor->DataType())) != kTypeIdMap.end()) {
    file_name += kTypeIdMap.at(static_cast<int>(tensor->DataType()));
  }
  auto tensor_format = tensor->format();
  if (kTensorFormatMap.find(tensor_format) != kTensorFormatMap.end()) {
    file_name += "_" + kTensorFormatMap.at(tensor_format) + ".bin";
  }

  file_name += +".bin";
  return file_name;
}
#endif
}  // namespace

int BenchmarkUnifiedApi::InitPrintTensorDataCallbackParameter() {
  // before callback
  ms_before_call_back_ = [&](const std::vector<mindspore::MSTensor> &before_inputs,
                             const std::vector<mindspore::MSTensor> &before_outputs,
                             const MSCallBackParam &call_param) { return true; };

  // after callback
  ms_after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                            const std::vector<mindspore::MSTensor> &after_outputs, const MSCallBackParam &call_param) {
    std::cout << "================================================================" << std::endl;
    std::cout << call_param.node_name << " inputs : " << std::endl;
    for (auto ms_tensor : after_inputs) {
      std::cout << DumpMSTensor(&ms_tensor) << std::endl;
    }
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << call_param.node_name << " outputs : " << std::endl;
    for (auto ms_tensor : after_outputs) {
      std::cout << DumpMSTensor(&ms_tensor) << std::endl;
    }
    std::cout << "================================================================" << std::endl;
    return true;
  };
  return RET_OK;
}
int BenchmarkUnifiedApi::InitDumpTensorDataCallbackParameter() {
#ifndef BENCHMARK_CLIP_JSON
  // before callback
  ms_before_call_back_ = [&](const std::vector<mindspore::MSTensor> &before_inputs,
                             const std::vector<mindspore::MSTensor> &before_outputs,
                             const MSCallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == 0 || std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == 0 || input_output_mode == 1) {
        for (size_t i = 0; i < before_inputs.size(); i++) {
          auto ms_tensor = before_inputs.at(i);
          auto file_name = GenerateOutputFileName(&ms_tensor, call_param.node_name, "input", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor.MutableData(), ms_tensor.DataSize()) != RET_OK) {  // save to file
            MS_LOG(ERROR) << "write tensor data to file failed.";
            return false;
          }
        }
      }
    }
    return true;
  };

  // after callback
  ms_after_call_back_ = [&](const std::vector<mindspore::MSTensor> &after_inputs,
                            const std::vector<mindspore::MSTensor> &after_outputs, const MSCallBackParam &call_param) {
    auto dump_mode = dump_cfg_json_[dump::kSettings][dump::kMode].get<int>();
    auto input_output_mode = dump_cfg_json_[dump::kSettings][dump::kInputOutput].get<int>();
    auto kernels = dump_cfg_json_[dump::kSettings][dump::kKernels].get<std::vector<std::string>>();
    if (dump_mode == kDumpInputsAndOutputs ||
        std::find(kernels.begin(), kernels.end(), call_param.node_name) != kernels.end()) {
      if (input_output_mode == kDumpInputsAndOutputs || input_output_mode == kDumpOutputs) {
        for (size_t i = 0; i < after_outputs.size(); i++) {
          auto ms_tensor = after_outputs.at(i);
          auto file_name = GenerateOutputFileName(&ms_tensor, call_param.node_name, "output", i);
          auto abs_file_path = dump_file_output_dir_ + "/" + file_name;
          if (WriteToBin(abs_file_path, ms_tensor.MutableData(), ms_tensor.DataSize()) != RET_OK) {  // save to file
            MS_LOG(ERROR) << "write tensor data to file failed.";
            return false;
          }
        }
      }
    }
    return true;
  };
#endif
  return RET_OK;
}

BenchmarkUnifiedApi::~BenchmarkUnifiedApi() {
#ifdef PARALLEL_INFERENCE
  for (auto tensor : ms_inputs_for_api_) {
    auto data = tensor.MutableData();
    if (data != nullptr) {
      tensor.SetData(nullptr);
    }
  }
  for (auto &input : all_inputs_data_) {
    for (auto &data : input) {
      if (data != nullptr) {
        auto buf = static_cast<char *>(data);
        delete[] buf;
        data = nullptr;
      }
    }
  }
#endif
}
}  // namespace lite
}  // namespace mindspore
