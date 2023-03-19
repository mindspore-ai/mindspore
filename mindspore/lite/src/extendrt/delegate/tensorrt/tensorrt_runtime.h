/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_RUNTIME_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_RUNTIME_H_
#include <NvInfer.h>
#include "include/errorcode.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_allocator.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cublas_utils.h"
#include "src/common/log_adapter.h"
#define MAX_BATCH_SIZE 64

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::lite {
class TensorRTLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
      MS_LOG(ERROR) << msg;
    } else if (severity == Severity::kWARNING) {
      MS_LOG(WARNING) << msg;
    } else if (severity == Severity::kINFO) {
      MS_LOG(INFO) << msg;
    } else {
      MS_LOG(DEBUG) << msg;
    }
  }
};

enum RuntimePrecisionMode : int { RuntimePrecisionMode_FP32, RuntimePrecisionMode_FP16 };

class TensorRTRuntime {
 public:
  TensorRTRuntime() = default;

  ~TensorRTRuntime();

  int Init();

  nvinfer1::IBuilder *GetBuilder() { return this->builder_; }

  int GetBatchSize() { return batch_size_; }

  void SetBatchSize(int batch_size) { batch_size_ = batch_size; }

  void SetCudaStream(cudaStream_t stream, cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle) {
    allocator_->SetCudaStream(stream);
    cublas_handle_ = cublas_handle;
    cublaslt_handle_ = cublaslt_handle;
  }

  RuntimePrecisionMode GetRuntimePrecisionMode() { return runtime_percision_mode_; }

  int GetTransformerEncoderInputIdx() { return transformer_encoder_input_idx_; }

  int GetTransformerDecoderInputIdx() { return transformer_decoder_input_idx_; }

  bool GetTransformerFfnFp16() { return transformer_ffn_fp16_; }

  int GetVslEncoderPluginId() { return vsl_encoder_plugin_id_; }

  int GetVslDecoderPluginId() { return vsl_decoder_plugin_id_; }

  void SetRuntimePrecisionMode(RuntimePrecisionMode runtime_percision_mode) {
    runtime_percision_mode_ = runtime_percision_mode;
  }

  void SetTransformerEncoderInputIdx(int transformer_encoder_input_idx) {
    transformer_encoder_input_idx_ = transformer_encoder_input_idx;
  }

  void SetTransformerDecoderInputIdx(int transformer_decoder_input_idx) {
    transformer_decoder_input_idx_ = transformer_decoder_input_idx;
  }

  void SetTransformerFfnFp16(bool is_ffn_fp16) { transformer_ffn_fp16_ = is_ffn_fp16; }

  void SetVslEncoderPluginId(int plugin_id) { vsl_encoder_plugin_id_ = plugin_id; }

  void SetVslDecoderPluginId(int plugin_id) { vsl_decoder_plugin_id_ = plugin_id; }

  TensorRTAllocator *GetAllocator() { return this->allocator_; }

  void SetDeviceID(uint32_t device_id) { device_id_ = device_id; }

  uint32_t GetDeviceID() { return device_id_; }
  cublasHandle_t GetCublasHandle() { return cublas_handle_; }
  cublasLtHandle_t GetCublasLtHandle() { return cublaslt_handle_; }

 private:
  bool is_init_{false};
  nvinfer1::IBuilder *builder_{nullptr};
  TensorRTLogger logger_;
  TensorRTAllocator *allocator_{nullptr};
  int batch_size_{0};
  uint32_t device_id_{0};
  RuntimePrecisionMode runtime_percision_mode_{RuntimePrecisionMode::RuntimePrecisionMode_FP32};
  int transformer_encoder_input_idx_{-1};
  int transformer_decoder_input_idx_{-1};
  bool transformer_ffn_fp16_{true};
  int vsl_encoder_plugin_id_{-1};
  int vsl_decoder_plugin_id_{-1};
  cublasHandle_t cublas_handle_{nullptr};
  cublasLtHandle_t cublaslt_handle_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_RUNTIME_H_
