/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/tensorrt/op/quant_dtype_cast_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "tools/converter/quantizer/fse_decoder.h"

namespace mindspore::lite {
namespace {
constexpr size_t kTableExtend = 3;
constexpr size_t kAlignOffset = 7;
}  // namespace
using mindspore::lite::quant::FSEDecoder;
QuantDTypeCastTensorRT::~QuantDTypeCastTensorRT() {
  if (int32_align_ != nullptr) {
    free(int32_align_);
    int32_align_ = nullptr;
  }
  if (scale_device_ptr_ != nullptr) {
    cudaFree(scale_device_ptr_);
    scale_device_ptr_ = nullptr;
  }
  if (zp_device_ptr_ != nullptr) {
    cudaFree(zp_device_ptr_);
    zp_device_ptr_ = nullptr;
  }
  if (states_table_ != nullptr) {
    free(states_table_);
    states_table_ = nullptr;
  }
  if (bit_count_table_ != nullptr) {
    free(bit_count_table_);
    bit_count_table_ = nullptr;
  }
  if (symbol_table_ != nullptr) {
    free(symbol_table_);
    symbol_table_ = nullptr;
  }
  if (states_table_device_ != nullptr) {
    cudaFree(scale_device_ptr_);
    states_table_device_ = nullptr;
  }
  if (bit_count_table_device_ != nullptr) {
    cudaFree(bit_count_table_device_);
    bit_count_table_device_ = nullptr;
  }
  if (symbol_table_device_ != nullptr) {
    cudaFree(symbol_table_device_);
    symbol_table_device_ = nullptr;
  }
  if (centroids_device_ != nullptr) {
    cudaFree(centroids_device_);
    centroids_device_ = nullptr;
  }
  if (states_table_ != nullptr) {
    free(scale_device_ptr_);
    scale_device_ptr_ = nullptr;
  }
  if (bit_count_table_ != nullptr) {
    free(bit_count_table_);
    bit_count_table_ = nullptr;
  }
  if (symbol_table_ != nullptr) {
    free(symbol_table_);
    symbol_table_ = nullptr;
  }
}

int QuantDTypeCastTensorRT::IsSupport(const schema::Primitive *primitive,
                                      const std::vector<mindspore::MSTensor> &in_tensors,
                                      const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  quant_dtype_cast_ = this->op_primitive_->value_as_QuantDTypeCast();
  if (quant_dtype_cast_ == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }
  auto int8_dequant = quant_dtype_cast_->src_t() == kNumberTypeInt8 && quant_dtype_cast_->dst_t() == kNumberTypeFloat32;
  if (!int8_dequant) {
    MS_LOG(ERROR) << "Dont support data type, src type is " << quant_dtype_cast_->src_t() << " and dst type is "
                  << quant_dtype_cast_->dst_t();
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastTensorRT::Deserialize(int8_t *data8, size_t data_size) {
  size_t i = 0;
  CHECK_NULL_RETURN(data8);
  // 16bit for frequency_count
  uint16_t frequency_count = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 16bit for table_log
  table_log_ = *(reinterpret_cast<uint16_t *>(&data8[i]));
  i += sizeof(uint16_t);
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 32bit for ChunkCount
  chunk_count_ = *(reinterpret_cast<uint32_t *>(&data8[i]));
  const size_t offset = 2;
  // 32bit for CurrChunkIndex
  bs_.curr_chunk_index = chunk_count_ - offset;
  i += sizeof(uint32_t);
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 32bit * frequency_count for frequency
  auto frequency = reinterpret_cast<uint32_t *>(&data8[i]);
  i += frequency_count * sizeof(uint32_t);
  // Used for 8-byte(64bit) alignment
  i = ((i + kAlignOffset) >> kTableExtend) << kTableExtend;
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 32bit * frequency_count for centroids
  centroids_ = reinterpret_cast<void *>(&data8[i]);
  centroids_size_ = frequency_count * sizeof(float);
  i += centroids_size_;
  // Used for 8-byte(64bit) alignment
  i = ((i + kAlignOffset) >> kTableExtend) << kTableExtend;
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 64bit * bs_.GetCurrChunkIndex() + 1 for Chunks.
  chunks_ = reinterpret_cast<uint64_t *>(&data8[i]);
  chunk_size_ = (bs_.curr_chunk_index + 1) * sizeof(uint64_t);
  i += chunk_size_;
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 64bit for CurrChunk
  bs_.curr_chunk = *(reinterpret_cast<uint64_t *>(&data8[i]));
  i += sizeof(uint64_t);
  if (i > data_size) {
    MS_LOG(ERROR) << "index over total size"
                  << " index:" << i << " total size:" << data_size;
    return RET_ERROR;
  }
  // 8bit for CurrBitCount
  bs_.curr_bit_count = *(reinterpret_cast<uint8_t *>(&data8[i]));

  table_size_ = 1u << table_log_;
  states_table_ = static_cast<uint16_t *>(malloc(table_size_ * sizeof(uint16_t)));
  CHECK_NULL_RETURN(states_table_);
  bit_count_table_ = static_cast<uint8_t *>(malloc(table_size_ * sizeof(uint8_t)));
  CHECK_NULL_RETURN(bit_count_table_);
  symbol_table_ = static_cast<uint16_t *>(malloc(table_size_ * sizeof(uint16_t)));
  CHECK_NULL_RETURN(symbol_table_);

  auto ret = FSEDecoder::FSECreateStatesForDecoding(frequency, frequency_count, table_log_, states_table_,
                                                    bit_count_table_, symbol_table_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FSE create states for decoding failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int QuantDTypeCastTensorRT::AddQuantPlugin(TensorRTContext *ctx, mindspore::lite::Tensor *lite_tensor) {
  CHECK_NULL_RETURN(ctx);
  CHECK_NULL_RETURN(lite_tensor);

  std::shared_ptr<QuantDTypeCastPlugin> plugin;
  for (auto &quant : lite_tensor->quant_params()) {
    scales_.push_back(quant.scale);
    zps_.push_back(quant.zeroPoint);
  }
  if (lite_tensor->quant_params().size() == 1) {
    plugin = std::make_shared<QuantDTypeCastPlugin>(op_name_, quant_dtype_cast_->axis(), scales_.data(), zps_.data(),
                                                    lite_tensor->quant_params().size(), lite_tensor->shape());
  } else {  // PerChannel
    cudaMalloc(&scale_device_ptr_, scales_.size() * sizeof(float));
    cudaMemcpy(scale_device_ptr_, scales_.data(), scales_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&zp_device_ptr_, zps_.size() * sizeof(int));
    cudaMemcpy(zp_device_ptr_, zps_.data(), zps_.size() * sizeof(int), cudaMemcpyHostToDevice);
    plugin =
      std::make_shared<QuantDTypeCastPlugin>(op_name_, quant_dtype_cast_->axis(), scale_device_ptr_, zp_device_ptr_,
                                             lite_tensor->quant_params().size(), lite_tensor->shape());
  }

  CHECK_NULL_RETURN(plugin);
  nvinfer1::Dims dims{};
  dims.nbDims = 1;
  dims.d[0] = 1;
  for (size_t i = 0; i < lite_tensor->shape().size(); i++) {
    dims.d[0] *= lite_tensor->shape().at(i);
  }

  dims.d[0] = UP_DIV(dims.d[0], sizeof(int32_t));
  int32_align_ = malloc(dims.d[0] * sizeof(int32_t));
  CHECK_MALLOC_RES(int32_align_, RET_ERROR);
  CHECK_NULL_RETURN(lite_tensor->data());
  memcpy(int32_align_, lite_tensor->data(), lite_tensor->Size());

  nvinfer1::IConstantLayer *constant_tensor;

  nvinfer1::Weights weights{nvinfer1::DataType::kINT32, int32_align_, dims.d[0]};
  constant_tensor = ctx->network()->addConstant(dims, weights);
  ctx->RegisterLayer(constant_tensor, lite_tensor->tensor_name());
  auto tensor_ptr = constant_tensor->getOutput(0);
  nvinfer1::ITensor *input_tensors[] = {tensor_ptr};
  auto quant_layer = ctx->network()->addPluginV2(input_tensors, 1, *plugin);
  if (quant_layer == nullptr) {
    MS_LOG(ERROR) << op_name_ << " create cast layer failed for: " << op_name_;
    return RET_ERROR;
  }

  quant_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *quant_out = quant_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{quant_out, Format::NHWC, true, true}, out_tensors_[0].Name());
  this->layer_ = quant_layer;
  return RET_OK;
}

int QuantDTypeCastTensorRT::AddFSEPlugin(TensorRTContext *ctx, mindspore::lite::Tensor *lite_tensor) {
  CHECK_NULL_RETURN(ctx);
  CHECK_NULL_RETURN(lite_tensor);

  nvinfer1::Dims dims{};
  dims.nbDims = 1;

  dims.d[0] = chunk_size_ / sizeof(int32_t);

  nvinfer1::IConstantLayer *constant_tensor;

  nvinfer1::Weights weights{nvinfer1::DataType::kINT32, chunks_, dims.d[0]};
  constant_tensor = ctx->network()->addConstant(dims, weights);
  ctx->RegisterLayer(constant_tensor, lite_tensor->tensor_name());
  auto tensor_ptr = constant_tensor->getOutput(0);
  nvinfer1::ITensor *input_tensors[] = {tensor_ptr};

  // Copy buff to device
  cudaMalloc(&states_table_device_, table_size_ * sizeof(uint16_t));
  cudaMemcpy(states_table_device_, states_table_, table_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice);

  cudaMalloc(&bit_count_table_device_, table_size_ * sizeof(uint8_t));
  cudaMemcpy(bit_count_table_device_, bit_count_table_, table_size_ * sizeof(uint8_t), cudaMemcpyHostToDevice);

  cudaMalloc(&symbol_table_device_, table_size_ * sizeof(uint16_t));
  cudaMemcpy(symbol_table_device_, symbol_table_, table_size_ * sizeof(uint16_t), cudaMemcpyHostToDevice);

  cudaMalloc(&centroids_device_, centroids_size_);
  cudaMemcpy(centroids_device_, centroids_, centroids_size_, cudaMemcpyHostToDevice);

  auto plugin =
    std::make_shared<FSEPlugin>(op_name_, lite_tensor->shape(), bs_, states_table_device_, bit_count_table_device_,
                                symbol_table_device_, table_size_, table_log_, centroids_device_, centroids_size_);
  CHECK_NULL_RETURN(plugin);
  auto quant_layer = ctx->network()->addPluginV2(input_tensors, 1, *plugin);
  if (quant_layer == nullptr) {
    MS_LOG(ERROR) << op_name_ << " create cast layer failed for: " << op_name_;
    return RET_ERROR;
  }

  quant_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *quant_out = quant_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{quant_out, Format::NHWC, true, true}, out_tensors_[0].Name());
  this->layer_ = quant_layer;
  return RET_OK;
}

int QuantDTypeCastTensorRT::AddInnerOp(TensorRTContext *ctx) {
  // perlayer dequant
  auto tensor = in_tensors_[0];
  CHECK_NULL_RETURN(tensor.impl());
  auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl());
  CHECK_NULL_RETURN(lite_impl);
  auto lite_tensor = static_cast<mindspore::lite::Tensor *>(lite_impl->lite_tensor());
  CHECK_NULL_RETURN(lite_tensor);

  int ret;
  if (lite_tensor->get_compress_type() == kFSEInfer) {
    ret = Deserialize(static_cast<int8_t *>(const_cast<void *>(tensor.Data().get())), tensor.DataSize());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name_ << " Deserialize failed";
      return ret;
    }
    ret = AddFSEPlugin(ctx, lite_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name_ << " AddFSEPlugin failed";
      return ret;
    }
  } else {
    ret = AddQuantPlugin(ctx, lite_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << op_name_ << " AddQuantPlugin failed";
      return ret;
    }
  }
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_QuantDTypeCast, QuantDTypeCastTensorRT)
}  // namespace mindspore::lite
