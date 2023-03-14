/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/fse_decoder_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/fse_decode.cuh"
#include "ops/fse_decode.h"
#include "tools/converter/quantizer/fse_chunk_end.h"

namespace mindspore::lite {
namespace {
constexpr std::size_t kTwo = 2;
constexpr std::size_t kThree = 3;
constexpr std::size_t kFour = 4;
constexpr std::size_t kSix = 6;
constexpr std::size_t kInputSize = 7;
}  // namespace

template <typename T>
bool isValidData(const T *data, size_t data_size, int value) {
  for (size_t i = 0; i < data_size; i++) {
    if (data[i] >= value) {
      return false;
    }
  }
  return true;
}

bool FseDecoderTensorRT::IsChunkEndDataValid() {
  auto bit_count_buff = reinterpret_cast<const uint8_t *>(in_tensors_[kTwo].Data());
  auto ptable_buff = reinterpret_cast<const uint8_t *>(in_tensors_[kSix].Data());
  auto ptable_size = in_tensors_[kSix].ElementNum();
  auto chunk_size = in_tensors_[0].ElementNum();
  auto state_size = in_tensors_[1].ElementNum();

  for (size_t i = 0; i < static_cast<size_t>(ptable_size); i++) {
    mindspore::lite::quant::ChunkEndData ptable_data(ptable_buff[i]);
    if (ptable_data.state >= state_size) {
      MS_LOG(ERROR) << "ERROR: ptable[" << i << "].state: " << ptable_data.state;
      return false;
    }
    if (ptable_data.bit_count != static_cast<uint16_t>(bit_count_buff[ptable_data.state])) {
      MS_LOG(ERROR) << "ERROR: ptable[" << i << "].bit_count: " << ptable_data.bit_count << ", bit_count_buff["
                    << ptable_data.state << "]: " << static_cast<uint16_t>(bit_count_buff[ptable_data.state]);
      return false;
    }
    uint64_t chunk_index = ptable_data.bs_position / (CHAR_BIT * sizeof(uint64_t));
    if (chunk_index >= (uint64_t)chunk_size) {
      MS_LOG(ERROR) << "ERROR: ptable[" << i << "].bs_position: " << ptable_data.bs_position
                    << "chunk_size:" << chunk_size << "ptable_size:" << ptable_size;
      return false;
    }
  }
  return true;
}

// FSEDecode TensorRT op
int FseDecoderTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                  const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != kInputSize) {
    MS_LOG(ERROR) << "Unsupported number of inputs, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int FseDecoderTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  auto fse_decoder_op = AsOps<ops::FSEDecode>();
  if (fse_decoder_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }

  // Check tables validity
  MS_ASSERT(in_tensors_[1].ElementNum() == in_tensors_[kThree].ElementNum());
  MS_ASSERT(in_tensors_[kTwo].ElementNum() == in_tensors_[kThree].ElementNum());

  if (!isValidData(reinterpret_cast<const uint16_t *>(in_tensors_[1].Data()), in_tensors_[1].ElementNum(),
                   in_tensors_[1].ElementNum()) ||
      !isValidData(reinterpret_cast<const uint8_t *>(in_tensors_[kTwo].Data()), in_tensors_[kTwo].ElementNum(), 64) ||
      !isValidData(reinterpret_cast<const uint16_t *>(in_tensors_[kThree].Data()), in_tensors_[kThree].ElementNum(),
                   in_tensors_[kFour].ElementNum())) {
    MS_LOG(ERROR) << "Invalid data in tables";
    return RET_ERROR;
  }

  MS_ASSERT(IsChunkEndDataValid());

  uint64_t curr_chunk_idx = static_cast<uint64_t>(fse_decoder_op->get_curr_chunk_index());
  int64_t dst_type = fse_decoder_op->get_dst_t();
  uint64_t curr_bit_count = static_cast<uint64_t>(fse_decoder_op->get_curr_bit_count());
  uint64_t table_log = static_cast<uint64_t>(fse_decoder_op->get_table_log());
  uint64_t curr_chunk = static_cast<uint64_t>(fse_decoder_op->get_curr_chunk());
  const int input_number = inputs().size();
  auto output = outputs().at(0);
  auto output_shape = output.Shape();

  // Convert tensors to int32 for TensorRT
  nvinfer1::Dims dims{};
  dims.nbDims = 1;
  size_t start_const = C0NUM;
  size_t end_const = C4NUM;
  for (size_t i = 0; i < in_tensors_.size(); i++) {
    auto in_tensor = input(ctx, i);
    if ((i >= start_const && i < end_const) || (i == kSix)) {
      auto size = inputs().at(i).DataSize();
      dims.d[0] = size / sizeof(int32_t);
      nvinfer1::IConstantLayer *constant_tensor;
      nvinfer1::Weights weights{nvinfer1::DataType::kINT32, inputs().at(i).Data(), dims.d[0]};
      constant_tensor = ctx->network()->addConstant(dims, weights);
      ctx->RegisterLayer(constant_tensor, inputs().at(i).Name() + "_" + op_name_);
      in_tensor.trt_tensor_ = constant_tensor->getOutput(0);
      ctx->RegisterTensor(in_tensor, inputs().at(i).Name());
    } else {
      in_tensor.trt_tensor_ = lite::ConvertConstantTensor(ctx, inputs().at(i), op_name_);
      ctx->RegisterTensor(in_tensor, inputs().at(i).Name());
    }
  }
  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto plugin = std::make_shared<FseDecoderPlugin>(input_tensor->getName(), curr_chunk_idx, dst_type, curr_bit_count,
                                                   table_log, curr_chunk, output_shape, device_id_);
  nvinfer1::ITensor *inputTensors[input_number];
  for (int i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }
  nvinfer1::IPluginV2Layer *fse_decoder_layer = ctx->network()->addPluginV2(inputTensors, input_number, *plugin);
  if (fse_decoder_layer == nullptr) {
    MS_LOG(ERROR) << "add fse decoder op failed for TensorRT.";
    return RET_ERROR;
  }
  fse_decoder_layer->setName((op_name_ + "plugin_fse_decoder").c_str());
  nvinfer1::ITensor *fse_decoder_tensor = fse_decoder_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{fse_decoder_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = fse_decoder_layer;
  return RET_OK;
}

//  PLUGIN of FSE Decode Layer
REGISTER_TENSORRT_PLUGIN(FseDecoderPluginCreater);
template class TensorRTPluginCreater<FseDecoderPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

int FseDecoderPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                              const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) noexcept {
  if (dst_type_ == mindspore::kNumberTypeFloat16) {
    return RunFseDecoder<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
  } else {
    return RunFseDecoder<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
  }
}

template <typename T>
std::unique_ptr<T[]> getTensor(T *tensor, int size) {
  using non_const_T = std::remove_const_t<T>;
  auto buff = std::make_unique<non_const_T[]>(size);
  cudaMemcpy(reinterpret_cast<void *>(buff.get()), tensor, size * sizeof(T), cudaMemcpyDeviceToHost);
  return buff;
}

template <typename T>
int FseDecoderPlugin::RunFseDecoder(const nvinfer1::PluginTensorDesc *inputDesc,
                                    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                    void *const *outputs, void *workspace, cudaStream_t stream) {
  auto chunks = reinterpret_cast<const uint64_t *>(inputs[0]);
  auto states_table = reinterpret_cast<const uint16_t *>(inputs[1]);
  auto bit_count_table = reinterpret_cast<const uint8_t *>(inputs[kTwo]);
  auto symbol_table = reinterpret_cast<const uint16_t *>(inputs[kThree]);
  auto centroids = reinterpret_cast<const T *>(inputs[kFour]);
  auto ptable = reinterpret_cast<const uint64_t *>(inputs[kSix]);
  auto out = reinterpret_cast<T *>(outputs[0]);
  nvinfer1::Dims output_dims = outputDesc[0].dims;
  nvinfer1::Dims input_dims = inputDesc[kSix].dims;
  auto out_size = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1, std::multiplies<int64_t>());
  int ptable_size = (std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>()));
  ptable_size = ptable_size * sizeof(int32_t) /
                sizeof(uint64_t);  // transform to original size due to previous conversion (from int32 to uint64)
  bool use_curr_chunk = (curr_bit_count_ > table_log_);
  FSE_Decode<T>(chunks, states_table, bit_count_table, symbol_table, ptable, ptable_size, centroids, out_size, out,
                device_id_, curr_chunk_, use_curr_chunk, stream);
  return 0;
}

bool FseDecoderPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                                 int nbOutputs) noexcept {
  bool format = (tensorsDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  auto type = tensorsDesc[pos].type;
  if (pos == nbInputs) {
    format &= (type == nvinfer1::DataType::kFLOAT);
  } else {
    switch (pos) {
      case C0NUM:
      case C1NUM:
      case C2NUM:
      case C3NUM:
      case C5NUM:
        format &= (type == nvinfer1::DataType::kINT32);
        break;
      case C4NUM:
        format &= (type == nvinfer1::DataType::kFLOAT);
        break;
      default:
        break;
    }
  }
  return format;
}

void FseDecoderPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {}

size_t FseDecoderPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                          const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept {
  return 0;
}

nvinfer1::DataType FseDecoderPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                       int nbInputs) const noexcept {
  return nvinfer1::DataType::kFLOAT;
}

nvinfer1::DimsExprs FseDecoderPlugin::getOutputDimensions(int32_t index, const nvinfer1::DimsExprs *inputs,
                                                          int nbInputDims,
                                                          nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = output_shape_.size();
  for (int i = 0; i < out_dims.nbDims; i++) {
    out_dims.d[i] = exprBuilder.constant(output_shape_[i]);
  }
  return out_dims;
}

nvinfer1::IPluginV2DynamicExt *FseDecoderPlugin::clone() const noexcept {
  auto *plugin = new FseDecoderPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

int FseDecoderPlugin::initialize() noexcept { return 0; }

void FseDecoderPlugin::terminate() noexcept {}

size_t FseDecoderPlugin::getSerializationSize() const noexcept { return INPUT_SIZE4 * sizeof(int); }

void FseDecoderPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &curr_chunk_idx_, sizeof(int));
  SerializeValue(&buffer, &dst_type_, sizeof(int));
  SerializeValue(&buffer, &curr_bit_count_, sizeof(int));
  SerializeValue(&buffer, &table_log_, sizeof(int));
  SerializeValue(&buffer, &curr_chunk_, sizeof(int));
}

REGISTER_TENSORRT_CREATOR(ops::kNameFSEDecode, FseDecoderTensorRT)
}  // namespace mindspore::lite
