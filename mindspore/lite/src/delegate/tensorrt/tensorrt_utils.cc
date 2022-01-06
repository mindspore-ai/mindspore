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

#include "src/delegate/tensorrt/tensorrt_utils.h"
#include <cuda_runtime_api.h>
#include <map>
#include <numeric>
#include <functional>

namespace mindspore::lite {
nvinfer1::Dims ConvertCudaDims(int data, size_t size) {
  nvinfer1::Dims dims{};
  dims.nbDims = -1;
  if (size > static_cast<size_t>(dims.MAX_DIMS)) {
    MS_LOG(ERROR) << "invalid shape size: " << size;
    return dims;
  }
  dims.nbDims = size;
  for (size_t i = 0; i < size; i++) {
    dims.d[i] = data;
  }
  return dims;
}

nvinfer1::Dims ConvertCudaDims(const void *data, int64_t size) {
  nvinfer1::Dims dims{};
  dims.nbDims = -1;
  if (size > static_cast<int64_t>(dims.MAX_DIMS)) {
    MS_LOG(ERROR) << "invalid shape size: " << size;
    return dims;
  }
  dims.nbDims = size;
  const int *dims_data = reinterpret_cast<const int *>(data);
  for (int i = 0; i < size; i++) {
    dims.d[i] = *(dims_data + i);
  }
  return dims;
}

bool SameDims(nvinfer1::Dims dims, const std::vector<int64_t> &shape) {
  if (dims.nbDims != static_cast<int>(shape.size())) {
    return false;
  }
  // dynamic dim, only channel dim know
  for (int i = 0; i < dims.nbDims; i++) {
    if (dims.d[i] == -1) {
      continue;
    }
    if (dims.d[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> ConvertMSShape(const nvinfer1::Dims dims) {
  std::vector<int64_t> shape;
  for (int i = 0; i < dims.nbDims; i++) {
    shape.push_back(dims.d[i]);
  }
  return shape;
}

std::vector<int64_t> NHWC2NCHW(std::vector<int64_t> nhwc_shape) {
  std::vector<int64_t> nchw_shape;
  if (nhwc_shape.size() != DIMENSION_4D) {
    return nhwc_shape;
  }
  nchw_shape.push_back(nhwc_shape[kNHWC_N]);
  nchw_shape.push_back(nhwc_shape[kNHWC_C]);
  nchw_shape.push_back(nhwc_shape[kNHWC_H]);
  nchw_shape.push_back(nhwc_shape[kNHWC_W]);
  return nchw_shape;
}

nvinfer1::IShuffleLayer *SetTranspose(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &input,
                                      nvinfer1::Permutation permutation) {
  nvinfer1::IShuffleLayer *layer = network->addShuffle(const_cast<nvinfer1::ITensor &>(input));
  if (layer == nullptr) {
    MS_LOG(ERROR) << "failed to create ShuffleLayer when create transpose op.";
    return nullptr;
  }
  layer->setFirstTranspose(permutation);
  return layer;
}

nvinfer1::DataType ConvertDataType(DataType type_id) {
  std::map<DataType, nvinfer1::DataType> data_type_map = {
    {DataType::kNumberTypeInt8, nvinfer1::DataType::kINT8},
    {DataType::kNumberTypeInt32, nvinfer1::DataType::kINT32},
    {DataType::kNumberTypeFloat32, nvinfer1::DataType::kFLOAT},
    {DataType::kNumberTypeFloat16, nvinfer1::DataType::kHALF},
  };
  auto iter = data_type_map.find(type_id);
  nvinfer1::DataType data_type;
  if (iter != data_type_map.end()) {
    data_type = iter->second;
  } else {
    data_type = nvinfer1::DataType::kFLOAT;
    MS_LOG(WARNING) << "invalid data_type for TensorRT, need check: " << static_cast<int>(type_id);
  }
  return data_type;
}

nvinfer1::IShuffleLayer *NHWC2NCHW(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &input) {
  // NHWC 0123 NCHW 0312
  nvinfer1::Permutation perm{{0, 3, 1, 2}};
  return SetTranspose(network, input, perm);
}

nvinfer1::IShuffleLayer *NCHW2NHWC(nvinfer1::INetworkDefinition *network, const nvinfer1::ITensor &input) {
  // NCHW 0123 NHWC 0231
  nvinfer1::Permutation perm{{0, 2, 3, 1}};
  return SetTranspose(network, input, perm);
}

nvinfer1::ITensor *ConvertConstantTensor(nvinfer1::INetworkDefinition *network, const mindspore::MSTensor &ms_tensor,
                                         const std::string &op_name) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is null for ConvertConstantTensor";
    return nullptr;
  }
  nvinfer1::Dims dims = ConvertCudaDims(ms_tensor.Shape());
  if (dims.nbDims == -1) {
    MS_LOG(WARNING) << ms_tensor.Name() << " ConvertCudaDims failed, convert as scalar.";
    dims.nbDims = 1;
    dims.d[0] = 1;
  }
  nvinfer1::DataType data_type = ConvertDataType(ms_tensor.DataType());
  if (ms_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "ConvertConstantTensor from a MSTensor with nullptr data: " << ms_tensor.Name();
    return nullptr;
  }
  nvinfer1::Weights weights{data_type, ms_tensor.Data().get(), ms_tensor.ElementNum()};
  nvinfer1::IConstantLayer *constant_tensor = network->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  auto name = ms_tensor.Name() + "_" + op_name;
  constant_tensor->setName(name.c_str());
  return constant_tensor->getOutput(0);
}

nvinfer1::ITensor *ConvertScalarToITensor(nvinfer1::INetworkDefinition *network, size_t shape_size, const void *value,
                                          const DataType data_type, const std::string &op_name) {
  nvinfer1::Dims dims = ConvertCudaDims(1, shape_size);
  if (dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name;
    return nullptr;
  }
  nvinfer1::Weights weights{ConvertDataType(data_type), value, 1};
  nvinfer1::IConstantLayer *constant_tensor = network->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  auto name = op_name + "_constant";
  constant_tensor->setName(name.c_str());
  return constant_tensor->getOutput(0);
}

ActivationParams ConvertActivationType(schema::ActivationType activation_type) {
  std::map<schema::ActivationType, ActivationParams> action_map = {
    {schema::ActivationType_RELU, ActivationParams{nvinfer1::ActivationType::kRELU, false, 0, false, 0}},
    {schema::ActivationType_SIGMOID, ActivationParams{nvinfer1::ActivationType::kSIGMOID, false, 0, false, 0}},
    {schema::ActivationType_TANH, ActivationParams{nvinfer1::ActivationType::kTANH, false, 0, false, 0}},
    {schema::ActivationType_LEAKY_RELU, ActivationParams{nvinfer1::ActivationType::kLEAKY_RELU, true, 0, false, 0}},
    {schema::ActivationType_ELU, ActivationParams{nvinfer1::ActivationType::kELU, true, 0, false, 0}},
    {schema::ActivationType_SELU, ActivationParams{nvinfer1::ActivationType::kSELU, true, 0, true, 0}},
    {schema::ActivationType_SOFTSIGN, ActivationParams{nvinfer1::ActivationType::kSOFTSIGN, false, 0, false, 0}},
    {schema::ActivationType_SOFTPLUS, ActivationParams{nvinfer1::ActivationType::kSOFTPLUS, true, 0, true, 0}},
    {schema::ActivationType_THRESHOLDRELU,
     ActivationParams{nvinfer1::ActivationType::kTHRESHOLDED_RELU, true, 0, false, 0}},
    {schema::ActivationType_RELU6, ActivationParams{nvinfer1::ActivationType::kCLIP, true, 0, true, 6}},
    {schema::ActivationType_RELU1, ActivationParams{nvinfer1::ActivationType::kCLIP, true, 0, true, 1}},
    {schema::ActivationType_HARD_TANH, ActivationParams{nvinfer1::ActivationType::kCLIP, true, -1, true, 1}}};
  auto iter = action_map.find(activation_type);
  ActivationParams action_param = ActivationParams{nvinfer1::ActivationType::kRELU, false, 0, false, 0};
  if (iter != action_map.end()) {
    action_param = iter->second;
  } else {
    MS_LOG(WARNING) << "Unsupported op action type for TensorRT: " << activation_type;
  }
  return action_param;
}

nvinfer1::ITensor *ConvertTensorWithExpandDims(nvinfer1::INetworkDefinition *network,
                                               const mindspore::MSTensor &ms_tensor, size_t expand_shape_size,
                                               const std::string &op_name) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is null for ConvertConstantTensor";
    return nullptr;
  }
  std::vector<int64_t> shape(expand_shape_size);
  size_t shape_size = ms_tensor.Shape().size();
  size_t expand_size = expand_shape_size - shape_size;
  for (size_t i = 0; i < expand_shape_size; ++i) {
    if (i < expand_size) {
      shape[i] = 1;
    } else {
      shape[i] = ms_tensor.Shape()[i - expand_size];
    }
  }
  nvinfer1::Dims dims = ConvertCudaDims(shape);
  if (dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name;
    return nullptr;
  }
  nvinfer1::DataType data_type = ConvertDataType(ms_tensor.DataType());
  if (ms_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "ConvertTensorWithExpandDims from a MSTensor with nullptr data";
    return nullptr;
  }
  nvinfer1::Weights weights{data_type, ms_tensor.Data().get(), ms_tensor.ElementNum()};
  nvinfer1::IConstantLayer *constant_tensor = network->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  auto name = ms_tensor.Name() + "_" + op_name;
  constant_tensor->setName(name.c_str());
  return constant_tensor->getOutput(0);
}

nvinfer1::Weights TransposeWeight(const mindspore::MSTensor &ms_tensor, void **pack_weight) {
  nvinfer1::Weights weights{};
  MS_LOG(DEBUG) << "ms_tensor.DataType(): " << static_cast<int>(ms_tensor.DataType());
  if (ms_tensor.DataType() == DataType::kNumberTypeFloat16) {
    weights.type = nvinfer1::DataType::kHALF;
    weights.count = ms_tensor.ElementNum();
    void *pack_weight_tmp = malloc(ms_tensor.DataSize());
    if (pack_weight_tmp == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return weights;
    }
    MS_ASSERT(ms_tensor.Data());
    auto weight_shape = ms_tensor.Shape();
    PackNHWCToNCHWFp16(ms_tensor.Data().get(), pack_weight_tmp, weight_shape[0], weight_shape[1] * weight_shape[2],
                       weight_shape[3], 0, 0);
    *pack_weight = pack_weight_tmp;
    weights.values = pack_weight_tmp;
    return weights;
  } else {
    return TransposeWeightFP32(ms_tensor, pack_weight);
  }
}

nvinfer1::Weights TransposeWeightFP32(const mindspore::MSTensor &ms_tensor, void **pack_weight) {
  // usage notice: malloc addr saved to pack_weight, save pack_weight ptr and free it when deconstruct
  nvinfer1::Weights weights{};
  weights.count = ms_tensor.ElementNum();
  if (lite::ConvertDataType(ms_tensor.DataType()) != nvinfer1::DataType::kFLOAT) {
    MS_LOG(WARNING) << "weights data type is not float32";
  }
  weights.type = nvinfer1::DataType::kFLOAT;
  auto weight_shape = ms_tensor.Shape();
  const void *src_ptr = ms_tensor.Data().get();
  if (src_ptr == nullptr) {
    MS_LOG(ERROR) << "TransposeWeight from a MSTensor with nullptr data";
    return weights;
  }

  float *pack_weight_tmp = reinterpret_cast<float *>(malloc(ms_tensor.ElementNum() * sizeof(float)));
  if (pack_weight_tmp == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return weights;
  }
  PackNHWCToNCHWFp32(src_ptr, pack_weight_tmp, weight_shape[0], weight_shape[1] * weight_shape[2], weight_shape[3], 0,
                     0);
  weights.values = pack_weight_tmp;
  *pack_weight = pack_weight_tmp;
  return weights;
}

nvinfer1::Weights ConvertWeight(const mindspore::MSTensor &ms_tensor) {
  nvinfer1::Weights weights{};
  weights.type = ConvertDataType(ms_tensor.DataType());
  weights.values = ms_tensor.Data().get();
  weights.count = ms_tensor.ElementNum();
  if (weights.values == nullptr) {
    MS_LOG(ERROR) << "ConvertWeight from a MSTensor with nullptr data";
  }
  return weights;
}

int SetCudaDevice(std::shared_ptr<GPUDeviceInfo> device_info_) {
  return SetCudaDevice(static_cast<int>(device_info_->GetDeviceID()));
}

int SetCudaDevice(int device_id) {
  int device = 0;
  auto ret = cudaGetDevice(&device);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaGetDevice failed, device is untrustable. error code: " << ret;
    return RET_ERROR;
  }
  int set_device_id = device_id;
  int deviceCnt = 0;

  ret = cudaGetDeviceCount(&deviceCnt);
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "cudaGetDeviceCount failed.";
    return RET_ERROR;
  }

  if (set_device_id > deviceCnt - 1) {
    MS_LOG(ERROR) << "invalid input device id as " << set_device_id << " for current device count " << deviceCnt;
    return RET_ERROR;
  }
  if (device != set_device_id) {
    ret = cudaSetDevice(set_device_id);
    if (ret != cudaSuccess) {
      MS_LOG(ERROR) << "cudaSetDevice failed, error code: " << ret;
      return RET_ERROR;
    }
  }
  if (cudaGetDevice(&device) != cudaSuccess) {
    MS_LOG(ERROR) << "cudaGetDevice failed, device is untrustable.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "cuda is running on device: " << device;
  return RET_OK;
}

Format GetOutputFormat(Format input_format, nvinfer1::Permutation perm) {
  if (input_format == Format::NHWC) {
    if (perm.order[0] == 0 && perm.order[1] == 3 && perm.order[2] == 2 && perm.order[3] == 1) {
      return Format::NCHW;
    }
  } else if (input_format == Format::NCHW) {
    if (perm.order[0] == 0 && perm.order[1] == 2 && perm.order[2] == 3 && perm.order[3] == 1) {
      return Format::NHWC;
    }
  }
  MS_LOG(WARNING) << "transpose out format needs to check for " << input_format;
  return input_format;
}
int ConvertAxisFromNHWC2NCHW(int nhwc_axis) {
  // N0H1W2C3->N0C1H2W3
  if (nhwc_axis > kNHWC_C) {
    return nhwc_axis;
  }
  switch (nhwc_axis) {
    case kNHWC_N:
      return kNCHW_N;
    case kNHWC_H:
      return kNCHW_H;
    case kNHWC_W:
      return kNCHW_W;
    case kNHWC_C:
      return kNCHW_C;
    default:
      MS_LOG(ERROR) << "invalid input axis for nhwc: " << nhwc_axis;
  }
  return nhwc_axis;
}

void PackNHWCToNCHWFp16(const void *src, void *dst, size_t batches, size_t plane, size_t channel, size_t task_id,
                        size_t thread_count) {
  size_t hw8 = plane / C8NUM;
  size_t task_start = 0;
  size_t task_end = plane;
  if (thread_count > 0) {
    size_t offset_hw = UP_DIV(hw8, thread_count) * C8NUM;
    task_start = offset_hw * task_id;
    size_t count = plane - task_start;
    if (count == 0) {
      return;
    }
    task_end = (task_id + 1) == thread_count ? plane : MSMIN(plane, task_start + offset_hw);
    hw8 = task_start + ((task_end - task_start) >= offset_hw ? offset_hw : 0);
  } else {
    hw8 *= C8NUM;
  }
  size_t c8 = channel / C8NUM * C8NUM;
  size_t batch = plane * channel;
  for (size_t n = 0; n < batches; n++) {
    const uint16_t *src_batch = static_cast<const uint16_t *>(src) + n * batch;
    uint16_t *dst_batch = static_cast<uint16_t *>(dst) + n * batch;
    size_t hw = task_start;
    for (; hw < hw8; hw += C8NUM) {
      size_t c = 0;
      for (; c < c8; c += C8NUM) {
        const uint16_t *src_ptr = src_batch + hw * channel + c;
        uint16_t *dst_ptr = dst_batch + c * plane + hw;
        for (size_t tr = 0; tr < C8NUM; tr++) {
          for (size_t tc = 0; tc < C8NUM; tc++) {
            dst_ptr[tc * plane + tr] = src_ptr[tr * channel + tc];
          }
        }
      }
      for (; c < channel; c++) {
        const uint16_t *src_ptr = src_batch + hw * channel + c;
        uint16_t *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < task_end; hw++) {
      const uint16_t *src_ptr = src_batch + hw * channel;
      uint16_t *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
}
std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor, mindspore::Format format, bool is_same) {
  nvinfer1::Dims dims = trt_tensor->getDimensions();
  std::string is_same_string = is_same ? " is same with ms tensor " : " is different from ms tensor ";
  std::string out_string = "tensor " + std::string(trt_tensor->getName()) + ": format (NHWC:1, NCHW:0) is " +
                           std::to_string(static_cast<int>(format)) + is_same_string + ", dims is ";
  std::string dim_string = "[";
  for (int i = 0; i < dims.nbDims; i++) {
    dim_string += std::to_string(dims.d[i]);
    if (i != dims.nbDims - 1) {
      dim_string += ", ";
    }
  }
  dim_string += "]";
  out_string += dim_string;
  return out_string;
}

std::string GetTensorFormat(ITensorHelper tensor_helper) {
  return GetTensorFormat(tensor_helper.trt_tensor_, tensor_helper.format_, tensor_helper.same_format_);
}

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor) { return GetTensorFormat(trt_tensor, Format::NHWC, true); }

nvinfer1::ReduceOperation ConvertTRTReduceMode(schema::ReduceMode mode) {
  std::map<schema::ReduceMode, nvinfer1::ReduceOperation> reduce_ops_ = {
    {schema::ReduceMode::ReduceMode_ReduceMean, nvinfer1::ReduceOperation::kAVG},
    {schema::ReduceMode::ReduceMode_ReduceMax, nvinfer1::ReduceOperation::kMAX},
    {schema::ReduceMode::ReduceMode_ReduceMin, nvinfer1::ReduceOperation::kMIN},
    {schema::ReduceMode::ReduceMode_ReduceProd, nvinfer1::ReduceOperation::kPROD},
    {schema::ReduceMode::ReduceMode_ReduceSum, nvinfer1::ReduceOperation::kSUM},
  };
  auto iter = reduce_ops_.find(mode);
  nvinfer1::ReduceOperation trt_mode;
  if (iter != reduce_ops_.end()) {
    trt_mode = iter->second;
  } else {
    trt_mode = nvinfer1::ReduceOperation::kAVG;
    MS_LOG(WARNING) << "invalid reduce for TensorRT, need check: " << static_cast<int>(mode);
  }
  return trt_mode;
}
int PreprocessInputs2SameDim(nvinfer1::INetworkDefinition *network, const ITensorHelper &input_tensor_helper,
                             ITensorHelper *out_tensor_helper) {
  out_tensor_helper->trt_tensor_ = input_tensor_helper.trt_tensor_;
  out_tensor_helper->format_ = input_tensor_helper.format_;
  out_tensor_helper->same_format_ = true;
  if (input_tensor_helper.trt_tensor_->getDimensions().nbDims == DIMENSION_4D && !input_tensor_helper.same_format_) {
    if (input_tensor_helper.format_ == Format::NCHW) {
      // transpose: NCHW->NHWC
      nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(network, *input_tensor_helper.trt_tensor_);
      if (transpose_layer_in == nullptr) {
        MS_LOG(ERROR) << "op action convert failed";
        return RET_ERROR;
      }
      transpose_layer_in->setName(
        (std::string(input_tensor_helper.trt_tensor_->getName()) + "_input_transpose2NHWC").c_str());
      out_tensor_helper->trt_tensor_ = transpose_layer_in->getOutput(0);
      out_tensor_helper->format_ = Format::NHWC;
    } else {
      // transpose: NHWC->NCHW
      nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *input_tensor_helper.trt_tensor_);
      if (transpose_layer_in == nullptr) {
        MS_LOG(ERROR) << "op action convert failed";
        return RET_ERROR;
      }
      transpose_layer_in->setName(
        (std::string(input_tensor_helper.trt_tensor_->getName()) + "_input_transpose2NCHW").c_str());
      out_tensor_helper->trt_tensor_ = transpose_layer_in->getOutput(0);
      out_tensor_helper->format_ = Format::NCHW;
    }
  }
  return RET_OK;
}

int GetDimsVolume(const nvinfer1::Dims &dims) {
  if (dims.nbDims == 0) {
    return 0;
  }
  return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
}

int GetDimsVolume(const std::vector<int64_t> &shape) {
  if (shape.size() == 0) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

void SerializeValue(void **buffer, const void *value, size_t cpy_size) {
  std::memcpy(*buffer, value, cpy_size);
  *buffer = static_cast<char *>(*buffer) + cpy_size;
}

void DeserializeValue(void const **buffer, size_t *buffer_size, void *value, size_t cpy_size) {
  if (cpy_size > *buffer_size) {
    MS_LOG(ERROR) << "invalid desirialize size, buffer size: " << *buffer_size << ", value size: " << cpy_size;
    return;
  }
  std::memcpy(value, *buffer, cpy_size);
  *buffer = static_cast<const char *>(*buffer) + cpy_size;
  *buffer_size -= cpy_size;
}
}  // namespace mindspore::lite
