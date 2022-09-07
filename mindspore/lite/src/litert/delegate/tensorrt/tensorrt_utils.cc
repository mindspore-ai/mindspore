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

#include <cuda_runtime_api.h>
#include <map>
#include <unordered_set>
#include <numeric>
#include <functional>
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"
#include "src/litert/delegate/tensorrt/op/cast_plugin.h"
#include "src/litert/delegate/tensorrt/distribution/distribution_collective.h"

namespace mindspore::lite {
namespace {
const int INPUT2 = 2;
}
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
  const int *dims_data = static_cast<const int *>(data);
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

nvinfer1::IShuffleLayer *SetTranspose(TensorRTContext *ctx, const nvinfer1::ITensor &input,
                                      nvinfer1::Permutation permutation) {
  nvinfer1::IShuffleLayer *layer = ctx->network()->addShuffle(const_cast<nvinfer1::ITensor &>(input));
  if (layer == nullptr) {
    MS_LOG(ERROR) << "failed to create ShuffleLayer when create transpose op.";
    return nullptr;
  }
  layer->setFirstTranspose(permutation);
  return layer;
}

nvinfer1::DataType ConvertDataType(DataType type_id) {
  std::map<DataType, nvinfer1::DataType> data_type_map = {
#if TRT_VERSION_GE(7, 2)
    {DataType::kNumberTypeBool, nvinfer1::DataType::kBOOL},
#endif
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

cudaDataType ConvertDataType(nvinfer1::DataType type_id) {
  std::map<nvinfer1::DataType, cudaDataType> data_type_map = {
    {nvinfer1::DataType::kINT8, CUDA_R_8I},
    {nvinfer1::DataType::kINT32, CUDA_R_32I},
    {nvinfer1::DataType::kFLOAT, CUDA_R_32F},
    {nvinfer1::DataType::kHALF, CUDA_R_16F},
  };
  auto iter = data_type_map.find(type_id);
  cudaDataType data_type;
  if (iter != data_type_map.end()) {
    data_type = iter->second;
  } else {
    data_type = CUDA_R_32F;
    MS_LOG(WARNING) << "invalid data_type for TensorRT, need check: " << static_cast<int>(type_id);
  }
  return data_type;
}

nvinfer1::IShuffleLayer *NHWC2NCHW(TensorRTContext *ctx, const nvinfer1::ITensor &input) {
  // NHWC 0123 NCHW 0312
  nvinfer1::Permutation perm{{0, 3, 1, 2}};
  return SetTranspose(ctx, input, perm);
}

nvinfer1::IShuffleLayer *NCHW2NHWC(TensorRTContext *ctx, const nvinfer1::ITensor &input) {
  // NCHW 0123 NHWC 0231
  nvinfer1::Permutation perm{{0, 2, 3, 1}};
  return SetTranspose(ctx, input, perm);
}

nvinfer1::ITensor *ConvertConstantTensor(TensorRTContext *ctx, const mindspore::MSTensor &ms_tensor,
                                         const std::string &op_name) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is null for ConvertConstantTensor";
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
  nvinfer1::IConstantLayer *constant_tensor = ctx->network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  ctx->RegisterLayer(constant_tensor, ms_tensor.Name() + "_" + op_name);
  auto tensor_ptr = constant_tensor->getOutput(0);
  return tensor_ptr;
}

nvinfer1::ITensor *ConvertScalarToITensor(TensorRTContext *ctx, size_t shape_size, const void *value,
                                          const DataType data_type, const std::string &op_name) {
  nvinfer1::Dims dims = ConvertCudaDims(1, shape_size);
  if (dims.nbDims == -1) {
    MS_LOG(ERROR) << "ConvertCudaDims failed for " << op_name;
    return nullptr;
  }
  nvinfer1::Weights weights{ConvertDataType(data_type), value, 1};
  nvinfer1::IConstantLayer *constant_tensor = ctx->network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  ctx->RegisterLayer(constant_tensor, op_name + "_constant");
  return constant_tensor->getOutput(0);
}

nvinfer1::ITensor *ConvertScalarToITensor(TensorRTContext *ctx, size_t shape_size, const mindspore::MSTensor &ms_tensor,
                                          const DataType data_type, const std::string &op_name) {
  const void *value = ms_tensor.Data().get();
  auto tensor_ptr = ConvertScalarToITensor(ctx, shape_size, value, data_type, op_name);
  return tensor_ptr;
}

std::experimental::optional<ActivationParams> TryConvertActivationType(schema::ActivationType activation_type) {
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
    {schema::ActivationType_HARD_TANH, ActivationParams{nvinfer1::ActivationType::kCLIP, true, -1, true, 1}},
    // using plugin
    {schema::ActivationType_GELU, ActivationParams{nvinfer1::ActivationType::kTHRESHOLDED_RELU, false, 0, false, 0}},
    {schema::ActivationType_SWISH, ActivationParams{nvinfer1::ActivationType::kSIGMOID, false, 0, false, 0}}};
  return action_map.find(activation_type) != action_map.end()
           ? std::experimental::optional<ActivationParams>(action_map[activation_type])
           : std::experimental::nullopt;
}

void AlignShapeRank(std::vector<int64_t> *in_shape_ptr, const std::vector<int64_t> &out_shape) {
  const size_t last_dim = in_shape_ptr->size() - 1;
  const int in_rank = in_shape_ptr->size();
  int index = out_shape.size() - 1;
  for (; index >= 0; index--) {
    if (out_shape[index] == in_shape_ptr->at(last_dim)) {
      break;
    }
  }
  const int align_rank = index + 1;
  if (index <= 0 || align_rank == in_rank) return;
  for (int i = 0; i < index + 1 - in_rank; i++) {
    in_shape_ptr->insert(in_shape_ptr->begin(), 1);
  }
}

nvinfer1::ITensor *ConvertTensorWithExpandDims(TensorRTContext *ctx, const mindspore::MSTensor &ms_tensor,
                                               const std::vector<int64_t> &expect_shape, const std::string &op_name) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network is null for ConvertTensorWithExpandDims";
    return nullptr;
  }
  auto origin_shape = ms_tensor.Shape();
  std::vector<int64_t> convert_shape(expect_shape);
  AlignShapeRank(&origin_shape, convert_shape);
  size_t origin_index = 0;
  for (size_t i = 0; i < convert_shape.size(); ++i) {
    if (origin_index >= origin_shape.size()) {
      convert_shape[i] = 1;
      continue;
    }
    if (origin_shape[origin_index] != convert_shape[i]) {
      convert_shape[i] = origin_shape[origin_index];
    }
    origin_index++;
  }
  if (ms_tensor.ElementNum() !=
      std::accumulate(convert_shape.begin(), convert_shape.end(), 1, std::multiplies<int64_t>())) {
    MS_LOG(ERROR) << "ExpandDims failed for " << op_name;
    return nullptr;
  }
  nvinfer1::Dims dims = ConvertCudaDims(convert_shape);
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
  nvinfer1::IConstantLayer *constant_tensor = ctx->network()->addConstant(dims, weights);
  if (constant_tensor == nullptr) {
    MS_LOG(ERROR) << "create constant_tensor failed.";
    return nullptr;
  }
  ctx->RegisterLayer(constant_tensor, ms_tensor.Name() + "_" + op_name);
  auto tensor_ptr = constant_tensor->getOutput(0);
  return tensor_ptr;
}

nvinfer1::ITensor *ConvertConstantTensorWithDims(TensorRTContext *ctx, const mindspore::MSTensor &ms_tensor,
                                                 const std::vector<int64_t> &expect_shape, const std::string &op_name) {
  nvinfer1::ITensor *constant_input{nullptr};
  std::string tensor_name = op_name + "_" + ms_tensor.Name();
  if (ms_tensor.Shape().size() == 0 || ms_tensor.ElementNum() == 1) {
    constant_input =
      lite::ConvertScalarToITensor(ctx, expect_shape.size(), ms_tensor, ms_tensor.DataType(), tensor_name);
    if (constant_input == nullptr) {
      MS_LOG(ERROR) << "create Itensor from scalar tensor failed: " << tensor_name;
      return nullptr;
    }
  } else if (ms_tensor.Shape().size() == expect_shape.size()) {
    constant_input = lite::ConvertConstantTensor(ctx, ms_tensor, tensor_name);
    if (constant_input == nullptr) {
      MS_LOG(ERROR) << "create Itensor from constant tensor failed: " << tensor_name;
      return nullptr;
    }
  } else if (ms_tensor.ElementNum() >= 1) {
    constant_input = ConvertTensorWithExpandDims(ctx, ms_tensor, expect_shape, tensor_name);
    if (constant_input == nullptr) {
      MS_LOG(ERROR) << "create Itensor from ConvertTensorWithExpandDims failed: " << tensor_name;
      return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "const tensor value needs check: " << tensor_name;
  }
  return constant_input;
}

nvinfer1::Weights TransposeWeight4D(const mindspore::MSTensor &ms_tensor, void **pack_weight) {
  // usage notice: malloc addr saved to pack_weight, save pack_weight ptr and free it when deconstruct
  nvinfer1::Weights weights{};
  weights.count = ms_tensor.ElementNum();
  auto weight_shape = ms_tensor.Shape();
  if (weight_shape.size() != DIMENSION_4D) {
    MS_LOG(ERROR) << ms_tensor.Name() << " dims is " << weight_shape.size();
    return weights;
  }
  if (ms_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << ms_tensor.Name() << " has null data";
    return weights;
  }
  void *pack_weight_tmp = malloc(ms_tensor.DataSize());
  if (pack_weight_tmp == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return weights;
  }
  *pack_weight = pack_weight_tmp;
  weights.values = pack_weight_tmp;

  switch (ms_tensor.DataType()) {
    case DataType::kNumberTypeFloat16: {
      weights.type = nvinfer1::DataType::kHALF;
      PackNHWCToNCHWFp16(ms_tensor.Data().get(), pack_weight_tmp, weight_shape[0], weight_shape[1] * weight_shape[2],
                         weight_shape[3], 0, 0);
      break;
    }
    case DataType::kNumberTypeFloat32: {
      weights.type = nvinfer1::DataType::kFLOAT;
      PackNHWCToNCHWFp32(ms_tensor.Data().get(), pack_weight_tmp, weight_shape[0], weight_shape[1] * weight_shape[2],
                         weight_shape[3], 0, 0);
      break;
    }
    default: {
      MS_LOG(ERROR) << ms_tensor.Name() << " has unsupported tensor datatype for transpose data : "
                    << static_cast<int>(ms_tensor.DataType());
    }
  }
  return weights;
}

nvinfer1::Weights TransposeWeight2D(const mindspore::MSTensor &ms_tensor, void **pack_weight) {
  // usage notice: malloc addr saved to pack_weight, save pack_weight ptr and free it when deconstruct
  nvinfer1::Weights weights{};
  weights.count = ms_tensor.ElementNum();
  auto weight_shape = ms_tensor.Shape();
  if (weight_shape.size() != DIMENSION_2D) {
    MS_LOG(ERROR) << ms_tensor.Name() << " dims is " << weight_shape.size();
    return weights;
  }
  if (ms_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << ms_tensor.Name() << " has null data";
    return weights;
  }
  void *pack_weight_tmp = malloc(ms_tensor.DataSize());
  if (pack_weight_tmp == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return weights;
  }
  *pack_weight = pack_weight_tmp;
  weights.values = pack_weight_tmp;

  int row = weight_shape[0];
  int col = weight_shape[1];

  switch (ms_tensor.DataType()) {
    case DataType::kNumberTypeFloat16: {
      weights.type = nvinfer1::DataType::kHALF;
      auto src = static_cast<const uint16_t *>(ms_tensor.Data().get());
      auto dst = static_cast<uint16_t *>(pack_weight_tmp);
      for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
          dst[c * row + r] = src[r * col + c];
        }
      }
      break;
    }
    case DataType::kNumberTypeFloat32: {
      weights.type = nvinfer1::DataType::kFLOAT;
      auto dst = static_cast<float *>(pack_weight_tmp);
      auto src = static_cast<const float *>(ms_tensor.Data().get());
      for (int r = 0; r < row; ++r) {
        for (int c = 0; c < col; ++c) {
          dst[c * row + r] = src[r * col + c];
        }
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << ms_tensor.Name() << " has unsupported tensor datatype for transpose data : "
                    << static_cast<int>(ms_tensor.DataType());
    }
  }
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

nvinfer1::ITensor *TRTTensorCast(TensorRTContext *ctx, nvinfer1::ITensor *trt_tensor, nvinfer1::DataType data_type,
                                 const std::string &name) {
#if TRT_VERSION_GE(7, 2)
  data_type == nvinfer1::DataType::kBOOL ? nvinfer1::DataType::kINT32 : data_type;
  auto cast_layer = ctx->network()->addIdentity(*trt_tensor);
#else
  auto plugin = std::make_shared<CastPlugin>(name, trt_tensor->getType(), data_type);
  nvinfer1::ITensor *inputTensors[] = {trt_tensor};
  nvinfer1::IPluginV2Layer *cast_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
#endif
  if (cast_layer == nullptr) {
    MS_LOG(ERROR) << "create cast layer failed for: " << name;
    return nullptr;
  }
#if TRT_VERSION_GE(7, 2)
  cast_layer->setOutputType(0, data_type);
#endif
  cast_layer->setName(name.c_str());
  nvinfer1::ITensor *cast_out = cast_layer->getOutput(0);
  cast_out->setName((name + "_output").c_str());
  return cast_out;
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
    if (perm.order[kNHWC_N] == kNHWC_N && perm.order[kNHWC_H] == kNHWC_C && perm.order[kNHWC_W] == kNHWC_W &&
        perm.order[kNHWC_C] == kNHWC_H) {
      return Format::NCHW;
    }
  } else if (input_format == Format::NCHW) {
    if (perm.order[kNCHW_N] == kNCHW_N && perm.order[kNCHW_C] == kNCHW_H && perm.order[kNCHW_H] == kNCHW_W &&
        perm.order[kNCHW_W] == kNCHW_C) {
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
std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor, mindspore::Format format, bool is_same, bool is_tensor) {
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
  out_string += " is_tensor(0/1): " + std::to_string(is_tensor);
  return out_string;
}

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor, mindspore::Format format, bool is_same) {
  return GetTensorFormat(trt_tensor, format, is_same, true);
}

std::string GetTensorFormat(ITensorHelper tensor_helper) {
  return GetTensorFormat(tensor_helper.trt_tensor_, tensor_helper.format_, tensor_helper.same_format_,
                         tensor_helper.is_tensor_);
}

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor) { return GetTensorFormat(trt_tensor, Format::NHWC, true); }

std::experimental::optional<nvinfer1::ReduceOperation> TryConvertTRTReduceMode(schema::ReduceMode mode) {
  std::map<schema::ReduceMode, nvinfer1::ReduceOperation> reduce_ops_ = {
    {schema::ReduceMode::ReduceMode_ReduceMean, nvinfer1::ReduceOperation::kAVG},
    {schema::ReduceMode::ReduceMode_ReduceMax, nvinfer1::ReduceOperation::kMAX},
    {schema::ReduceMode::ReduceMode_ReduceMin, nvinfer1::ReduceOperation::kMIN},
    {schema::ReduceMode::ReduceMode_ReduceProd, nvinfer1::ReduceOperation::kPROD},
    {schema::ReduceMode::ReduceMode_ReduceL2, nvinfer1::ReduceOperation::kSUM},
    {schema::ReduceMode::ReduceMode_ReduceSum, nvinfer1::ReduceOperation::kSUM},
  };
  return reduce_ops_.find(mode) != reduce_ops_.end()
           ? std::experimental::optional<nvinfer1::ReduceOperation>(reduce_ops_[mode])
           : std::experimental::nullopt;
}
int PreprocessInputs2SameDim(TensorRTContext *ctx, ITensorHelper input_tensor_helper,
                             ITensorHelper *out_tensor_helper) {
  out_tensor_helper->trt_tensor_ = input_tensor_helper.trt_tensor_;
  out_tensor_helper->format_ = input_tensor_helper.format_;
  out_tensor_helper->same_format_ = true;
  if (input_tensor_helper.trt_tensor_->getDimensions().nbDims == DIMENSION_4D && !input_tensor_helper.same_format_) {
    if (input_tensor_helper.format_ == Format::NCHW) {
      // transpose: NCHW->NHWC
      nvinfer1::IShuffleLayer *transpose_layer_in = NCHW2NHWC(ctx, *input_tensor_helper.trt_tensor_);
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
      nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(ctx, *input_tensor_helper.trt_tensor_);
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
  if (dims.nbDims <= 0) {
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

std::experimental::optional<nvinfer1::Dims> SqueezeDims(const nvinfer1::Dims &in_dims, int pos) {
  if (in_dims.nbDims <= 1) {
    MS_LOG(ERROR) << "invalid shape size: " << in_dims.nbDims << "for squeeze.";
    return {};
  }
  nvinfer1::Dims out_dims;
  int i = 0;
  for (int j = 0; j <= in_dims.nbDims; ++j) {
    if (j != pos) {
      out_dims.d[i++] = in_dims.d[j];
    }
  }
  out_dims.nbDims = in_dims.nbDims - 1;
  return std::experimental::optional<nvinfer1::Dims>(out_dims);
}

std::experimental::optional<nvinfer1::Dims> UnsqueezeDims(const nvinfer1::Dims &in_dims, int pos, int val) {
  if (in_dims.nbDims >= static_cast<size_t>(in_dims.MAX_DIMS)) {
    MS_LOG(ERROR) << "invalid shape size: " << in_dims.nbDims << "for unsqueeze.";
    return {};
  }
  nvinfer1::Dims out_dims;
  int i = 0;
  for (int j = 0; j <= in_dims.nbDims; ++j) {
    if (j == pos) {
      out_dims.d[j] = val;
    } else {
      out_dims.d[j] = in_dims.d[i++];
    }
  }
  out_dims.nbDims = in_dims.nbDims + 1;
  return std::experimental::optional<nvinfer1::Dims>(out_dims);
}

int ParseData2Vector(const mindspore::MSTensor &ms_tensor, std::vector<float> *dst) {
  if (ms_tensor.Data() == nullptr) {
    MS_LOG(ERROR) << "ignore tensor: " << ms_tensor.Name();
    return RET_ERROR;
  }
  dst->clear();
  dst->resize(ms_tensor.ElementNum());
  switch (ms_tensor.DataType()) {
    case DataType::kNumberTypeInt64: {
      Data2Vector<int64_t>(dst, ms_tensor.Data().get());
      break;
    }
    case DataType::kNumberTypeInt32: {
      Data2Vector<int>(dst, ms_tensor.Data().get());
      break;
    }
    default: {
      MS_LOG(ERROR) << ms_tensor.Name() << " has more datatype to parse";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

nvinfer1::ITensor *ExpandDim(TensorRTContext *ctx, nvinfer1::ITensor *input_tensor, int axis) {
  // input has to prepocess to nchw
  auto input_dims = input_tensor->getDimensions();
  nvinfer1::IShuffleLayer *shuffle_layer = ctx->network()->addShuffle(*input_tensor);
  // if expand dim not at last dim and shape is dynamic, change to expanddim at last dim and transpose
  bool special_expand = false;
  for (int i = 0; i < input_dims.nbDims; i++) {
    special_expand = special_expand || input_dims.d[i] == -1;
  }
  special_expand = special_expand && (axis != -1 && axis != input_dims.nbDims);

  if (special_expand) {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_dims.nbDims; i++) {
      new_shape.push_back(input_dims.d[i] == -1 ? 0 : input_dims.d[i]);
    }
    new_shape.push_back(1);
    nvinfer1::Dims new_dims = ConvertCudaDims(new_shape);
    if (new_dims.nbDims == -1) {
      return nullptr;
    }

    shuffle_layer->setReshapeDimensions(new_dims);
    // transpose
    nvinfer1::Permutation perm{};
    for (int i = 0; i < new_dims.nbDims; i++) {
      if (i < axis) {
        perm.order[i] = i;
      } else if (i == axis) {
        perm.order[i] = new_dims.nbDims - 1;
      } else {
        perm.order[i] = i - 1;
      }
    }
    nvinfer1::IShuffleLayer *trans_layer = ctx->network()->addShuffle(*shuffle_layer->getOutput(0));
    if (trans_layer == nullptr) {
      MS_LOG(ERROR) << "add transpose layer failed for special expand dims op ";
      return nullptr;
    }
    trans_layer->setFirstTranspose(perm);
    return trans_layer->getOutput(0);
  } else {
    std::vector<int64_t> new_shape;
    for (int i = 0; i < input_dims.nbDims; i++) {
      if (axis == i) {
        new_shape.push_back(1);
      }
      new_shape.push_back(input_dims.d[i] == -1 ? 0 : input_dims.d[i]);
    }
    if (axis == -1 || axis == input_dims.nbDims) {
      new_shape.push_back(1);
    }
    nvinfer1::Dims new_dims = ConvertCudaDims(new_shape);
    if (new_dims.nbDims == -1) {
      return nullptr;
    }
    shuffle_layer->setReshapeDimensions(new_dims);
    return shuffle_layer->getOutput(0);
  }
}

nvinfer1::ITensor *Broadcast(TensorRTContext *ctx, nvinfer1::ITensor *input, nvinfer1::ITensor *shape) {
  int rank = shape->getDimensions().d[0];

  nvinfer1::Dims starts{rank};
  std::fill(starts.d, starts.d + rank, 0);
  nvinfer1::Dims strides{rank};
  std::fill(strides.d, strides.d + rank, 1);

  auto slice_layer = ctx->network()->addSlice(*input, starts, {}, strides);
#if TRT_VERSION_GE(7, 2)
  slice_layer->setMode(nvinfer1::SliceMode::kWRAP);
#endif
  slice_layer->setInput(INPUT2, *shape);

  auto shuffler_output = slice_layer->getOutput(0);
  if (shuffler_output == nullptr) {
    MS_LOG(ERROR) << "add slice layer failed";
  }
  return shuffler_output;
}

nvinfer1::ITensor *Reshape(TensorRTContext *ctx, nvinfer1::ITensor *input, const std::vector<int64_t> &shape) {
  return Reshape(ctx, input, ConvertCudaDims(shape));
}

nvinfer1::ITensor *Reshape(TensorRTContext *ctx, nvinfer1::ITensor *input, const nvinfer1::Dims &shape) {
  auto reshape_layer = ctx->network()->addShuffle(*input);
  if (reshape_layer == nullptr) {
    MS_LOG(ERROR) << "add reshape_layer failed";
    return nullptr;
  }
  reshape_layer->setReshapeDimensions(shape);
  return reshape_layer->getOutput(0);
}

void DebugDims(const std::string &key, const nvinfer1::Dims &dims) {
  MS_LOG(DEBUG) << key << ":" << dims.nbDims;
  for (int i = 0; i != dims.nbDims; ++i) {
    MS_LOG(DEBUG) << dims.d[i];
  }
}

template <>
nvinfer1::DataType GetNvinferDataType<float>() {
  return nvinfer1::DataType::kFLOAT;
}

template <>
nvinfer1::DataType GetNvinferDataType<int>() {
  return nvinfer1::DataType::kINT32;
}

template nvinfer1::DataType GetNvinferDataType<float>();
template nvinfer1::DataType GetNvinferDataType<int>();
}  // namespace mindspore::lite
