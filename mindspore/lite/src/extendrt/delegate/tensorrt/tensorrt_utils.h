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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_UTILS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSORRT_UTILS_H_
#include <experimental/optional>
#include <vector>
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <memory>
#include <string>
#include "src/extendrt/delegate/tensorrt/tensorrt_context.h"
#include "src/extendrt/delegate/tensorrt/tensor_info.h"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cublas_utils.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "schema/ops_generated.h"
#include "nnacl/pack.h"
#include "include/api/context.h"
#include "mindapi/base/types.h"

#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3
#define kNHWC_N 0
#define kNHWC_H 1
#define kNHWC_W 2
#define kNHWC_C 3

namespace mindspore::lite {
#define TRT_VERSION_GE(major, minor) \
  (NV_TENSORRT_MAJOR > major) || ((NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR >= minor))
#define TRT_VERSION_LS(major, minor) \
  (NV_TENSORRT_MAJOR < major) || ((NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR < minor))
struct ActivationParams {
  nvinfer1::ActivationType activation_type;
  bool has_alpha;
  float alpha;
  bool has_beta;
  float beta;
};

typedef union float32_bits {
  unsigned int u;
  float f;
} float32_bits;

// Convert Tensor data to Cuda dims.
nvinfer1::Dims ConvertCudaDims(const void *data, int64_t size);

nvinfer1::Dims ConvertCudaDims(int data, size_t size);

nvinfer1::Dims ConvertCudaDims(const TensorInfo &ms_tensor);

std::string CudaDimsAsString(const nvinfer1::Dims &dims);

std::vector<int32_t> ConvertTensorAsIntVector(const TensorInfo &ms_tensor);

bool SameDims(nvinfer1::Dims dims, const std::vector<int64_t> &shape);

std::vector<int64_t> ConvertMSShape(const nvinfer1::Dims dims);

std::vector<int64_t> NHWC2NCHW(std::vector<int64_t> nhwc_shape);

nvinfer1::DataType ConvertDataType(DataType type_id);

cudaDataType ConvertDataType(nvinfer1::DataType type_id);

nvinfer1::IShuffleLayer *NHWC2NCHW(TensorRTContext *ctx, const nvinfer1::ITensor &input);

nvinfer1::IShuffleLayer *NCHW2NHWC(TensorRTContext *ctx, const nvinfer1::ITensor &input);

std::experimental::optional<ActivationParams> TryConvertActivationType(ActivationType activation_type);

nvinfer1::ITensor *ConvertConstantTensor(TensorRTContext *ctx, const TensorInfo &ms_tensor, const std::string &op_name);

nvinfer1::ITensor *ConvertTensorWithExpandDims(TensorRTContext *ctx, const TensorInfo &ms_tensor,
                                               const std::vector<int64_t> &expect_shape, const std::string &op_name);

nvinfer1::ITensor *ConvertScalarToITensor(TensorRTContext *ctx, size_t shape_size, const void *value,
                                          const DataType data_type, const std::string &op_name);

nvinfer1::ITensor *ConvertScalarToITensor(TensorRTContext *ctx, size_t shape_size, const TensorInfo &ms_tensor,
                                          const DataType data_type, const std::string &op_name);

nvinfer1::ITensor *ConvertConstantTensorWithDims(TensorRTContext *ctx, const TensorInfo &ms_tensor,
                                                 const std::vector<int64_t> &expect_shape, const std::string &op_name);

nvinfer1::Weights TransposeWeight2D(const TensorInfo &ms_tensor, void **pack_weight);

nvinfer1::Weights ConvertWeight(const TensorInfo &ms_tensor);

nvinfer1::ITensor *TRTTensorCast(TensorRTContext *ctx, nvinfer1::ITensor *tensor, nvinfer1::DataType data_type,
                                 const std::string &name);

int SetCudaDevice(std::shared_ptr<GPUDeviceInfo> device_info_);

int SetCudaDevice(int device_id);

Format GetOutputFormat(Format input_format, nvinfer1::Permutation perm);

int ConvertAxisFromNHWC2NCHW(int nhwc_axis);

void PackNHWCToNCHWFp16(const void *src, void *dst, size_t batch, size_t plane, size_t channel, size_t task_id,
                        size_t thread_count);

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensor, mindspore::Format format, bool is_same);

std::string GetTensorFormat(ITensorHelper tensor_helper);

std::string GetTensorFormat(nvinfer1::ITensor *trt_tensors);

std::experimental::optional<nvinfer1::ReduceOperation> TryConvertTRTReduceMode(ReduceMode mode);

int PreprocessInputs2SameDim(TensorRTContext *ctx, ITensorHelper input_tensor_helper, ITensorHelper *out_tensor_helper);

int GetDimsVolume(const nvinfer1::Dims &dims);

int GetDimsVolume(const std::vector<int64_t> &shape);

std::experimental::optional<nvinfer1::Dims> SqueezeDims(const nvinfer1::Dims &in_dims, int pos);

std::experimental::optional<nvinfer1::Dims> UnsqueezeDims(const nvinfer1::Dims &in_dims, int pos, int val);

nvinfer1::ITensor *Reshape(TensorRTContext *ctx, nvinfer1::ITensor *input, const std::vector<int64_t> &shape);

nvinfer1::ITensor *Reshape(TensorRTContext *ctx, nvinfer1::ITensor *input, const nvinfer1::Dims &shape);

nvinfer1::ITensor *ConvertConstantTensor1D(TensorRTContext *ctx, int *weights_vec, nvinfer1::DataType data_type);

int ParseData2Vector(const TensorInfo &ms_tensor, std::vector<float> *dst);

void DebugDims(const std::string &key, const nvinfer1::Dims &dims);

nvinfer1::ITensor *ExpandDim(TensorRTContext *ctx, nvinfer1::ITensor *input_tensor, int axis);

nvinfer1::ITensor *Broadcast(TensorRTContext *ctx, nvinfer1::ITensor *input, nvinfer1::ITensor *shape);

template <typename T>
nvinfer1::DataType GetNvinferDataType();

template <typename T1, typename T2>
bool SameDims(const std::vector<T1> &shape1, const std::vector<T2> &shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  for (size_t i = 0; i < shape1.size(); i++) {
    if (std::abs(shape1[i] - shape2[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

template <typename T>
nvinfer1::Dims ConvertCudaDims(const std::vector<T> &shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = -1;
  if (!shape.empty() && shape.size() <= static_cast<size_t>(dims.MAX_DIMS)) {
    dims.nbDims = shape.size();
    for (int i = 0; i < dims.nbDims; i++) {
      dims.d[i] = static_cast<int>(shape[i]);
    }
  } else {
    MS_LOG(INFO) << "ms shape is invalid!shape size: " << shape.size();
  }
  return dims;
}

inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(WARNING) << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}
template <typename T>
void Data2Vector(std::vector<float> *dst, const void *src) {
  auto src_ptr = static_cast<const T *>(src);
  for (size_t i = 0; i < dst->size(); i++) {
    dst->at(i) = static_cast<float>(src_ptr[i]);
  }
}
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_UTILS_H_
