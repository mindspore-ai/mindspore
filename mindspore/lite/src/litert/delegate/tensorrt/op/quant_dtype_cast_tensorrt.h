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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_TENSORRT_H_
#include <string>
#include <vector>
#include "src/litert/delegate/tensorrt/op/quant_dtype_cast_plugin.h"
#include "src/litert/delegate/tensorrt/op/fse_plugin.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_op.h"
#include "src/litert/delegate/tensorrt/op/tensorrt_plugin.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/quant_impl.cuh"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/tensor.h"

namespace mindspore::lite {
class QuantDTypeCastTensorRT : public TensorRTOp {
 public:
  QuantDTypeCastTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                         const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                         const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~QuantDTypeCastTensorRT();

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  int Deserialize(int8_t *data8, size_t data_size);
  int AddQuantPlugin(TensorRTContext *ctx, mindspore::lite::Tensor *lite_tensor);
  int AddFSEPlugin(TensorRTContext *ctx, mindspore::lite::Tensor *lite_tensor);

  void *int32_align_ = nullptr;
  const schema::QuantDTypeCast *quant_dtype_cast_ = nullptr;

  std::vector<float> scales_;
  std::vector<int> zps_;
  float *scale_device_ptr_ = nullptr;
  int *zp_device_ptr_ = nullptr;

  BitStreamState bs_;
  size_t table_log_ = 0;
  size_t table_size_ = 0;
  uint64_t *chunks_ = nullptr;  // the actual memory
  uint32_t chunk_count_ = 0;
  size_t chunk_size_ = 0;
  uint16_t *states_table_ = nullptr;
  uint8_t *bit_count_table_ = nullptr;
  uint16_t *symbol_table_ = nullptr;
  void *centroids_ = nullptr;
  size_t centroids_size_ = 0;

  uint16_t *states_table_device_ = nullptr;
  uint8_t *bit_count_table_device_ = nullptr;
  uint16_t *symbol_table_device_ = nullptr;
  void *centroids_device_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_QUANT_DTYPE_CAST_TENSORRT_H_
