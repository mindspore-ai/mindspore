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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_AKG_TENSORRT_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_AKG_TENSORRT_H_

#include <cuda.h>
#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
const int MAX_REGISTER_PER_THREAD_BLOCK = 65536;
const int REGISTER_UNIT_IN_WARP = 256;
const int WARP_SIZE = 32;
const int WARP_ALLOC_GRAN = 4;
const int LEN_LIMIT = 200;

typedef struct {
  char ptx_path[LEN_LIMIT];
  char kernel_name[LEN_LIMIT];
  size_t ptx_path_len;
  uint32_t output_shapes[LEN_LIMIT];
  uint32_t output_shapes_separators[LEN_LIMIT];
  size_t output_types_len;
  size_t output_types[LEN_LIMIT];
  size_t bx;
  size_t by;
  size_t bz;
  size_t tx;
  size_t ty;
  size_t tz;
} AkgParamT;
class AkgTensorRT : public TensorRTOp {
 public:
  AkgTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
              const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {
    dynamic_shape_params_.support_dynamic_ = false;
    dynamic_shape_params_.support_hw_dynamic_ = false;
  }

  ~AkgTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto AKG_PLUGIN_NAME{"AkgPlugin"};
class AkgPlugin : public TensorRTPlugin {
 public:
  AkgPlugin(const std::string name, AkgParamT params, uint32_t device_id)
      : TensorRTPlugin(name, std::string(AKG_PLUGIN_NAME), device_id), params_(params) {}

  AkgPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(AKG_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    params_ = static_cast<const AkgParamT *>(fields[0].data)[0];
  }

  AkgPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(AKG_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &params_, sizeof(AkgParamT));
  }

  AkgPlugin() = delete;

  ~AkgPlugin() override {}

  CUresult GetFunction(CUfunction *func);
  bool Launch(const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream);

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
    noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                 int nbOutputs) noexcept override;
  void terminate() noexcept override;
  int initialize() noexcept override;

 private:
  const std::string layer_name_;
  std::string name_space_;
  mutable AkgParamT params_;
  CUfunction kernel_addr_{nullptr};
  size_t num_of_inputs_ = 0;
  size_t num_of_outputs_ = 0;
  size_t num_of_workspace_ = 0;
};
class AkgPluginCreater : public TensorRTPluginCreater<AkgPlugin> {
 public:
  AkgPluginCreater() : TensorRTPluginCreater(std::string(AKG_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_AKG_TENSORRT_H_
