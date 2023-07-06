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

#include "src/extendrt/delegate/tensorrt/op/akg_tensorrt.h"
#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <numeric>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "ops/attention.h"
#include "ops/custom.h"
#include "tools/graph_kernel/common/utils.h"

namespace mindspore::lite {
std::string ReadFileToString(std::string filename) {
  std::ifstream ifile(filename.c_str());
  std::ostringstream buf;
  char ch;
  while (buf && ifile.get(ch)) {
    buf.put(ch);
  }
  return buf.str();
}

int AkgTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }

  auto akg_op = AsOps<ops::Custom>();
  if (akg_op == nullptr) {
    MS_LOG(ERROR) << "op action convert failed";
    return RET_ERROR;
  }

  auto attr_map = akg_op->get_attr();
  AkgParamT params;
  memset_s(&params, sizeof(params), 0, sizeof(params));
  std::string ptx_path_str =
    std::string(reinterpret_cast<const char *>(attr_map["ptx_path"].data()), attr_map["ptx_path"].size());
  params.ptx_path_len = ptx_path_str.length();
  auto res0 = snprintf_s(params.ptx_path, LEN_LIMIT, LEN_LIMIT - 1, "%s", ptx_path_str.c_str());
  if (res0 < 0) {
    MS_LOG(ERROR) << "snprintf_s encountered an encoding error or a runtime-constraint violation.";
  } else if (res0 >= static_cast<int>(LEN_LIMIT)) {
    MS_LOG(ERROR) << "snprintf_s output was truncated.";
  }

  std::string kernel_name_str =
    std::string(reinterpret_cast<const char *>(attr_map["kernel_name"].data()), attr_map["kernel_name"].size());
  kernel_name_str += "_kernel0";
  auto res1 = snprintf_s(params.kernel_name, LEN_LIMIT, LEN_LIMIT - 1, "%s", kernel_name_str.c_str());
  if (res1 < 0) {
    MS_LOG(ERROR) << "snprintf_s encountered an encoding error or a runtime-constraint violation.";
  } else if (res1 >= static_cast<int>(LEN_LIMIT)) {
    MS_LOG(ERROR) << "snprintf_s output was truncated.";
  }
  std::string outputs_shape_str(reinterpret_cast<const char *>(attr_map["outputs_shape"].data()),
                                attr_map["outputs_shape"].size());

  std::vector<std::vector<int>> outputs_shape;
  (void)graphkernel::GetCustomShape(outputs_shape_str, &outputs_shape);
  size_t idx = 0;
  size_t num_output = 0;
  for (auto shp : outputs_shape) {
    for (auto v : shp) {
      params.output_shapes[idx] = v;
      idx += 1;
    }
    params.output_shapes_separators[num_output] = idx;
    num_output += 1;
  }

  std::string outputs_type_str(reinterpret_cast<const char *>(attr_map["outputs_type"].data()),
                               attr_map["outputs_type"].size());
  std::vector<std::string> outputs_type = graphkernel::SplitString(outputs_type_str, ',');
  params.output_types_len = outputs_type.size();
  for (size_t i = 0; i < outputs_type.size(); i++) {
    params.output_types[i] = std::stoul(outputs_type[i]);
  }

  params.bx =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["GridDimX"].data()), attr_map["GridDimX"].size()));
  params.by =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["GridDimY"].data()), attr_map["GridDimY"].size()));
  params.bz =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["GridDimZ"].data()), attr_map["GridDimZ"].size()));
  params.tx =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["BlockDimX"].data()), attr_map["BlockDimX"].size()));
  params.ty =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["BlockDimY"].data()), attr_map["BlockDimY"].size()));
  params.tz =
    std::stoul(std::string(reinterpret_cast<const char *>(attr_map["BlockDimZ"].data()), attr_map["BlockDimZ"].size()));

  nvinfer1::ITensor *input_tensor = input(ctx, 0).trt_tensor_;
  auto plugin = std::make_shared<AkgPlugin>(input_tensor->getName(), params, device_id_);
  const size_t input_number = inputs().size();
  nvinfer1::ITensor *inputTensors[input_number];
  for (size_t i = 0; i < input_number; i++) {
    inputTensors[i] = input(ctx, i).trt_tensor_;
  }
  nvinfer1::IPluginV2Layer *akg_layer = ctx->network()->addPluginV2(inputTensors, input_number, *plugin);
  if (akg_layer == nullptr) {
    MS_LOG(ERROR) << "add akg op failed for TensorRT.";
    return RET_ERROR;
  }
  akg_layer->setName((op_name_ + "plugin_akg").c_str());  // should be set as Fused_MatMul_Add_XXX_fusion??
  const size_t output_number = outputs().size();
  for (size_t i = 0; i < output_number; i++) {
    nvinfer1::ITensor *out_tensor = akg_layer->getOutput(i);
    ctx->RegisterTensor(ITensorHelper{out_tensor, Format::DEFAULT_FORMAT, false, true}, out_tensors_[i].Name());
  }
  this->layer_ = akg_layer;
  return RET_OK;
}

int AkgTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

//  PLUGIN of Akg Layer
REGISTER_TENSORRT_PLUGIN(AkgPluginCreater);
template class TensorRTPluginCreater<AkgPlugin>;
template <class T>
nvinfer1::PluginFieldCollection TensorRTPluginCreater<T>::field_collection_{};
template <class T>
std::vector<nvinfer1::PluginField> TensorRTPluginCreater<T>::fields_;

CUresult AkgPlugin::GetFunction(CUfunction *func) {
  CUmodule module;
  CUjit_option options[1];
  options[0] = CU_JIT_MAX_REGISTERS;
  void *values[1];
  int total_threads = params_.tx * params_.ty * params_.tz;
  int total_warps = std::ceil(static_cast<float>(total_threads) / static_cast<float>(WARP_SIZE));
  int limit_warps = (total_warps + WARP_ALLOC_GRAN - 1) / WARP_ALLOC_GRAN * WARP_ALLOC_GRAN;
  int total_register_unit_nums = MAX_REGISTER_PER_THREAD_BLOCK / REGISTER_UNIT_IN_WARP;
  int register_unit_nums_per_warp = total_register_unit_nums / limit_warps;
  int64_t register_nums = (register_unit_nums_per_warp * REGISTER_UNIT_IN_WARP) / WARP_SIZE;
  values[0] = reinterpret_cast<void *>(register_nums);

  std::string ptx_path_str = std::string(params_.ptx_path);
  CUresult result = cuModuleLoadDataEx(&module, ReadFileToString(ptx_path_str).c_str(), 1, options, values);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "cuModuleLoadDataEx failed. Kernel name: << " << params_.kernel_name << ". Error message: " << msg;
    return result;
  }
  result = cuModuleGetFunction(func, module, params_.kernel_name);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "cuModuleGetFunction failed. Kernel name: << " << params_.kernel_name
                  << ". Error message: " << msg;
    return result;
  }
  return result;
}

bool AkgPlugin::Launch(const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
  if (stream == 0) {
    MS_LOG(ERROR) << "stream should not be nullptr. Kernel name: " << params_.kernel_name;
    return false;
  }
  std::vector<void *> runtimeargs;
  void *addr_ptrs[num_of_inputs_ + num_of_outputs_];
  void *addr[num_of_inputs_ + num_of_outputs_];
  for (size_t i = 0; i < num_of_inputs_; i++) {
    addr[i] = const_cast<void *>(inputs[i]);
    addr_ptrs[i] = &(addr[i]);
    runtimeargs.push_back(addr_ptrs[i]);
  }
  for (size_t i = 0; i < num_of_outputs_; i++) {
    addr[i + num_of_inputs_] = const_cast<void *>(outputs[i]);
    addr_ptrs[i + num_of_inputs_] = &(addr[i + num_of_inputs_]);
    runtimeargs.push_back(addr_ptrs[i + num_of_inputs_]);
  }

  CUresult result = cuLaunchKernel(kernel_addr_, params_.bx, params_.by, params_.bz, params_.tx, params_.ty, params_.tz,
                                   0, stream, reinterpret_cast<void **>(&runtimeargs[0]), 0);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_LOG(ERROR) << "Launch kernel failed. Kernel name: " << params_.kernel_name
                  << ". cuLaunchKernel error message: " << msg;
    return false;
  }
  return true;
}

int AkgPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                       const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  (void)Launch(inputs, outputs, workspace, stream);
  return RET_OK;
}

bool AkgPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *tensorsDesc, int nbInputs,
                                          int nbOutputs) noexcept {
  return true;
}

nvinfer1::DimsExprs AkgPlugin::getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                                   nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs dims;
  size_t start = 0;
  size_t end = 0;
  if (index != 0) {
    start = params_.output_shapes_separators[index - 1];
  }
  end = params_.output_shapes_separators[index];
  dims.nbDims = end - start;
  for (size_t i = start; i < end; i++) {
    dims.d[i - start] = exprBuilder.constant(params_.output_shapes[i]);
  }
  return dims;
}

nvinfer1::DataType AkgPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
  noexcept {
  if (params_.output_types[index] == size_t(DataType::kNumberTypeFloat32)) {
    return nvinfer1::DataType::kFLOAT;
  } else if (params_.output_types[index] == size_t(DataType::kNumberTypeFloat16)) {
    return nvinfer1::DataType::kHALF;
  } else if (params_.output_types[index] == size_t(DataType::kNumberTypeInt32)) {
    return nvinfer1::DataType::kINT32;
  } else if (params_.output_types[index] == size_t(DataType::kNumberTypeInt8)) {
    return nvinfer1::DataType::kINT8;
  } else if (params_.output_types[index] == size_t(DataType::kNumberTypeBool)) {
    return nvinfer1::DataType::kBOOL;
  } else {
    MS_EXCEPTION(TypeError);
  }
}

nvinfer1::IPluginV2DynamicExt *AkgPlugin::clone() const noexcept {
  auto *plugin = new AkgPlugin(*this);
  if (plugin == nullptr) {
    MS_LOG(ERROR) << "plugin is null";
    return nullptr;
  }
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

void AkgPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
  num_of_inputs_ = nbInputs;
  num_of_outputs_ = nbOutputs;
}
int AkgPlugin::initialize() noexcept {
  CUresult result = GetFunction(&kernel_addr_);
  if (result != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorName(result, &msg);
    MS_EXCEPTION(RuntimeError) << "Get function " << params_.kernel_name << " failed. Error message: " << msg;
  }
  return 0;
}

void AkgPlugin::terminate() noexcept {}

size_t AkgPlugin::getSerializationSize() const noexcept { return sizeof(AkgParamT); }

void AkgPlugin::serialize(void *buffer) const noexcept { SerializeValue(&buffer, &params_, sizeof(AkgParamT)); }

REGISTER_TENSORRT_CREATOR("CustomAkgGpu", AkgTensorRT)
}  // namespace mindspore::lite
