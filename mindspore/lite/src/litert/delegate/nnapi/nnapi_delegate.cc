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

#include "src/litert/delegate/nnapi/nnapi_delegate.h"
#include <queue>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include "include/errorcode.h"
#include "src/litert/delegate/delegate_utils.h"
#include "src/litert/delegate/nnapi/nnapi_utils.h"
#include "src/litert/delegate/nnapi/op/activation_nnapi.h"
#include "src/litert/delegate/nnapi/op/arithmetic_nnapi.h"
#include "src/litert/delegate/nnapi/op/cast_nnapi.h"
#include "src/litert/delegate/nnapi/op/concat_nnapi.h"
#include "src/litert/delegate/nnapi/op/conv_nnapi.h"
#include "src/litert/delegate/nnapi/op/conv_transpose_nnapi.h"
#include "src/litert/delegate/nnapi/op/full_connection_nnapi.h"
#include "src/litert/delegate/nnapi/op/gather_nnapi.h"
#include "src/litert/delegate/nnapi/op/instance_norm_nnapi.h"
#include "src/litert/delegate/nnapi/op/padding_nnapi.h"
#include "src/litert/delegate/nnapi/op/pooling_nnapi.h"
#include "src/litert/delegate/nnapi/op/reshape_nnapi.h"
#include "src/litert/delegate/nnapi/op/resize_nnapi.h"
#include "src/litert/delegate/nnapi/op/reduce_nnapi.h"
#include "src/litert/delegate/nnapi/op/scale_nnapi.h"
#include "src/litert/delegate/nnapi/op/split_nnapi.h"
#include "src/litert/delegate/nnapi/op/stack_nnapi.h"
#include "src/litert/delegate/nnapi/op/softmax_nnapi.h"
#include "src/litert/delegate/nnapi/op/strided_slice_nnapi.h"
#include "src/litert/delegate/nnapi/op/transpose_nnapi.h"
#include "src/litert/delegate/nnapi/op/topk_nnapi.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
void GetSpecifiedDevices(const std::vector<std::string> &specified_devices, bool only_use_acc_device,
                         bool disable_cpu_device, std::vector<ANeuralNetworksDevice *> *devices) {
  uint32_t device_count;
  nnapi_->ANeuralNetworks_getDeviceCount(&device_count);
  int32_t type;
  const char *name;
  ANeuralNetworksDevice *device;
  for (auto idx = 0; idx < device_count; idx++) {
    nnapi_->ANeuralNetworks_getDevice(idx, &device);
    nnapi_->ANeuralNetworksDevice_getName(device, &name);
    nnapi_->ANeuralNetworksDevice_getType(device, &type);
    MS_LOG(DEBUG) << "Found available device: " << name << ", and the type is: " << type;
    if (std::find(specified_devices.begin(), specified_devices.end(), name) != specified_devices.end()) {
      devices->push_back(device);
      continue;
    }
    if (specified_devices.empty() && only_use_acc_device && type == ANEURALNETWORKS_DEVICE_ACCELERATOR) {
      devices->push_back(device);
      continue;
    }
    if (specified_devices.empty() && disable_cpu_device && type != ANEURALNETWORKS_DEVICE_CPU) {
      devices->push_back(device);
    }
  }
}

Status NNAPIDelegate::Init() {
  if (nnapi_ == nullptr || !nnapi_->nnapi_exists) {
    MS_LOG(ERROR) << "NNAPI is not available.";
    return mindspore::kLiteNullptr;
  }
  if (nnapi_->android_sdk_version >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
    GetSpecifiedDevices(specified_devices_, only_use_acc_device_, disable_cpu_device_, &devices_);
  }
  op_func_lists_ = {
    {schema::PrimitiveType_Activation, GetNNAPIOp<NNAPIActivation>},
    {schema::PrimitiveType_AddFusion, GetNNAPIOp<NNAPIArithmetic>},
    {schema::PrimitiveType_SubFusion, GetNNAPIOp<NNAPIArithmetic>},
    {schema::PrimitiveType_MulFusion, GetNNAPIOp<NNAPIArithmetic>},
    {schema::PrimitiveType_DivFusion, GetNNAPIOp<NNAPIArithmetic>},
    {schema::PrimitiveType_Cast, GetNNAPIOp<NNAPICast>},
    {schema::PrimitiveType_Concat, GetNNAPIOp<NNAPIConcat>},
    {schema::PrimitiveType_Conv2DFusion, GetNNAPIOp<NNAPIConv>},
    {schema::PrimitiveType_Conv2dTransposeFusion, GetNNAPIOp<NNAPIConvTranspose>},
    {schema::PrimitiveType_Equal, GetNNAPIOp<NNAPICommon>},
    {schema::PrimitiveType_ExpFusion, GetNNAPIOp<NNAPICommon>},
    {schema::PrimitiveType_Floor, GetNNAPIOp<NNAPICommon>},
    {schema::PrimitiveType_FullConnection, GetNNAPIOp<NNAPIFullConnection>},
    {schema::PrimitiveType_Gather, GetNNAPIOp<NNAPIGather>},
    {schema::PrimitiveType_InstanceNorm, GetNNAPIOp<NNAPIInstanceNorm>},
    {schema::PrimitiveType_AvgPoolFusion, GetNNAPIOp<NNAPIPooling>},
    {schema::PrimitiveType_MaxPoolFusion, GetNNAPIOp<NNAPIPooling>},
    {schema::PrimitiveType_PadFusion, GetNNAPIOp<NNAPIPadding>},
    {schema::PrimitiveType_Reshape, GetNNAPIOp<NNAPIReshape>},
    {schema::PrimitiveType_Resize, GetNNAPIOp<NNAPIResize>},
    {schema::PrimitiveType_ReduceFusion, GetNNAPIOp<NNAPIReduce>},
    {schema::PrimitiveType_Rsqrt, GetNNAPIOp<NNAPICommon>},
    {schema::PrimitiveType_Softmax, GetNNAPIOp<NNAPISoftmax>},
    {schema::PrimitiveType_Split, GetNNAPIOp<NNAPISplit>},
    {schema::PrimitiveType_Stack, GetNNAPIOp<NNAPIStack>},
    {schema::PrimitiveType_Transpose, GetNNAPIOp<NNAPITranspose>},
    {schema::PrimitiveType_ScaleFusion, GetNNAPIOp<NNAPIScale>},
    {schema::PrimitiveType_StridedSlice, GetNNAPIOp<NNAPIStridedSlice>},
    {schema::PrimitiveType_TopKFusion, GetNNAPIOp<NNAPITopk>},
  };
  return mindspore::kSuccess;
}

Status NNAPIDelegate::Build(DelegateModel<schema::Primitive> *model) {
  std::vector<NNAPIOp *> condidate_ops;
  auto begin_iter = model->BeginKernelIterator();
  for (auto iter = begin_iter; iter != model->EndKernelIterator(); iter++) {
    auto kernel = *iter;
    auto primitive = model->GetPrimitive(kernel);
    MS_ASSERT(primitive != nullptr);
    auto prim_type = primitive->value_type();
    if (op_func_lists_.find(prim_type) == op_func_lists_.end()) {
      MS_LOG(WARNING) << "Unsupported to get NNAPI Op with type of " << prim_type;
      remained_kernels_.push_back(kernel);
      continue;
    }
    auto get_op_func = op_func_lists_.at(prim_type);
    auto nnapi_op = get_op_func(kernel->name(), primitive, kernel->inputs(), kernel->outputs(), kernel->quant_type());
    if (nnapi_op == nullptr) {
      MS_LOG(WARNING) << "Get NNAPI op failed for " << prim_type;
      remained_kernels_.push_back(kernel);
      continue;
    }
    condidate_ops.push_back(nnapi_op);
  }
  if (condidate_ops.empty()) {
    return mindspore::kSuccess;
  }

  inputs_ = model->inputs();
  std::vector<kernel::Kernel *> ready_kenrels;
  auto ret = FindReadyKernels<kernel::Kernel>(&remained_kernels_, &ready_kenrels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FindReadyKernels failed.";
    for (auto op : condidate_ops) {
      delete op;
      op = nullptr;
    }
    return mindspore::kLiteError;
  }
  sorted_kernels_.insert(sorted_kernels_.end(), ready_kenrels.begin(), ready_kenrels.end());
  ready_kenrels.clear();

  // for every op, find pre and next ops
  FindPreNextOps<NNAPIOp>(condidate_ops);
  while (!condidate_ops.empty()) {
    auto nnapi_kernel = CreateNNAPISubGraph(model, &condidate_ops);
    if (nnapi_kernel != nullptr) {
      sorted_kernels_.push_back(reinterpret_cast<kernel::Kernel *>(nnapi_kernel));
      nnapi_kernels_.push_back(nnapi_kernel);
    } else {
      MS_LOG(WARNING) << "Create NPU Graph failed.";
      for (auto nnapi_op : condidate_ops) {
        delete nnapi_op;
        nnapi_op = nullptr;
      }
      return mindspore::kLiteError;
    }

    ret = FindReadyKernels<kernel::Kernel>(&remained_kernels_, &ready_kenrels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "FindReadyKernels failed.";
      for (auto nnapi_op : condidate_ops) {
        delete nnapi_op;
      }
      return mindspore::kLiteError;
    }
    sorted_kernels_.insert(sorted_kernels_.end(), ready_kenrels.begin(), ready_kenrels.end());
    ready_kenrels.clear();
  }
  if (!remained_kernels_.empty() || sorted_kernels_.empty()) {
    MS_LOG(ERROR) << "NNAPI delegate build failed.";
    return mindspore::kLiteError;
  }
  for (auto nnapi_kernel : nnapi_kernels_) {
    ret = nnapi_kernel->CompileNNAPIModel();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Compile nnapi model failed.";
      return mindspore::kLiteError;
    }
  }

  // Update the kernels of delegate model.
  ReplaceNodes(std::shared_ptr<LiteDelegateGraph>(model));
  return mindspore::kSuccess;
}

void NNAPIDelegate::ReplaceNodes(const std::shared_ptr<LiteDelegateGraph> &graph) {
  MS_ASSERT(graph != nullptr);
  auto nodes = graph->nodes();
  nodes->erase(nodes->begin(), nodes->end());
  nodes->insert(nodes->begin(), sorted_kernels_.begin(), sorted_kernels_.end());
}

NNAPISubGraph *NNAPIDelegate::CreateNNAPISubGraph(DelegateModel<schema::Primitive> *model,
                                                  std::vector<NNAPIOp *> *condidate_ops) {
  // find kernels that in the same subgraph
  std::vector<NNAPIOp *> chosen_ops;
  auto ret = FindReadyKernels<NNAPIOp>(condidate_ops, &chosen_ops);
  if (ret != RET_OK || chosen_ops.empty()) {
    MS_LOG(ERROR) << "Find ready NNAPI ops failed.";
    return nullptr;
  }
  // find inputs and outputs
  auto inputs = GetGraphInTensors<NNAPIOp>(chosen_ops, nullptr);
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Find inputs of subgraph failed.";
    return nullptr;
  }
  auto outputs = GetGraphOutTensors<NNAPIOp>(chosen_ops);
  // find the output tensor which is an input of other kernel.
  for (auto nnapi_op : chosen_ops) {
    for (auto kernel : remained_kernels_) {
      std::for_each(kernel->inputs().begin(), kernel->inputs().end(), [&nnapi_op, &outputs](const MSTensor &tensor) {
        if (std::find(outputs.begin(), outputs.end(), tensor) == outputs.end() &&
            std::find(nnapi_op->outputs().begin(), nnapi_op->outputs().end(), tensor) != nnapi_op->outputs().end()) {
          outputs.push_back(tensor);
        }
      });
    }
  }
  if (outputs.empty()) {
    MS_LOG(ERROR) << "Find outputs of subgraph failed.";
    return nullptr;
  }

  auto nnapi_kernel = new (std::nothrow) NNAPISubGraph(chosen_ops, inputs, outputs, devices_, relax_fp32_to_fp16_);
  if (nnapi_kernel == nullptr) {
    MS_LOG(ERROR) << "new NNAPI subgraph kernel failed.";
    return nullptr;
  }
  ret = nnapi_kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init NNAPI model failed.";
    delete nnapi_kernel;
    return nullptr;
  }
  ret = nnapi_kernel->CreateNNAPIModel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Create NNAPI model failed.";
    delete nnapi_kernel;
    return nullptr;
  }
  return nnapi_kernel;
}
}  // namespace lite
}  // namespace mindspore
