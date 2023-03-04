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

#define USE_DEPRECATED_API
#include "tools/optimizer/const_fold/fold_utils.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/common/optimizer/helper.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/context_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/inner_context.h"
#include "src/tensor.h"
#include "src/common/ops/anf_utils.h"
#include "src/litert/infer_manager.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "tools/optimizer/common/helper.h"

using mindspore::lite::KernelRegistry;
using mindspore::lite::Tensor;
namespace mindspore {
namespace opt {
namespace {
bool IsInferInRunning(const CNodePtr &cnode) { return CheckPrimitiveType(cnode, prim::kPrimWhere); }
ParameterPtr CreateNewParamter(const FuncGraphPtr &func_graph, Tensor *tensor) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor != nullptr);
  auto parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(parameter != nullptr, nullptr);
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });

  auto tensor_info = std::make_shared<tensor::Tensor>(tensor->data_type(), shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return nullptr;
  }
  if (tensor->MutableData() != nullptr && tensor->ElementsNum() != 0) {
    auto tensor_data = static_cast<uint8_t *>(tensor_info->data_c());
    auto ret = memcpy_s(tensor_data, tensor_info->Size(), tensor->data(), tensor->Size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return nullptr;
    }
  }
  auto status = lite::InitParameterFromTensorInfo(parameter, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed";
    return nullptr;
  }
  return parameter;
}
kernel::KernelExec *GetKernelExec(std::vector<Tensor *> inputs, std::vector<Tensor *> *outputs, const CNodePtr &cnode,
                                  lite::InnerContext *context, mindspore::Context *ms_context) {
  MS_ASSERT(outputs != nullptr && cnode != nullptr && context != nullptr && ms_context != nullptr);
  OpParameter *parameter = nullptr;
  auto ret = lite::FetchOpParameterFromNode(cnode->input(0), &parameter);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " FetchOpParameterFromNode failed. ";
    return nullptr;
  }
  parameter->thread_num_ = 1;
  ret = KernelInferShape(inputs, *outputs, parameter);
  if (ret != lite::RET_OK && (ret != lite::RET_INFER_INVALID || !IsInferInRunning(cnode))) {
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " infershape failed!type: ";
    return nullptr;
  }
  auto data_type = inputs.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, NHWC,
                         static_cast<schema::PrimitiveType>(parameter->type_)};
  kernel::KernelExec *kernel_exec = nullptr;
  ret = lite::KernelRegistry::GetInstance()->GetKernelExec(inputs, *outputs, context, ms_context, desc, parameter,
                                                           &kernel_exec);
  if (ret != lite::RET_OK) {
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    return nullptr;
  }
  ret = kernel_exec->Prepare();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "init failed.";
    delete kernel_exec;  // parameter will be freed in destructor of lite-kernel.
    return nullptr;
  }
  return kernel_exec;
}

lite::STATUS ReplaceCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, std::vector<Tensor *> output_tensors) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, lite::RET_NULL_PTR);
  if (output_tensors.size() != 1) {
    for (size_t k = 0; k < output_tensors.size(); k++) {
      auto used_node_list = Helper::GetRealNodeUsedListByOutputIdx(func_graph, cnode, k);
      if (used_node_list->empty()) {
        MS_LOG(DEBUG) << "this output don't be used by other node.";
        continue;
      }
      if (used_node_list->size() != 1) {
        MS_LOG(ERROR) << " output must tuple_getitem";
        return lite::RET_ERROR;
      }
      auto tuple_node = used_node_list->at(0).first;
      if (CheckPrimitiveType(tuple_node, prim::kPrimTupleGetItem)) {
        auto new_parameter = CreateNewParamter(func_graph, output_tensors.at(k));
        if (new_parameter == nullptr) {
          MS_LOG(ERROR) << "CreateNewParamter failed, name: " << cnode->fullname_with_scope();
          return lite::RET_ERROR;
        }
        new_parameter->set_name(cnode->fullname_with_scope() + "_const_" + std::to_string(k));
        (void)manager->Replace(tuple_node, new_parameter);
      } else {
        MS_LOG(ERROR) << " multi out tensor must connect tuple-getitem: " << cnode->fullname_with_scope();
        return lite::RET_ERROR;
      }
    }
  } else {
    auto new_parameter = CreateNewParamter(func_graph, output_tensors.front());
    if (new_parameter == nullptr) {
      MS_LOG(ERROR) << "CreateNewParamter failed, name: " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    new_parameter->set_name("constfold_" + cnode->fullname_with_scope());
    (void)manager->Replace(cnode, new_parameter);
  }
  return lite::RET_OK;
}

lite::STATUS CopyQuantParams(const CNodePtr &cnode, const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) {
  MS_ASSERT(cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_RET(prim != nullptr, lite::RET_ERROR);
  auto quant_tensor_info_ptr = prim->GetAttr("quant_params");
  if (quant_tensor_info_ptr == nullptr) {
    return lite::RET_OK;
  }
  auto quant_param_holder = quant_tensor_info_ptr->cast<lite::QuantParamHolderPtr>();
  if (quant_param_holder == nullptr) {
    MS_LOG(ERROR) << "quant param is invalid.";
    return lite::RET_ERROR;
  }
  auto input_quant_params = quant_param_holder->get_input_quant_params();
  // unmod node may make input size diff input quant params size
  auto input_size = inputs.size() > input_quant_params.size() ? input_quant_params.size() : inputs.size();
  for (size_t m = 0; m < input_size; m++) {
    for (auto inputQuantParam : input_quant_params[m]) {
      lite::LiteQuantParam quant_arg{};
      quant_arg.scale = inputQuantParam.scale;
      quant_arg.zeroPoint = inputQuantParam.zeroPoint;
      quant_arg.roundType = inputQuantParam.roundType;
      quant_arg.multiplier = inputQuantParam.multiplier;
      inputs[m]->AddQuantParam(quant_arg);
    }
  }
  auto output_quant_params = quant_param_holder->get_output_quant_params();
  for (size_t m = 0; m < output_quant_params.size(); m++) {
    for (auto outputQuantParam : output_quant_params[m]) {
      lite::LiteQuantParam quant_arg{};
      quant_arg.scale = outputQuantParam.scale;
      quant_arg.zeroPoint = outputQuantParam.zeroPoint;
      quant_arg.roundType = outputQuantParam.roundType;
      quant_arg.multiplier = outputQuantParam.multiplier;
      outputs[m]->AddQuantParam(quant_arg);
    }
  }
  return lite::RET_OK;
}
}  //  namespace

bool ConstFoldProcessor::Init() {
  if (context_ == nullptr) {
    context_ = std::make_shared<lite::InnerContext>();
    MS_CHECK_TRUE_RET(context_ != nullptr, false);
    if (context_->Init() != RET_OK) {
      MS_LOG(ERROR) << "init context failed.";
      return false;
    }
  }
  if (ms_context_ == nullptr) {
    ms_context_ = std::shared_ptr<mindspore::Context>(lite::MSContextFromContext(context_.get()));
    MS_CHECK_TRUE_RET(ms_context_ != nullptr, false);
  }
  return true;
}

int ConstFoldProcessor::DoConstantFold(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (func_graph == nullptr || cnode == nullptr) {
    MS_LOG(ERROR) << "input param is nullptr.";
    return lite::RET_NULL_PTR;
  }
  if (!Init()) {
    MS_LOG(ERROR) << "initial context failed.";
    return lite::RET_ERROR;
  }
  std::vector<TensorPtr> inputs_ptr;
  if (LiteTensorExtractor::GetCNodeInputTensors(cnode, &inputs_ptr, fmk_type_, train_flag_, true) != lite::RET_OK) {
    MS_LOG(ERROR) << "extract input tensor from cnode failed. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (std::any_of(inputs_ptr.begin(), inputs_ptr.end(),
                  [](const TensorPtr &input) { return input->data_type() == kObjectTypeTensorType; })) {
    MS_LOG(DEBUG) << "this op is control flow op, which is not supported now.";
    return lite::RET_OK;
  }
  std::vector<TensorPtr> outputs_ptr;
  if (LiteTensorExtractor::GetCNodeOutputTensors(cnode, &outputs_ptr, train_flag_) != lite::RET_OK) {
    MS_LOG(ERROR) << "extract output tensor from cnode failed. " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (std::any_of(outputs_ptr.begin(), outputs_ptr.end(),
                  [](const TensorPtr &output) { return output->data_type() == kObjectTypeTensorType; })) {
    MS_LOG(DEBUG) << "this op is control flow op, which is not supported now.";
    return lite::RET_OK;
  }
  std::vector<lite::Tensor *> input_tensors;
  (void)std::transform(inputs_ptr.begin(), inputs_ptr.end(), std::back_inserter(input_tensors),
                       [](const TensorPtr &input) { return input.get(); });
  std::vector<lite::Tensor *> output_tensors;
  (void)std::transform(outputs_ptr.begin(), outputs_ptr.end(), std::back_inserter(output_tensors),
                       [](const TensorPtr &output) { return output.get(); });
  if (CopyQuantParams(cnode, input_tensors, output_tensors) != lite::RET_OK) {
    MS_LOG(ERROR) << "copy quant params failed.";
    return lite::RET_ERROR;
  }
  auto kernel_exec = GetKernelExec(input_tensors, &output_tensors, cnode, context_.get(), ms_context_.get());
  if (kernel_exec == nullptr) {
    MS_LOG(ERROR) << "constant_folding schedule node lite kernel nullptr, node " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  for (auto output_tensor : output_tensors) {
    auto status = output_tensor->MallocData();
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "MallocData failed";
      delete (kernel_exec);
      return lite::RET_ERROR;
    }
  }
  auto status = static_cast<mindspore::kernel::LiteKernel *>(kernel_exec->kernel())->Run();
  delete (kernel_exec);
  kernel_exec = nullptr;
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "run kernel failed, name: " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  // replace cnode by new param
  status = ReplaceCNode(func_graph, cnode, output_tensors);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "constant_folding replace cnode failed";
  } else {
    MS_LOG(DEBUG) << "fold node:" << cnode->fullname_with_scope() << " success ";
  }
  return status;
}
}  // namespace opt
}  // namespace mindspore
