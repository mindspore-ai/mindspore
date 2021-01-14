/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include <memory>
#include <set>
#include <vector>
#include <algorithm>
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "src/kernel_registry.h"
#include "src/inner_context.h"
#include "src/ops/primitive_c.h"
#include "src/tensor.h"
#include "src/ops/populate/populate_register.h"

using mindspore::lite::KernelRegistry;
using mindspore::lite::PrimitiveC;
using mindspore::lite::Tensor;
namespace mindspore::opt {
namespace {
std::vector<Tensor *> GetCNodeInputTensors(const CNodePtr &CNode) {
  MS_ASSERT(CNode != nullptr);
  auto tmp_meta_graph = std::make_unique<schema::MetaGraphT>();
  auto tmp_fb_node = std::make_unique<schema::CNodeT>();
  lite::AnfExporter anfExporter;
  anfExporter.SetOpInputNode(CNode, tmp_meta_graph, tmp_fb_node.get());
  std::vector<Tensor *> input_tensors;
  for (auto input_index : tmp_fb_node->inputIndex) {
    auto tensorT = tmp_meta_graph->allTensors.at(input_index).get();
    auto tensor_shape = tensorT->dims;
    auto lite_tensor = new (std::nothrow) Tensor(
      TypeId(tensorT->dataType), tensor_shape, tensorT->format,
      lite::TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
    if (lite_tensor == nullptr) {
      MS_LOG(ERROR) << "lite tensor is nullptr";
      return input_tensors;
    }
    auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
    // when tensorT as graph input
    if (lite_tensor_size <= 0) {
      delete lite_tensor;
      return input_tensors;
    }
    auto tensor_data = new (std::nothrow) uint8_t[lite_tensor_size / sizeof(char)];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      delete lite_tensor;
      return input_tensors;
    }
    auto ret = memcpy_s(tensor_data, lite_tensor_size, tensorT->data.data(), lite_tensor_size);
    if (ret != EOK) {
      delete lite_tensor;
      delete[](tensor_data);
      MS_LOG(ERROR) << "memcpy error: " << ret;
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return {};
    }
    lite_tensor->set_data(tensor_data);
    input_tensors.emplace_back(lite_tensor);
  }
  return input_tensors;
}

ParameterPtr CreateNewParamter(const FuncGraphPtr &func_graph, Tensor *tensor) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor != nullptr);
  auto parameter = func_graph->add_parameter();
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector;
  (void)std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                       [](const int32_t &value) { return static_cast<int64_t>(value); });
  auto type_id = static_cast<TypeId>(tensor->data_type());
  auto type_ptr = TypeIdToType(type_id);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(type_id);
  param_value->set_format(tensor->format());
  if (tensor->MutableData() != nullptr) {
    auto size = tensor->Size();
    auto tensor_data = new (std::nothrow) uint8_t[size];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      return nullptr;
    }
    auto ret = memcpy_s(tensor_data, size, tensor->MutableData(), tensor->Size());
    if (ret != EOK) {
      delete[] tensor_data;
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return nullptr;
    }
    param_value->SetTensorData(tensor_data, size);
  }
  parameter->set_default_param(param_value);
  return parameter;
}
kernel::LiteKernel *GetLiteKernel(std::vector<Tensor *> inputs, const std::vector<Tensor *> &outputs,
                                  OpParameter *parameter, lite::InnerContext *context,
                                  mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(nullptr != lite_primitive);
  auto data_type = inputs.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, (schema::PrimitiveType)primitive->Type()};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  if (creator != nullptr) {
    auto lite_kernel = creator(inputs, outputs, parameter, context, desc, primitive);
    return lite_kernel;
  }
  return nullptr;
}

lite::STATUS ReplaceCNode(const FuncGraphPtr &func_graph, const CNodePtr &any_node, const AnfNodePtr &input_node,
                          std::vector<Tensor *> output_tensors, size_t replace_index) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  if (output_tensors.size() != 1) {
    for (size_t k = 0; k < output_tensors.size(); k++) {
      auto used_node_list = GetRealNodeUsedListByOutputIdx(func_graph, input_node, k);
      if (used_node_list->empty()) {
        MS_LOG(DEBUG) << "this output don't be used by other node.";
        continue;
      }
      if (used_node_list->size() != 1) {
        MS_LOG(ERROR) << " output must tuple_getitem";
        return lite::RET_ERROR;
      }
      auto tuple_node = used_node_list->at(0).first;
      if (GetCNodeType(tuple_node) == schema::PrimitiveType_TupleGetItem) {
        auto new_parameter = CreateNewParamter(func_graph, output_tensors.at(k));
        if (new_parameter == nullptr) {
          MS_LOG(ERROR) << "CreateNewParamter failed, name: " << input_node->fullname_with_scope();
          return lite::RET_ERROR;
        }
        new_parameter->set_name(input_node->fullname_with_scope() + "_const_" + std::to_string(k));
        manager->Replace(tuple_node, new_parameter);
      } else {
        MS_LOG(ERROR) << " multi out tensor must connect tuple-getitem: " << input_node->fullname_with_scope();
        return lite::RET_ERROR;
      }
    }
  } else {
    auto new_parameter = CreateNewParamter(func_graph, output_tensors.front());
    if (new_parameter == nullptr) {
      MS_LOG(ERROR) << "CreateNewParamter failed, name: " << input_node->fullname_with_scope();
      return lite::RET_ERROR;
    }
    new_parameter->set_name("constfold_" + input_node->fullname_with_scope());
    manager->Replace(input_node, new_parameter);
  }
  return lite::RET_OK;
}
}  //  namespace
void FreeTensors(std::vector<Tensor *> *input_tensor, std::vector<Tensor *> *output_tensor) {
  if (input_tensor != nullptr) {
    for (auto &i : *input_tensor) {
      delete i;
      i = nullptr;
    }
  }
  if (output_tensor != nullptr) {
    for (auto &i : *output_tensor) {
      delete i;
      i = nullptr;
    }
  }
}

const AnfNodePtr ConstFoldPass::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK ||
      !node->isa<CNode>()) {
    return nullptr;
  }
  auto any_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(any_node) != lite::RET_OK) {
    return nullptr;
  }
  bool changed = false;
  for (size_t i = 1; i < any_node->inputs().size(); i++) {
    auto input_node = any_node->input(i);
    if (!input_node->isa<CNode>() || !CheckIsAllInputsParam(input_node)) {
      continue;
    }
    auto input_cnode = input_node->cast<CNodePtr>();
    auto input_tensors = GetCNodeInputTensors(input_cnode);
    if (input_tensors.empty() || input_tensors.size() != input_cnode->inputs().size() - 1) {
      FreeTensors(&input_tensors, nullptr);
      continue;
    }
    changed = true;
    auto output_nums = GetOutputTensorNum(input_cnode);
    std::vector<Tensor *> output_tensors;
    for (size_t j = 0; j < output_nums; j++) {
      output_tensors.push_back(new (std::nothrow) Tensor());
    }
    auto lite_primitive = GetValueNode<std::shared_ptr<PrimitiveC>>(input_cnode->input(0));
    if (lite_primitive == nullptr) {
      MS_LOG(ERROR) << "lite_primitive is nullptr";
      FreeTensors(&input_tensors, &output_tensors);
      return nullptr;
    }
    auto inputQuantParams = lite_primitive->input_quant_params();
    for (size_t m = 0; m < inputQuantParams.size(); m++) {
      for (auto inputQuantParam : inputQuantParams[m]) {
        lite::QuantArg quant_arg{};
        quant_arg.scale = inputQuantParam.scale;
        quant_arg.zeroPoint = inputQuantParam.zeroPoint;
        input_tensors[m]->AddQuantParam(quant_arg);
      }
    }
    auto outputQuantParams = lite_primitive->output_quant_params();
    for (size_t m = 0; m < outputQuantParams.size(); m++) {
      for (auto outputQuantParam : outputQuantParams[m]) {
        lite::QuantArg quant_arg{};
        quant_arg.scale = outputQuantParam.scale;
        quant_arg.zeroPoint = outputQuantParam.zeroPoint;
        output_tensors[m]->AddQuantParam(quant_arg);
      }
    }
    lite_primitive->InferShape(input_tensors, output_tensors);
    auto primitive = lite_primitive.get();
    if (primitive->Type() == schema::PrimitiveType_RandomStandardNormal) {
      return nullptr;
    }
    MS_ASSERT(primitive != nullptr);
    MS_ASSERT(primitive->Type() != nullptr);
    auto func_pointer =
      lite::PopulateRegistry::GetInstance()->GetParameterCreator(schema::PrimitiveType(primitive->Type()));
    if (func_pointer == nullptr) {
      MS_LOG(ERROR) << "ParameterCreator function pointer is nullptr, type: "
                    << schema::EnumNamePrimitiveType((schema::PrimitiveType)primitive->Type());
      return nullptr;
    }
    auto parameter = func_pointer(primitive);

    if (parameter == nullptr) {
      MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                    << schema::EnumNamePrimitiveType((schema::PrimitiveType)(lite_primitive->Type()));
      return nullptr;
    }
    auto lite_kernel = GetLiteKernel(input_tensors, output_tensors, parameter, context.get(), lite_primitive.get());
    if (lite_kernel == nullptr) {
      MS_LOG(ERROR) << "constant_folding schedule node lite kernel nullptr";
      FreeTensors(&input_tensors, &output_tensors);
      return nullptr;
    }
    for (auto output_tensor : output_tensors) {
      auto ret = output_tensor->MallocData();
      if (RET_OK != ret) {
        MS_LOG(ERROR) << "MallocData failed";
        FreeTensors(&input_tensors, &output_tensors);
        return nullptr;
      }
    }
    auto ret = lite_kernel->Run();
    if (0 != ret) {
      FreeTensors(&input_tensors, &output_tensors);
      MS_LOG(ERROR) << "run kernel failed, name: " << lite_kernel->name();
      return nullptr;
    }
    // replace cnode by new param
    if (ReplaceCNode(func_graph, any_node, input_node, output_tensors, i) != lite::RET_OK) {
      FreeTensors(&input_tensors, &output_tensors);
      delete (lite_kernel);
      MS_LOG(ERROR) << "constant_folding replace cnode failed";
      return nullptr;
    }
    MS_LOG(DEBUG) << "fold node:" << input_node->fullname_with_scope() << " success ";
    FreeTensors(&input_tensors, &output_tensors);
    delete (lite_kernel);
  }
  return changed ? any_node : nullptr;
}
}  // namespace mindspore::opt
