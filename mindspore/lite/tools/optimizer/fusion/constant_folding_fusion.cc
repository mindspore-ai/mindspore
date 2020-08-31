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
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/anf_exporter/anf_exporter.h"
#include "src/kernel_registry.h"
#include "include/context.h"
#include "src/populate_parameter.h"
#include "src/ops/primitive_c.h"

using mindspore::lite::KernelRegistry;
using mindspore::lite::PrimitiveC;
using mindspore::lite::tensor::Tensor;
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
    auto lite_tensor =
      new (std::nothrow) Tensor(TypeId(tensorT->dataType), tensor_shape, tensorT->format, tensorT->nodeType);
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
      MS_LOG(EXCEPTION) << "memcpy error: " << ret;
    }
    lite_tensor->SetData(tensor_data);
    input_tensors.emplace_back(lite_tensor);
  }
  return input_tensors;
}

ParameterPtr CreateNewParamter(const FuncGraphPtr &func_graph, Tensor *tensor) {
  auto parameter = func_graph->add_parameter();
  std::vector<int> shape(tensor->shape());
  auto type_id = static_cast<TypeId>(tensor->data_type());
  auto type_ptr = TypeIdToType(type_id);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
  parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(type_id);
  param_value->set_format(tensor->GetFormat());
  if (tensor->Data() != nullptr) {
    auto size = tensor->ElementsNum();
    auto tensor_data = new (std::nothrow) float[size];
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      return nullptr;
    }
    auto ret = memcpy_s(tensor_data, size * sizeof(float), tensor->Data(), size * sizeof(float));
    if (ret != EOK) {
      delete[] tensor_data;
      MS_LOG(ERROR) << "memcpy error: " << ret;
      return nullptr;
    }
    param_value->set_tensor_addr(tensor_data);
    param_value->set_tensor_size(size * sizeof(float) / sizeof(uint8_t));
  }
  parameter->set_default_param(param_value);
  return parameter;
}
kernel::LiteKernel *GetLiteKernel(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs, OpParameter *parameter,
                                  mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(nullptr != lite_primitive);
  auto data_type = inputs.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, (schema::PrimitiveType)primitive->Type()};
  lite::Context context;
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  if (creator != nullptr) {
    auto lite_kernel = creator(inputs, outputs, parameter, &context, desc, primitive);
    return lite_kernel;
  }
  return nullptr;
}
}  //  namespace
void FreeTensors(std::vector<Tensor *> *input_tensor, std::vector<Tensor *> *output_tensor) {
  if (input_tensor != nullptr) {
    for (size_t i = 0; i < input_tensor->size(); i++) {
      delete (*input_tensor)[i];
      (*input_tensor)[i] = nullptr;
    }
  }
  if (output_tensor != nullptr) {
    for (size_t i = 0; i < output_tensor->size(); i++) {
      delete (*output_tensor)[i];
      (*output_tensor)[i] = nullptr;
    }
  }
}

const AnfNodePtr ConstFoldPass::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  CheckIfFuncGraphIsNull(func_graph);
  CheckIfAnfNodeIsNull(node);
  if (!node->isa<CNode>()) {
    return node;
  }
  auto any_node = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(any_node);
  for (size_t i = 1; i < any_node->inputs().size(); i++) {
    auto input_node = any_node->input(i);
    if (input_node->isa<CNode>() && CheckIsAllInputsParam(input_node)) {
      auto input_cnode = input_node->cast<CNodePtr>();
      auto input_tensors = GetCNodeInputTensors(input_cnode);
      if (input_tensors.empty() || input_tensors.size() != input_cnode->inputs().size() - 1) {
        FreeTensors(&input_tensors, nullptr);
        continue;
      }
      MS_LOG(INFO) << "Begin fold node:" << input_node->fullname_with_scope();
      auto output_nums = GetOutputTensorNum(input_cnode);
      std::vector<Tensor *> output_tensors{output_nums, new Tensor()};
      auto lite_primitive = GetValueNode<std::shared_ptr<PrimitiveC>>(input_cnode->input(0));
      if (lite_primitive == nullptr) {
        MS_LOG(ERROR) << "lite_primitive is nullptr";
        FreeTensors(&input_tensors, &output_tensors);
        return nullptr;
      }
      // here, input_tensor's format need to be transposed nhwc according to fmkType,
      // but for the time being, we only transpose the tensor with 0/1/2/3D.
      // Others should be added in future.
      for (size_t j = 0; j < input_tensors.size(); ++j) {
        input_tensors[j]->SetFormat(schema::Format_NHWC);
        if (input_tensors[j]->shape().size() == 4) {
          MS_LOG(INFO) << "init input_tensor format to nhwc";
        }
      }
      lite_primitive->InferShape(input_tensors, output_tensors);
      auto parameter = kernel::PopulateParameter(lite_primitive.get());
      if (parameter == nullptr) {
        MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                      << schema::EnumNamePrimitiveType((schema::PrimitiveType)(lite_primitive->Type()));
        return nullptr;
      }
      auto lite_kernel = GetLiteKernel(input_tensors, output_tensors, parameter, lite_primitive.get());
      if (lite_kernel == nullptr) {
        MS_LOG(ERROR) << "constant_folding schedule node lite kernel nullptr";
        FreeTensors(&input_tensors, &output_tensors);
        return nullptr;
      }
      auto ret = lite_kernel->Run();
      if (0 != ret) {
        FreeTensors(&input_tensors, &output_tensors);
        MS_LOG(ERROR) << "run kernel failed, name: " << lite_kernel->name();
        return nullptr;
      }
      auto new_parameter = CreateNewParamter(func_graph, output_tensors.front());
      if (new_parameter == nullptr) {
        FreeTensors(&input_tensors, &output_tensors);
        MS_LOG(ERROR) << "CreateNewParamter failed, name: " << lite_kernel->name();
        return nullptr;
      }
      new_parameter->set_name(input_node->fullname_with_scope());
      any_node->set_input(i, new_parameter);
      FreeTensors(&input_tensors, &output_tensors);
      delete (lite_kernel);
    }
  }
  return any_node;
}
}  // namespace mindspore::opt
