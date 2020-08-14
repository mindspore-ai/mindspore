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
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/kernel_factory.h"
#include "src/common/anf_exporter/anf_exporter.h"
#include "src/scheduler.h"
#include "include/context.h"
#include "src/lite_session.h"
#include "src/ir/primitive_t_value.h"
#include "src/populate_parameter.h"

using mindspore::lite::KernelFactory;
using mindspore::lite::tensor::Tensor;
using mindspore::lite::PrimitiveTValue;
namespace mindspore::opt {
namespace {
const std::vector<Tensor *> GetCNodeInputTensors(const CNodePtr &CNode) {
  MS_ASSERT(CNode != nullptr);
  auto tmp_meta_graph = std::make_unique<schema::MetaGraphT>();
  auto tmp_fb_node = std::make_unique<schema::CNodeT>();
  lite::AnfExporter anfExporter;
  anfExporter.SetOpInputNode(CNode, tmp_meta_graph.get(), tmp_fb_node.get());
  std::vector<Tensor *> input_tensors;
  for (auto input_index : tmp_fb_node->inputIndex) {
    auto tensorT = tmp_meta_graph->allTensors.at(input_index).get();
    auto tensor_shape = tensorT->dims;
    auto lite_tensor =
        new(std::nothrow)Tensor(TypeId(tensorT->dataType), tensor_shape, tensorT->format, tensorT->nodeType);
    auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
    // when tensorT as graph input
    if (lite_tensor_size == 0) {
        return input_tensors;
    }
    auto tensor_data = new(std::nothrow)char[lite_tensor_size / sizeof(char)];
    auto ret = memcpy_s(tensor_data, lite_tensor_size, tensorT->data.data(), lite_tensor_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy error: " << ret;
    }
    lite_tensor->SetData(tensor_data);
    input_tensors.emplace_back(lite_tensor);
  }
  return input_tensors;
}
schema::Primitive *PackPrimitiveT(const CNodePtr &cnode) {
  auto primitiveT_value =
      GetValueNode<std::shared_ptr<PrimitiveTValue>>(cnode->input(0));
  if (primitiveT_value == nullptr) {
    MS_LOG(ERROR) << "PrimitiveT_value is nullptr";
    return nullptr;
  }

  auto *lite_primitive = primitiveT_value->GetPrimitiveT();
  if (lite_primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive in primitiveT_value is nullptr";
    return nullptr;
  }

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::Primitive::Pack(builder, lite_primitive);
  builder.Finish(offset);
  auto buf = builder.GetBufferPointer();
  auto primitive = flatbuffers::GetRoot<schema::Primitive>(buf);
  return const_cast<schema::Primitive *>(primitive);
}
const ParameterPtr CreateNewParamter(const FuncGraphPtr &func_graph, Tensor *tensor) {
  auto parameter = func_graph->add_parameter();
  std::vector<int> shape;
  std::copy(tensor->shape().begin(), tensor->shape().end(), std::back_inserter(shape));
  auto type_id = static_cast<TypeId>(tensor->data_type());
  auto type_ptr = TypeIdToType(type_id);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
  parameter->set_abstract(abstract_tensor);

  ParamValueLitePtr param_value = std::make_shared<ParamValueLite>();
  MS_ASSERT(param_value != nullptr);
  param_value->set_tensor_shape(shape);
  param_value->set_tensor_type(type_id);
  if (tensor->Data() != nullptr) {
    auto size = tensor->ElementsNum();
    auto tensor_data = new (std::nothrow) float[size];
    auto ret = memcpy_s(tensor_data, size * sizeof(float), tensor->Data(), size * sizeof(float));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy error: " << ret;
    }
    param_value->set_tensor_addr(tensor_data);
    param_value->set_tensor_size(size * sizeof(float) / sizeof(uint8_t));
  }
  parameter->set_default_param(param_value);
  return parameter;
}
kernel::LiteKernel *GetLiteKernel(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs,
                                  lite::Primitive *primitive) {
  MS_ASSERT(nullptr != lite_primitive);
  auto data_type = inputs.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, primitive->Type()};
  lite::Context context;
  auto parameter = kernel::PopulateParameter(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR)
            << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(primitive->Type());
    return nullptr;
  }
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  if (creator != nullptr) {
    auto lite_kernel = creator(inputs, outputs, parameter, &context, desc, primitive);
    return lite_kernel;
  }
  return nullptr;
}
}  //  namespace

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
          return any_node;
      }
      MS_LOG(INFO) << "Begin fold node:" << input_node->fullname_with_scope();
      auto output_nums = GetOutputTensorNum(input_cnode);
      std::vector<Tensor *> output_tensors{output_nums, new Tensor()};
      auto scheam_primitive = PackPrimitiveT(input_cnode);
      auto lite_primitive = lite::Primitive::CreatePrimitive(scheam_primitive);
      lite_primitive->InferShape(input_tensors, output_tensors);
      auto lite_kernel = GetLiteKernel(input_tensors, output_tensors, lite_primitive);
      if (lite_kernel == nullptr) {
        MS_LOG(ERROR) << "constant_folding schedule node lite kernel nullptr";
        return any_node;
      }
      auto ret = lite_kernel->Run();
      if (0 != ret) {
        MS_LOG(EXCEPTION) << "run kernel failed, name: " << lite_kernel->name();
      }
      auto new_parameter = CreateNewParamter(func_graph, output_tensors.front());
      any_node->set_input(i, new_parameter);
    }
  }
  return any_node;
}
}  // namespace mindspore::opt
