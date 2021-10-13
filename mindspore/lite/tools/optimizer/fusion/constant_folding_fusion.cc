/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "tools/anf_exporter/fetch_content.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/converter/converter_context.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/common.h"
#include "src/ops/populate/populate_register.h"
#include "src/kernel_registry.h"
#include "src/inner_context.h"
#include "src/tensor.h"
#include "src/ops/ops_utils.h"
#include "src/runtime/infer_manager.h"

using mindspore::lite::KernelRegistry;
using mindspore::lite::Tensor;
namespace mindspore::opt {
namespace {
constexpr size_t INITIAL_SIZE = 1024;
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

std::vector<Tensor *> GetCNodeInputTensors(const CNodePtr &cnode, converter::FmkType fmk_type) {
  MS_ASSERT(cnode != nullptr);
  std::vector<Tensor *> tensors;
  for (size_t i = 1; i < cnode->size(); ++i) {
    int status = 0;
    lite::DataInfo data_info;
    if (lite::ConverterContext::GetInstance()->GetGraphInputTensorShapeMapSize() > 0 &&
        CheckPrimitiveType(cnode, prim::kPrimShape)) {
      if (utils::isa<abstract::AbstractTensorPtr>(cnode->input(i)->abstract())) {
        auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(cnode->input(i)->abstract());
        if (abstract_tensor == nullptr) {
          MS_LOG(ERROR) << "abstract tensor is nullptr.";
          return {};
        }
        auto shape = utils::cast<abstract::ShapePtr>(abstract_tensor->GetShapeTrack())->shape();
        bool is_dynamic_shape = false;
        for (size_t j = 0; j < shape.size(); j++) {
          auto dim = shape[j];
          if (dim == -1) {
            is_dynamic_shape = true;
            break;
          }
        }
        if (!is_dynamic_shape) {
          status = lite::FetchDataFromParameterNode(cnode, i, fmk_type, false, &data_info);
        }
      }
    } else if (utils::isa<ParameterPtr>(cnode->input(i))) {
      if (!cnode->input(i)->cast<ParameterPtr>()->has_default()) {
        FreeTensors(&tensors, nullptr);
        return {};
      }
      status = lite::FetchDataFromParameterNode(cnode, i, fmk_type, false, &data_info);
    } else if (utils::isa<ValueNodePtr>(cnode->input(i))) {
      status = lite::FetchDataFromValueNode(cnode, i, fmk_type, false, &data_info);
    } else {
      MS_LOG(ERROR) << "input node is not const node.";
      FreeTensors(&tensors, nullptr);
      return {};
    }
    if (status == lite::RET_NO_CHANGE) {
      continue;
    }
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "parser const data failed.";
      FreeTensors(&tensors, nullptr);
      return {};
    }
    if (data_info.shape_.empty() && data_info.data_.empty()) {
      FreeTensors(&tensors, nullptr);
      MS_LOG(DEBUG) << "input node is graph input.";
      return {};
    }
    auto tensor = new (std::nothrow)
      Tensor(TypeId(data_info.data_type_), data_info.shape_, static_cast<mindspore::Format>(data_info.format_),
             lite::TensorCategory(0, data_info.shape_.size(), TypeId(data_info.data_type_), data_info.data_.size()));
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "new a tensor is nullptr.";
      FreeTensors(&tensors, nullptr);
      return {};
    }
    if (data_info.data_.empty()) {
      tensors.emplace_back(tensor);
      continue;
    }
    auto tensor_data = tensor->MutableData();
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "malloc data failed.";
      FreeTensors(&tensors, nullptr);
      return {};
    }
    if (memcpy_s(tensor_data, tensor->Size(), data_info.data_.data(), data_info.data_.size()) != EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      FreeTensors(&tensors, nullptr);
      return {};
    }
    tensors.emplace_back(tensor);
  }
  return tensors;
}

ParameterPtr CreateNewParamter(const FuncGraphPtr &func_graph, Tensor *tensor) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(tensor != nullptr);
  auto parameter = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(parameter != nullptr, nullptr);
  std::vector<int> shape(tensor->shape());
  std::vector<int64_t> shape_vector;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vector),
                 [](const int32_t &value) { return static_cast<int64_t>(value); });

  auto tensor_info = std::make_shared<tensor::Tensor>(tensor->data_type(), shape_vector);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return nullptr;
  }
  if (tensor->MutableData() != nullptr) {
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
kernel::LiteKernel *GetLiteKernel(std::vector<Tensor *> inputs, std::vector<Tensor *> *outputs, const CNodePtr &cnode,
                                  lite::InnerContext *context, mindspore::Context *ms_context) {
  MS_ASSERT(outputs != nullptr && cnode != nullptr && context != nullptr && ms_context != nullptr);
  auto prim_t = lite::GetPrimitiveT(cnode->input(0));
  if (prim_t == nullptr) {
    return nullptr;
  }
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
  auto prim = lite::ConvertToPrimitive(prim_t.get(), &fbb);
  if (prim == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "get primitive failed.";
    return nullptr;
  }
  auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(prim->value_type(), lite::SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
    return nullptr;
  }
  auto parameter = parameter_gen(prim);
  fbb.Clear();
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }
  parameter->thread_num_ = 1;
  auto ret = KernelInferShape(inputs, *outputs, parameter);
  if (ret != lite::RET_OK) {
    free(parameter);
    MS_LOG(ERROR) << "infershape failed!type: " << schema::EnumNamePrimitiveType(prim->value_type());
    return nullptr;
  }
  auto data_type = inputs.front()->data_type();
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, static_cast<schema::PrimitiveType>(parameter->type_)};
  kernel::LiteKernel *lite_kernel;
  ret = lite::KernelRegistry::GetInstance()->GetKernel(inputs, *outputs, context, ms_context, desc, parameter,
                                                       &lite_kernel);
  if (ret != lite::RET_OK) {
    free(parameter);
    return nullptr;
  }
  ret = lite_kernel->Init();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "init failed.";
    free(parameter);
    return nullptr;
  }
  return lite_kernel;
}

lite::STATUS ReplaceCNode(const FuncGraphPtr &func_graph, const CNodePtr &any_node, const AnfNodePtr &input_node,
                          std::vector<Tensor *> output_tensors) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, lite::RET_NULL_PTR);
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
      if (CheckPrimitiveType(tuple_node, prim::kPrimTupleGetItem)) {
        auto new_parameter = CreateNewParamter(func_graph, output_tensors.at(k));
        if (new_parameter == nullptr) {
          MS_LOG(ERROR) << "CreateNewParamter failed, name: " << input_node->fullname_with_scope();
          return lite::RET_ERROR;
        }
        new_parameter->set_name(input_node->fullname_with_scope() + "_const_" + std::to_string(k));
        (void)manager->Replace(tuple_node, new_parameter);
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
    (void)manager->Replace(input_node, new_parameter);
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

bool ConstFoldPass::PreProcess() const {
  if (context_ == nullptr) {
    context_ = std::make_shared<lite::InnerContext>();
    MS_CHECK_TRUE_RET(context_ != nullptr, false);
    if (context_->Init() != RET_OK) {
      return false;
    }
  }
  if (ms_context_ == nullptr) {
    ms_context_ = std::shared_ptr<mindspore::Context>(lite::MSContextFromContext(context_.get()));
    MS_CHECK_TRUE_RET(ms_context_ != nullptr, false);
  }
  return true;
}

bool ConstFoldPass::CheckCanFusion(const AnfNodePtr &input_node) const {
  if (!input_node->isa<CNode>() || !CheckIsAllInputsParam(input_node)) {
    return false;
  }
  if (CheckPrimitiveType(input_node, prim::kPrimTupleGetItem) || CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
    return false;
  }
  auto input_cnode = input_node->cast<CNodePtr>();
  if (IsMarkedTrainOp(input_cnode)) {
    return false;
  }
  return true;
}

const AnfNodePtr ConstFoldPass::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  if (!PreProcess()) {
    MS_LOG(ERROR) << "run pre-process failed.";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(func_graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(node != nullptr, nullptr);
  auto any_node = node->cast<CNodePtr>();
  if (any_node == nullptr) {
    return nullptr;
  }
  bool changed = false;
  for (size_t i = 1; i < any_node->inputs().size(); i++) {
    auto input_node = any_node->input(i);
    if (!CheckCanFusion(input_node)) {
      continue;
    }
    auto input_cnode = input_node->cast<CNodePtr>();
    auto input_tensors = GetCNodeInputTensors(input_cnode, fmk_type_);
    if (input_tensors.empty()) {
      continue;
    }
    changed = true;
    auto output_nums = GetOutputTensorNum(input_cnode);
    std::vector<Tensor *> output_tensors;
    for (size_t j = 0; j < output_nums; j++) {
      auto out_tensor = new (std::nothrow) Tensor();
      if (out_tensor == nullptr) {
        MS_LOG(ERROR) << "new a tensor failed.";
        FreeTensors(&input_tensors, &output_tensors);
        return nullptr;
      }
      output_tensors.push_back(out_tensor);
    }
    if (CopyQuantParams(input_cnode, input_tensors, output_tensors) != lite::RET_OK) {
      MS_LOG(ERROR) << "copy quant params failed.";
      FreeTensors(&input_tensors, &output_tensors);
      return nullptr;
    }
    auto lite_kernel = GetLiteKernel(input_tensors, &output_tensors, input_cnode, context_.get(), ms_context_.get());
    if (lite_kernel == nullptr) {
      FreeTensors(&input_tensors, &output_tensors);
      MS_LOG(ERROR) << "constant_folding schedule node lite kernel nullptr";
      return nullptr;
    }
    for (auto output_tensor : output_tensors) {
      auto status = output_tensor->MallocData();
      if (status != lite::RET_OK) {
        MS_LOG(ERROR) << "MallocData failed";
        FreeTensors(&input_tensors, &output_tensors);
        delete (lite_kernel);
        return nullptr;
      }
    }
    auto status = static_cast<mindspore::kernel::InnerKernel *>(lite_kernel->kernel())->Run();
    delete (lite_kernel);
    lite_kernel = nullptr;
    if (status != lite::RET_OK) {
      FreeTensors(&input_tensors, &output_tensors);
      MS_LOG(ERROR) << "run kernel failed, name: " << input_node->fullname_with_scope();
      return nullptr;
    }
    // replace cnode by new param
    if (ReplaceCNode(func_graph, any_node, input_node, output_tensors) != lite::RET_OK) {
      FreeTensors(&input_tensors, &output_tensors);
      MS_LOG(ERROR) << "constant_folding replace cnode failed";
      return nullptr;
    }
    MS_LOG(DEBUG) << "fold node:" << input_node->fullname_with_scope() << " success ";
    FreeTensors(&input_tensors, &output_tensors);
  }
  return changed ? any_node : nullptr;
}
}  // namespace mindspore::opt
