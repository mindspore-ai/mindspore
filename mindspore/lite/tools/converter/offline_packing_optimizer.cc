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

#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include "tools/common/graph_util.h"
#include "tools/converter/offline_packing_optimizer.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "src/common/primitive_t_utils.h"
#include "src/common/ops/anf_utils.h"
#include "src/common/file_utils.h"
#include "nnacl/matmul_parameter.h"
#include "src/litert/kernel/cpu/int8/matmul_dynamic_base_int8.h"

using mindspore::kernel::MatmulDynamicBaseInt8CPUKernel;

namespace mindspore::lite {
namespace {
constexpr const int kPrimIndex = 0;
constexpr const int kSingleThread = 1;
const char kAndroidArmCpuBackendOption[] = "ANDROID_ARM_CPU";
}  // namespace

mindspore::lite::InnerContext *InitInnerContextForAndroidArmCpu() {
  // if the operation use thread_pool in inner context will throw exception.
  auto inner_context = new (std::nothrow) lite::InnerContext();
  inner_context->Init();
  MS_CHECK_TRUE_MSG(inner_context != nullptr, nullptr, "Create InnerContext failed.");
  inner_context->thread_num_ = kSingleThread;
  inner_context->instructions_ctx_.support_sdot = true;
  return inner_context;
}

schema::PrimitiveType GetSchemaPrimitiveType(const AnfNodePtr &node) {
  auto primitive_t = GetPrimitiveT(node);
  if (primitive_t == nullptr) {
    MS_LOG(ERROR) << "Failed to generate PrimitiveT.";
    return schema::PrimitiveType::PrimitiveType_NONE;
  }
  return GetSchemaPrimType(primitive_t.get());
}

STATUS CreateMatmulPackDataIntoTable(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                     OpParameter *op_parameter, const kernel::KernelKey &desc,
                                     const mindspore::lite::InnerContext *ctx) {
  if (!KernelRegistry::GetInstance()->SupportKernel(desc)) {
    MS_LOG(ERROR) << op_parameter->name_ << " is not supported.";
    return RET_ERROR;
  }

  kernel::LiteKernel *kernel =
    KernelRegistry::GetInstance()->GetLiteKernel(in_tensors, out_tensors, ctx, desc, op_parameter);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Anf node cannot be nullptr.";
    return RET_ERROR;
  }
  kernel->set_name(op_parameter->name_);

  if (kernel->Prepare() != RET_OK) {
    MS_LOG(ERROR) << "Failed to generate pack data for " << op_parameter->name_ << ".";
    return RET_ERROR;
  }

  PackDataWrapper::GetInstance().AddPackedKernel(op_parameter->name_, kernel);
  return RET_OK;
}

schema::QuantType GetQuantType(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, schema::QuantType::QuantType_QUANT_NONE, "cnode cannot be nullptr.");
  const std::map<lite::quant::QuantType, schema::QuantType> cast_mapping{
    {lite::quant::QuantType::QUANT_NONE, schema::QuantType::QuantType_QUANT_NONE},
    {lite::quant::QuantType::QUANT_WEIGHT, schema::QuantType::QuantType_QUANT_WEIGHT},
    {lite::quant::QuantType::QUANT_ALL, schema::QuantType::QuantType_QUANT_ALL},
    {lite::quant::QuantType::QUANT_DYNAMIC, schema::QuantType::QuantType_QUANT_DYNAMIC},
  };
  lite::quant::QuantType quant_type = lite::quant::QuantType::QUANT_NONE;

  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  if (quant_param_holder != nullptr) {
    quant_type = quant_param_holder->quant_type();
  }
  if (cast_mapping.find(quant_type) == cast_mapping.end()) {
    MS_LOG(WARNING) << "Cannot find quant type " << quant_type << " in supported offline packing operations' list.";
    return schema::QuantType::QuantType_QUANT_NONE;
  }
  return cast_mapping.at(quant_type);
}

TypeId GetDataType(const CNodePtr &cnode, const std::vector<Tensor *> &in_tensors,
                   const std::vector<Tensor *> &out_tensors) {
  if (in_tensors.empty()) {
    MS_LOG(ERROR) << "in tensor is empty.";
    return kTypeUnknown;
  }

  // Currently, fp16 is not a supported option.
  TypeId data_type =
    in_tensors[0]->data_type() == kObjectTypeTensorType ? kNumberTypeFloat32 : in_tensors[0]->data_type();
  // How to judge quant type?
  auto quant_type = GetQuantType(cnode);
  if (quant_type == schema::QuantType_QUANT_WEIGHT) {
    data_type =
      in_tensors.front()->data_type() == kNumberTypeBool ? TypeId::kNumberTypeBool : TypeId::kNumberTypeFloat32;
  }
  return data_type;
}

void QuantParamTToQuantParam(const schema::QuantParamT &quant_param_t, lite::LiteQuantParam *quant_param) {
  quant_param->inited = true;
  quant_param->bitNum = quant_param_t.numBits;
  quant_param->scale = quant_param_t.scale;
  quant_param->zeroPoint = quant_param_t.zeroPoint;
  quant_param->var_corr = quant_param_t.varCorr;
  quant_param->mean_corr = quant_param_t.meanCorr;
  quant_param->roundType = quant_param_t.roundType;
  quant_param->multiplier = quant_param_t.multiplier;
  quant_param->dstDtype = quant_param_t.dstDtype;
  quant_param->min = quant_param_t.min;
  quant_param->max = quant_param_t.max;
}

void AddQuantParams(Tensor *in_tensor, const std::vector<schema::QuantParamT> &quant_param_t) {
  std::vector<lite::LiteQuantParam> lite_quant_params(quant_param_t.size());
  for (size_t i = 0; i < lite_quant_params.size(); i++) {
    QuantParamTToQuantParam(quant_param_t[i], &lite_quant_params[i]);
  }
  in_tensor->set_quant_params(lite_quant_params);
}

STATUS CreateLiteTensor(const CNodePtr &cnode, std::vector<Tensor *> *in_tensors, std::vector<Tensor *> *out_tensors) {
  std::vector<int> shape(0);
  mindspore::TypeId type_id = TypeId::kTypeUnknown;
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  std::vector<std::vector<schema::QuantParamT>> input_quant_params_vec;
  std::vector<std::vector<schema::QuantParamT>> output_quant_params_vec;
  if (quant_param_holder != nullptr) {
    input_quant_params_vec = quant_param_holder->get_input_quant_params();
    output_quant_params_vec = quant_param_holder->get_output_quant_params();
  }

  // Generate input tensor.
  for (size_t i = kPrimIndex + 1; i < cnode->inputs().size(); i++) {
    if (opt::GetDataTypeFromAnfNode(cnode->input(i), &type_id) != RET_OK) {
      MS_LOG(ERROR) << "Cannot get data type from " << cnode->input(i)->fullname_with_scope();
      return RET_ERROR;
    }
    void *tensor_data = nullptr;
    Category category = cnode->input(i)->isa<Parameter>() ? lite::Category::CONST_TENSOR : lite::Category::VAR;

    MS_CHECK_TRUE_MSG(GetCNodeOrParameterShapeVec(cnode->input(i), &shape) == RET_OK, RET_ERROR,
                      "Infer shape must be done when using offline packing.");
    MS_CHECK_TRUE_MSG(!shape.empty(), RET_ERROR, "Infer shape must be done when using offline packing.");
    // Get tensor data from parameter node.
    if (cnode->input(i)->isa<Parameter>() && cnode->input(i)->cast<ParameterPtr>()->has_default()) {
      auto param_node = cnode->input(i)->cast<ParameterPtr>();
      if (param_node->has_default()) {
        auto tensor_info = std::static_pointer_cast<tensor::Tensor>(param_node->default_param());
        tensor_data = tensor_info->data().data();
      }
    }
    auto in_tensor = new (std::nothrow) Tensor(type_id, shape);
    MS_CHECK_TRUE_MSG(in_tensor != nullptr, RET_ERROR, "Create input tensor failed.");
    in_tensor->set_category(category);
    // Tensor data is managed by funcGraph.
    in_tensor->set_data(tensor_data, false);
    // Setup quant params.
    if (type_id == TypeId::kNumberTypeInt8 && !input_quant_params_vec.empty()) {
      AddQuantParams(in_tensor, input_quant_params_vec.front());
      input_quant_params_vec.erase(input_quant_params_vec.begin());
    }
    in_tensors->emplace_back(in_tensor);
    shape.clear();
    type_id = TypeId::kTypeUnknown;
  }

  if (!input_quant_params_vec.empty()) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " quant params' count are not equal to inputs' size";
  }

  // Generate output tensor.
  MS_CHECK_TRUE_MSG(GetCNodeOrParameterShapeVec(cnode, &shape) == RET_OK, RET_ERROR,
                    "Infer shape must be done when using offline packing.");
  MS_CHECK_TRUE_MSG(!shape.empty(), RET_ERROR, "Infer shape must be done when using offline packing.");
  if (opt::GetDataTypeFromAnfNode(cnode, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "Cannot get data type from " + cnode->fullname_with_scope() + ".";
    return RET_ERROR;
  }
  auto out_tensor = new (std::nothrow) Tensor(type_id, shape);
  MS_CHECK_TRUE_MSG(out_tensor != nullptr, RET_ERROR, "Create output tensor failed.");
  if (type_id == TypeId::kNumberTypeInt8 && !output_quant_params_vec.empty()) {
    AddQuantParams(out_tensor, output_quant_params_vec.front());
    output_quant_params_vec.erase(output_quant_params_vec.begin());
  }
  out_tensors->emplace_back(out_tensor);

  if (in_tensors->size() != cnode->inputs().size() - 1 || out_tensors->empty()) {
    MS_LOG(ERROR) << "Failed to populate input tensors for " << cnode->fullname_with_scope() << ".";
    return RET_ERROR;
  }

  return RET_OK;
}

STATUS MatmulPacking(const mindspore::CNodePtr &cnode_ptr, const FuncGraphPtr &funcGraphPtr,
                     const lite::InnerContext *ctx) {
  if (cnode_ptr == nullptr) {
    MS_LOG(ERROR) << "Matmul node cannot be nullptr.";
    return RET_ERROR;
  }
  auto primT = mindspore::lite::GetPrimitiveT(cnode_ptr->input(kPrimIndex));
  if (primT == nullptr) {
    MS_LOG(ERROR) << "Failed to generate PrimitiveT for " << cnode_ptr->fullname_with_scope() << ".";
    return RET_ERROR;
  }
  OpParameter *op_parameter = GetOpParameter(primT.get());
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Failed to generate op parameter for " << cnode_ptr->fullname_with_scope() << ".";
    return RET_ERROR;
  }
  op_parameter->thread_num_ = kSingleThread;
  op_parameter->quant_type_ = GetQuantType(cnode_ptr);

  (void)snprintf(op_parameter->name_, cnode_ptr->fullname_with_scope().length() + 1, "%s",
                 cnode_ptr->fullname_with_scope().c_str());

  std::vector<Tensor *> in_tensors;
  std::vector<Tensor *> out_tensors;
  if (CreateLiteTensor(cnode_ptr, &in_tensors, &out_tensors) != RET_OK) {
    MS_LOG(ERROR) << "Failed to populate input tensors for " << cnode_ptr->fullname_with_scope() << ".";
    return RET_ERROR;
  }

  TypeId data_type = GetDataType(cnode_ptr, in_tensors, out_tensors);
  MS_CHECK_TRUE_MSG(data_type != TypeId::kTypeUnknown, RET_ERROR,
                    "Can't get data type from " + cnode_ptr->fullname_with_scope() + ".");
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, NHWC, op_parameter->type_};

  return CreateMatmulPackDataIntoTable(in_tensors, out_tensors, op_parameter, desc, ctx);
}

BackendType FindBackend(const std::string &target_backend) {
  if (target_backend == std::string(kAndroidArmCpuBackendOption)) {
    return BackendType::kAndroidArmCpuBackend;
  }
  return BackendType::kUnknownBackend;
}

STATUS OfflinePackingOptimizer::Optimize(const FuncGraphPtr &func_graph, const std::string &target_backend) {
  BackendType backend = FindBackend(target_backend);
  if (backend == BackendType::kUnknownBackend ||
      this->packing_strategies_selector_.find(backend) == this->packing_strategies_selector_.end() ||
      this->ctx_creator_selector_.find(backend) == this->ctx_creator_selector_.end()) {
    MS_LOG(ERROR) << target_backend << " is not supported to do offline packing.";
    return RET_ERROR;
  }

  // Get built-in backend optimizer.
  std::map<schema::PrimitiveType, OfflinePackingFunc> selected_backend_op_cvt =
    this->packing_strategies_selector_[backend];
  mindspore::lite::InnerContext *inner_context = this->ctx_creator_selector_[backend]();
  MS_CHECK_TRUE_MSG(inner_context != nullptr, RET_ERROR, "Failed to initialize runtime context.");

  auto anf_nodes = mindspore::TopoSort(func_graph->get_return());
  for (auto &anf_node : anf_nodes) {
    if (!utils::isa<CNodePtr>(anf_node)) {
      continue;
    }
    if (mindspore::opt::CheckPrimitiveType(anf_node, prim::kPrimReturn) ||
        mindspore::opt::CheckPrimitiveType(anf_node, prim::kPrimMakeTuple) ||
        mindspore::opt::CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto cnode = anf_node->cast<CNodePtr>();
    schema::PrimitiveType op_type = GetSchemaPrimitiveType(cnode->input(kPrimIndex));
    if (selected_backend_op_cvt.find(op_type) != selected_backend_op_cvt.end()) {
      OfflinePackingFunc packing_func = selected_backend_op_cvt[op_type];
      if (packing_func(cnode, func_graph, inner_context) != RET_OK) {
        MS_LOG(ERROR) << "Failed to pack for " << anf_node->fullname_with_scope();
        delete inner_context;
        return RET_ERROR;
      }
    }
  }
  delete inner_context;
  return RET_OK;
}
}  // namespace mindspore::lite
