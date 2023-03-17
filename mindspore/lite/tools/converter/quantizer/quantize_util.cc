/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include <deque>
#include "abstract/abstract_value.h"
#include "tools/common/graph_util.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/gather.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "ir/anf.h"
#include "tools/converter/export_model.h"
#include "tools/converter/parser/parser_utils.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
namespace {
constexpr size_t kGatherAxisIndex = 3;
constexpr int kDefaultThreadNum = 4;
constexpr size_t kEncMaxLen = 16;
constexpr size_t kModelSizeLimit = static_cast<size_t>(2) * 1024 * 1024 * 1024;
}  // namespace

int GetQuantType(const CNodePtr &cnode, quant::QuantType *quant_type) {
  CHECK_NULL_RETURN(cnode);
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  if (quant_param_holder == nullptr) {
    *quant_type = quant::QUANT_NONE;
    return RET_OK;
  }
  *quant_type = quant_param_holder->quant_type();
  return RET_OK;
}

void GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(all_func_graphs != nullptr);
  all_func_graphs->insert(func_graph);
  auto nodes = func_graph->GetOrderedCnodes();
  std::deque<CNodePtr> to_process{};
  to_process.insert(to_process.end(), nodes.begin(), nodes.end());
  while (!to_process.empty()) {
    auto &cur_cnode = to_process.front();
    for (auto &input : cur_cnode->inputs()) {
      if (!IsValueNode<FuncGraph>(input)) {
        continue;
      }
      auto new_fg = GetValueNode<FuncGraphPtr>(input);
      if (all_func_graphs->find(new_fg) != all_func_graphs->end()) {
        continue;
      }
      all_func_graphs->insert(new_fg);
      auto new_nodes = new_fg->GetOrderedCnodes();
      to_process.insert(to_process.end(), new_nodes.begin(), new_nodes.end());
    }
    to_process.pop_front();
  }
}

int UpdateDataType(const AnfNodePtr &node, TypeId new_data_type) {
  auto abstract_base = node->abstract();
  if (abstract_base == nullptr) {
    MS_LOG(ERROR) << "Abstract of node is nullptr, " << node->fullname_with_scope();
    return RET_NULL_PTR;
  }

  std::vector<AbstractBasePtr> abstracts;
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract_base);
    abstracts = abstract_tuple->elements();
  } else {
    abstracts.push_back(abstract_base);
  }
  for (auto &abstract : abstracts) {
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
    CHECK_NULL_RETURN(abstract_tensor);
    CHECK_NULL_RETURN(abstract_tensor->element());
    abstract_tensor->element()->set_type(TypeIdToType(new_data_type));
  }
  return RET_OK;
}

bool IsGraphInDTypeCast(const CNodePtr &cnode) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    return false;
  }
  auto input_node = cnode->input(1);
  MS_CHECK_FALSE(input_node == nullptr, false);
  return IsGraphInput(input_node);
}

bool IsGraphOutDTypeCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    return false;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  MS_CHECK_TRUE_RET(!node_users.empty(), RET_NULL_PTR);
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    CHECK_NULL_RETURN(output_cnode);
    if (!opt::CheckPrimitiveType(output_cnode, prim::kPrimReturn)) {
      return false;
    }
  }
  return true;
}

int GetCastNodeType(const FuncGraphPtr &func_graph, const CNodePtr &cnode, CastNodeType *cast_node_type) {
  if (!opt::CheckPrimitiveType(cnode, prim::kPrimQuantDTypeCast)) {
    MS_LOG(DEBUG) << "Not QuantDtypeCastNode, cnode name: " << cnode->fullname_with_scope();
    return RET_NOT_SUPPORT;
  }
  auto input_node = cnode->input(1);
  MS_CHECK_FALSE(input_node == nullptr, RET_ERROR);

  // input node
  TypeId pre_node_dtype = kTypeUnknown;
  if (opt::GetDataTypeFromAnfNode(input_node, &pre_node_dtype) != RET_OK) {
    MS_LOG(ERROR) << "Get data type failed, cnode name: " << input_node->fullname_with_scope();
    return RET_ERROR;
  }

  // output node
  TypeId post_node_dtype = kTypeUnknown;
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);
  auto node_users = manager->node_users()[cnode];
  MS_CHECK_TRUE_RET(!node_users.empty(), RET_NULL_PTR);
  auto output_cnode = node_users.begin()->first->cast<CNodePtr>();
  CHECK_NULL_RETURN(output_cnode);

  if (!opt::CheckPrimitiveType(output_cnode, prim::kPrimReturn)) {
    if (opt::GetDataTypeFromAnfNode(output_cnode, &post_node_dtype) != RET_OK) {
      MS_LOG(ERROR) << "Get data type failed, cnode name: " << output_cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (pre_node_dtype == kNumberTypeFloat32 &&
        (post_node_dtype == kNumberTypeInt8 || post_node_dtype == kNumberTypeUInt8)) {
      *cast_node_type = kQuant;
    } else if ((pre_node_dtype == kNumberTypeInt8 || pre_node_dtype == kNumberTypeUInt8) &&
               post_node_dtype == kNumberTypeFloat32) {
      *cast_node_type = kDeQuant;
    } else {
      MS_LOG(ERROR) << "Not support QuantDTypeCastNode, cnode name: " << cnode->fullname_with_scope();
    }
  } else {
    if (pre_node_dtype == kNumberTypeFloat32) {
      *cast_node_type = kQuant;
    } else if (pre_node_dtype == kNumberTypeInt8 || pre_node_dtype == kNumberTypeUInt8) {
      *cast_node_type = kDeQuant;
    } else {
      MS_LOG(ERROR) << "Not support QuantDTypeCastNode, cnode name: " << cnode->fullname_with_scope();
    }
  }
  return RET_OK;
}

std::string NodePrimitiveType(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "cnode is null";
    return "";
  }
  auto primitive_c = GetValueNode<std::shared_ptr<ops::PrimitiveC>>(cnode->input(0));
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "primitive_c is null";
    return "";
  }
  return primitive_c->name();
}

Status LargeModelBuildModel(const schema::MetaGraphT &meta_graph, const std::shared_ptr<ConverterPara> &param,
                            const std::shared_ptr<mindspore::Model> &model, const std::shared_ptr<Context> &context,
                            size_t *size) {
  if (param->mixedBitWeightQuantParam.workspace.empty()) {
    MS_LOG(ERROR) << "The model is larger than 2G, mixedBitWeightQuant config needs to set workspace to save tmp model";
    return kLiteError;
  }
  std::string tmp_save_file_path = param->mixedBitWeightQuantParam.workspace + "/tmp.ms";
  unsigned char encKey[kEncMaxLen] = {0};
  size_t keyLen = 0;
  auto status = MetaGraphSerializer::Save(meta_graph, tmp_save_file_path, size, encKey, keyLen, param->encrypt_mode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Save Large Model Failed: " << status << " " << GetErrorInfo(status);
    return kLiteError;
  }

  mindspore::ModelType model_type = kMindIR_Lite;
  auto ret = model->Build(tmp_save_file_path, model_type, context);
  return ret;
}

int DumpGraph(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
              const std::string &save_path) {
  FuncGraphPtr func_graph_clone;
  if (CloneFuncGraph(func_graph, param, &func_graph_clone) != RET_OK) {
    MS_LOG(ERROR) << "Clone func_graph failed";
    return RET_ERROR;
  }
  auto meta_graph = Export(func_graph_clone, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return RET_ERROR;
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    delete meta_graph;
    return RET_ERROR;
  }
  meta_graph->version = Version();

  status = UpdateGraphOutputName(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    delete meta_graph;
    return RET_ERROR;
  }

  unsigned char encKey[kEncMaxLen] = {0};
  size_t keyLen = 0;
  size_t size;
  status = MetaGraphSerializer::Save(*meta_graph, save_path, &size, encKey, keyLen, param->encrypt_mode);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Save Large Model Failed: " << status << " " << GetErrorInfo(status);
    return RET_ERROR;
  }
  return RET_OK;
}

Status BuildModelByFuncGraph(const std::shared_ptr<mindspore::Model> &model, const FuncGraphPtr &func_graph,
                             const std::shared_ptr<ConverterPara> &param, size_t *size) {
  FuncGraphPtr func_graph_clone;
  if (CloneFuncGraph(func_graph, param, &func_graph_clone) != RET_OK) {
    MS_LOG(ERROR) << "Clone func_graph failed";
    return kLiteNullptr;
  }
  auto meta_graph = Export(func_graph_clone, true, true);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Export to meta_graph failed";
    return kLiteNullptr;
  }

  // transform
  GraphDefTransform fb_transform;
  fb_transform.SetGraphDef(meta_graph);
  auto status = fb_transform.Transform(param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "FBTransform model failed";
    delete meta_graph;
    return kLiteError;
  }
  meta_graph->version = Version();

  status = UpdateGraphOutputName(meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UpdateGraphOutputName failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    delete meta_graph;
    return kLiteError;
  }

  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running.";
    delete meta_graph;
    return kLiteNullptr;
  }
  context->SetThreadNum(kDefaultThreadNum);
  context->SetThreadAffinity(kCpuBindMode);

  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  if (device_info == nullptr) {
    MS_LOG(ERROR) << "New device_info failed while running.";
    delete meta_graph;
    return kLiteNullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  device_list.push_back(device_info);

  size_t tensors_size = 0;
  for (auto &tensor : meta_graph->allTensors) {
    tensors_size += tensor->data.size();
  }

  if (tensors_size >= kModelSizeLimit) {
    auto ret = LargeModelBuildModel(*meta_graph, param, model, context, size);
    delete meta_graph;
    return ret;
  }

  flatbuffers::FlatBufferBuilder builder(kMaxNum1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  *size = builder.GetSize();
  auto *content = reinterpret_cast<const char *>(builder.GetBufferPointer());
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer return null";
    delete meta_graph;
    return kLiteNullptr;
  }

  auto ret = model->Build(content, *size, kMindIR, context);
  delete meta_graph;
  return ret;
}

mindspore::lite::Tensor *MSTensorToLiteTensor(const MSTensor &tensor) {
  if (tensor.impl() == nullptr) {
    MS_LOG(ERROR) << "Tensor " << tensor.Name() << " is nullptr.";
    return static_cast<lite::Tensor *>(nullptr);
  }
  auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(tensor.impl());
  return static_cast<mindspore::lite::Tensor *>(lite_impl->lite_tensor());
}

std::vector<mindspore::lite::Tensor *> MSTensorToLiteTensors(const std::vector<mindspore::MSTensor> &src_tensors) {
  std::vector<mindspore::lite::Tensor *> dst_tensors(src_tensors.size());
  for (const auto &src_tensor : src_tensors) {
    auto tensor = MSTensorToLiteTensor(src_tensor);
    if (tensor == nullptr) {
      return {};
    }
    dst_tensors.emplace_back(tensor);
  }
  return dst_tensors;
}

void GetParameterAndTensor(const AnfNodePtr &node, ParameterPtr *param_node, tensor::TensorPtr *tensor_info) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    return;
  }
  auto op_name = node->fullname_with_scope();

  *param_node = node->cast<ParameterPtr>();
  if (*param_node == nullptr) {
    MS_LOG(INFO) << op_name << " can not cast to ParameterPtr";
    return;
  }
  if (!(*param_node)->has_default()) {
    MS_LOG(INFO) << op_name << " not has_default";
    return;
  }

  *tensor_info = std::static_pointer_cast<tensor::Tensor>((*param_node)->default_param());
  if (*tensor_info == nullptr) {
    MS_LOG(INFO) << "default_param can not cast to tensor::Tensor";
    return;
  }
}

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, void *quant_datas, size_t new_size,
                            TypeId new_data_type) {
  MS_CHECK_TRUE_RET(weight != nullptr, RET_NULL_PTR);
  MS_CHECK_TRUE_RET(new_size > 0, RET_NULL_PTR);
  weight->set_data_type(new_data_type);
  if (new_size != static_cast<size_t>(weight->data().nbytes())) {
    MS_LOG(ERROR) << "Data size of tensor info is error.";
    return RET_ERROR;
  }
  if (memcpy_s(weight->data_c(), weight->data().nbytes(), quant_datas, new_size) != EOK) {
    MS_LOG(ERROR) << "memcpy data failed.";
    return RET_ERROR;
  }
  // set dtype
  auto ret = UpdateDataType(node, new_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << node->fullname_with_scope() << " set new dtype failed.";
    return ret;
  }
  return RET_OK;
}

int GetMatMulPreferredDim(const PrimitivePtr &primitive, int input_index, const std::vector<int> &dims) {
  size_t last_first_index = dims.size() - 1;
  size_t last_second_index = dims.size() - 2;
  auto matmul_prim = api::MakeShared<ops::MatMul>(primitive);
  MS_ASSERT(matmul_prim != nullptr);
  // For MatMul A
  if (input_index == 0) {
    if (matmul_prim->GetAttr(ops::kTransposeA) != nullptr && matmul_prim->get_transpose_a()) {
      return last_first_index;
    } else {
      return last_second_index;
    }
  }
  // For MatMul B
  if (input_index == 1) {
    if (matmul_prim->GetAttr(ops::kTransposeB) != nullptr && matmul_prim->get_transpose_b()) {
      return last_second_index;
    } else {
      return last_first_index;
    }
  }
  return 0;
}

int GetDeConvPreferredDim(const PrimitivePtr &primitive, const std::vector<int> &dims) {
  auto prim = api::MakeShared<ops::Conv2DTranspose>(primitive);
  MS_ASSERT(prim != nullptr);
  if (prim->get_in_channel() == prim->get_group() && prim->get_out_channel() == prim->get_group()) {
    // DepthWise-DeConv (CO\CI) KH KW 1
    return 0;
  }
  // DeConv:CI KH KW CO
  return dims.size() - 1;
}

int GetGatherPreferredDim(const CNodePtr &cnode) {
  if (cnode->size() < kGatherAxisIndex + kPrimOffset) {
    MS_LOG(WARNING) << "gather cnode size < 4.";
    return 0;
  }
  DataInfo data_info;
  auto output_type_node = cnode->input(kGatherAxisIndex);
  if (utils::isa<ParameterPtr>(output_type_node)) {
    if (FetchDataFromParameterNode(cnode, kGatherAxisIndex, converter::kFmkTypeMs, &data_info, true) != lite::RET_OK) {
      MS_LOG(WARNING) << "Fetch data from parameter node failed.";
      return 0;
    }
  } else if (utils::isa<ValueNodePtr>(output_type_node)) {
    if (FetchDataFromValueNode(cnode, kGatherAxisIndex, converter::kFmkTypeMs, false, &data_info, true) !=
        lite::RET_OK) {
      MS_LOG(WARNING) << "Fetch data from value node failed.";
      return 0;
    }
  } else {
    MS_LOG(WARNING) << "The data type is not a const.";
    return 0;
  }

  auto axis_data = reinterpret_cast<const int *>(data_info.data_.data());
  CHECK_NULL_RETURN(axis_data);
  return axis_data[0];
}

int GetPreferredDim(const CNodePtr &cnode, int input_index, const std::vector<int> &dims) {
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  if (primitive->name() == ops::kNameMatMulFusion || primitive->name() == ops::kNameMatMul) {
    return GetMatMulPreferredDim(primitive, input_index, dims);
  } else if (primitive->name() == ops::kNameConv2dTransposeFusion) {
    return GetDeConvPreferredDim(primitive, dims);
  } else if (primitive->name() == ops::kNameGather) {
    return GetGatherPreferredDim(cnode);
  }
  // The first index.
  return 0;
}

std::vector<int> ConvertShapeVectorToInt32(const ShapeVector &dims) {
  std::vector<int> shape;
  for (auto dim : dims) {
    if (dim > INT32_MAX || dim < INT32_MIN) {
      MS_LOG(ERROR) << dim << " over int32 range.";
      shape.push_back(-1);
    } else {
      shape.push_back(dim);
    }
  }
  return shape;
}

bool CheckNodeInSet(const CNodePtr &cnode, const std::set<PrimitivePtr> &support_primitive_types) {
  for (const auto &type : support_primitive_types) {
    if (opt::CheckPrimitiveType(cnode, type)) {
      return true;
    }
  }
  return false;
}

int DeQuantData(const mindspore::MSTensor *tensor, std::vector<double> *dequant_data) {
  return DeQuantData(reinterpret_cast<const int8_t *>(tensor->Data().get()), tensor->ElementNum(),
                     tensor->QuantParams(), dequant_data);
}

int GetElementNumFromShape(const std::vector<int> &dims, int *total_size) {
  CHECK_NULL_RETURN(total_size);
  *total_size = 1;
  for (auto dim : dims) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(*total_size, dim), RET_ERROR, "Int mul overflow.");
    *total_size *= dim;
  }
  return RET_OK;
}

int GetBucketAllIndex(const std::vector<int> &dims, int preferred_dim,
                      std::vector<std::vector<size_t>> *buckets_data_index) {
  CHECK_NULL_RETURN(buckets_data_index);
  int outer = 1;
  for (int i = 0; i < preferred_dim; i++) {
    outer *= dims[i];
  }
  int bucket_count = dims[preferred_dim];
  int inner = 1;
  for (size_t i = preferred_dim + 1; i < dims.size(); i++) {
    inner *= dims[i];
  }
  if (inner <= 0 || outer <= 0 || bucket_count <= 0) {
    return RET_ERROR;
  }
  for (int i = 0; i < bucket_count; i++) {
    auto index = i * inner;
    std::vector<size_t> bucket_index(inner * outer);
    for (int j = 0; j < outer; j++) {
      for (int k = 0; k < inner; k++) {
        bucket_index[j * inner + k] = index + k;
      }
      index += bucket_count * inner;
    }
    buckets_data_index->push_back(bucket_index);
  }
  return RET_OK;
}

bool CheckControlFlowType(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  std::map<std::string, PrimitivePtr> control_flow_ops = {{"PartialFusion", prim::kPrimPartialFusion},
                                                          {"Switch", prim::kPrimSwitch},
                                                          {"switch_layer", prim::kPrimSwitchLayer},
                                                          {"call", prim::kPrimCall}};

  if (node->isa<mindspore::CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    // control flow call
    if (!IsValueNode<mindspore::Primitive>(cnode->input(kPrimIndex))) {
      return true;
    }
    auto prim = GetValuePtr<mindspore::Primitive>(cnode->input(kPrimIndex));
    if (control_flow_ops.find(prim->name()) != control_flow_ops.end()) {
      return true;
    }
  } else if (node->isa<ValueNode>()) {
    auto prim = GetValuePtr<mindspore::Primitive>(node);
    if (control_flow_ops.find(prim->name()) != control_flow_ops.end()) {
      return true;
    }
  }
  return false;
}
int CloneFuncGraph(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                   FuncGraphPtr *func_graph_bak) {
  CHECK_NULL_RETURN(func_graph_bak);
  CHECK_NULL_RETURN(param);
  std::map<FuncGraphPtr, FuncGraphPtr> cloned_func_graph;
  *func_graph_bak = lite::CloneFuncGraph(func_graph, param, &cloned_func_graph);
  CHECK_NULL_RETURN(*func_graph_bak);
  static auto root_func_manager = Manage(*func_graph_bak);
  std::set<FuncGraphPtr> all_func_graphs = {};
  lite::GetAllFuncGraph(*func_graph_bak, &all_func_graphs);
  for (const auto &graph : all_func_graphs) {
    graph->set_manager(root_func_manager);
  }
  return RET_OK;
}

int MarkOriginDataType(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    TypeId type_id = kTypeUnknown;
    auto ret = opt::GetDataTypeFromAnfNode(cnode, &type_id);
    if (ret != RET_OK) {
      MS_LOG(INFO) << "CNode data type is unknown.";
      return RET_OK;
    }
    if (type_id != kTypeUnknown) {
      MS_LOG(INFO) << cnode->fullname_with_scope() << " origin type is " << type_id;
      cnode->AddAttr("origin_type", MakeValue(static_cast<int>(type_id)));
    }
  }
  return RET_OK;
}

int ConvertFp16ToFp32(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    auto ret = ConvertCNodeFp16ToFp32(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " convert fp16 To fp32 failed.";
      return ret;
    }
  }
  return RET_OK;
}

int ConvertCNodeFp32ToFp16(const CNodePtr &cnode) {
  for (size_t i = kPrimOffset; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    if (input->isa<Parameter>() && input->cast<ParameterPtr>()->has_default()) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " Parameter.";
      ParameterPtr param_node;
      tensor::TensorPtr tensor_info;
      GetParameterAndTensor(input, &param_node, &tensor_info);
      CHECK_NULL_RETURN(tensor_info);
      CHECK_NULL_RETURN(param_node);
      if (tensor_info->data_type() == kNumberTypeFloat32) {
        MS_LOG(INFO) << "convert " << input->fullname_with_scope() << " from fp32 to fp16.";
        auto data = static_cast<float *>(tensor_info->data_c());
        std::vector<float16> fp16_data(tensor_info->DataSize());
        for (size_t j = 0; j < tensor_info->DataSize(); j++) {
          fp16_data[j] = mindspore::Float16(data[j]);
        }
        mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(
          kNumberTypeFloat16, tensor_info->shape_c(), fp16_data.data(), fp16_data.size() * sizeof(float) / 2);
        param_node->set_default_param(tensor_ptr);
        param_node->set_abstract(tensor_ptr->ToAbstract());
      }
    } else if (input->isa<ValueNode>()) {
      auto value_node = input->cast<ValueNodePtr>();
      DataInfo data_info;
      auto ret = FetchDataFromValueNode(cnode, i, converter::kFmkTypeMs, false, &data_info, false);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Fetch data from value node failed.";
        return ret;
      }
      std::vector<int64_t> shapes;
      for (size_t j = 0; j < data_info.shape_.size(); ++j) {
        shapes.push_back(data_info.shape_.at(j));
      }
      int total_size = 0;
      ret = GetElementNumFromShape(data_info.shape_, &total_size);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "GetElementNumFromShape failed.";
        return ret;
      }
      if (data_info.data_type_ == kNumberTypeFloat32) {
        MS_LOG(ERROR) << "convert " << input->fullname_with_scope() << " from fp32 to fp16.";
        auto data = static_cast<float *>(data_info.data_ptr_);
        std::vector<float16> fp16_data(total_size);
        for (int j = 0; j < total_size; j++) {
          fp16_data[j] = mindspore::Float16(data[j]);
        }
        mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(
          kNumberTypeFloat16, shapes, fp16_data.data(), fp16_data.size() * sizeof(float) / 2);
        auto values = MakeValue(tensor_ptr);
        value_node->set_value(values);
        value_node->set_abstract(tensor_ptr->ToAbstract());
      }
    }
  }
  return RET_OK;
}

int ConvertFp32ToFp16(const FuncGraphPtr &func_graph) {
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    ConvertCNodeFp32ToFp16(cnode);
  }
  return RET_OK;
}

int ConvertCNodeFp16ToFp32(const CNodePtr &cnode) {
  for (size_t i = kPrimOffset; i < cnode->size(); ++i) {
    auto input = cnode->input(i);
    if (!input->isa<Parameter>() || !input->cast<ParameterPtr>()->has_default()) {
      continue;
    }
    ParameterPtr param_node;
    tensor::TensorPtr tensor_info;
    GetParameterAndTensor(input, &param_node, &tensor_info);
    CHECK_NULL_RETURN(tensor_info);
    CHECK_NULL_RETURN(param_node);
    if (tensor_info->data_type() == kNumberTypeFloat16) {
      MS_LOG(INFO) << "convert " << input->fullname_with_scope() << " from fp16 to fp32.";
      auto data = static_cast<float16 *>(tensor_info->data_c());
      std::vector<float> fp32_data(tensor_info->DataSize());
      for (size_t j = 0; j < tensor_info->DataSize(); j++) {
        fp32_data[j] = mindspore::Float16::ToFloat32(data[j]);
      }
      mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(
        kNumberTypeFloat32, tensor_info->shape_c(), fp32_data.data(), fp32_data.size() * sizeof(float));
      param_node->set_default_param(tensor_ptr);
      param_node->set_abstract(tensor_ptr->ToAbstract());
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
