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
#include "include/common/utils/convert_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "tools/common/graph_util.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/converter/graphdef_transform.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "ops/batch_matmul.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/ops_func_impl/gather.h"
#include "ops/op_utils.h"
#include "src/common/utils.h"
#include "src/common/file_utils.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "ir/anf.h"
#include "tools/converter/export_model.h"
#include "tools/converter/parser/parser_utils.h"
#include "ops/other_ops.h"
#include "utils/anf_utils.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/op_base.h"

using std::string;
using std::vector;

namespace mindspore::lite::quant {
namespace {
constexpr size_t kGatherAxisIndex = 3;
constexpr int kDefaultThreadNum = 4;
constexpr size_t kEncMaxLen = 16;
constexpr size_t kModelSizeLimit = static_cast<size_t>(2) * 1024 * 1024 * 1024;
constexpr int kFakeQuantMinIndex = 1;
constexpr int kFakeQuantMaxIndex = 2;
}  // namespace

int GetQuantType(const CNodePtr &cnode, quant::QuantType *quant_type) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(quant_type);
  auto quant_param_holder = GetCNodeQuantHolder(cnode);
  if (quant_param_holder == nullptr) {
    *quant_type = quant::QUANT_NONE;
    return RET_OK;
  }
  *quant_type = quant_param_holder->quant_type();
  return RET_OK;
}

int GetQuantTypeNew(const CNodePtr &cnode, quant::QuantType *quant_type) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(quant_type);
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr.";
    return RET_NULL_PTR;
  }
  auto quant_type_attr = primitive->GetAttr(quant::kQuantType);
  if (quant_type_attr == nullptr) {
    *quant_type = quant::QUANT_NONE;
    return RET_OK;
  }
  *quant_type = static_cast<quant::QuantType>(GetValue<int32_t>(quant_type_attr));
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
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr.");
  auto node_users = manager->node_users()[cnode];
  MS_CHECK_TRUE_MSG(!node_users.empty(), false, "node_users is empty.");
  for (auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(output_cnode != nullptr, false, "output_cnode is nullptr.");
    if (!opt::CheckPrimitiveType(output_cnode, prim::kPrimReturn)) {
      return false;
    }
  }
  return true;
}

int GetCastNodeType(const FuncGraphPtr &func_graph, const CNodePtr &cnode, CastNodeType *cast_node_type) {
  CHECK_NULL_RETURN(cast_node_type);
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
  if (size == nullptr) {
    return kLiteError;
  }
  if (param->commonQuantParam.workspace.empty()) {
    MS_LOG(ERROR) << "The model is larger than 2G, mixedBitWeightQuant config needs to set workspace to save tmp model";
    return kLiteError;
  }
  std::string tmp_save_file_path = param->commonQuantParam.workspace + "/tmp.ms";
  tmp_save_file_path = lite::RealPath(tmp_save_file_path.c_str());
  if (tmp_save_file_path.empty()) {
    MS_LOG(ERROR) << param->commonQuantParam.workspace << " is invalid path. Please check it again.";
    return kLiteError;
  }
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
  if (size == nullptr) {
    return kLiteError;
  }
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
  CHECK_NULL_RETURN_VOID(param_node);
  CHECK_NULL_RETURN_VOID(tensor_info);
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

int UpdateTensorDataAndSize(const AnfNodePtr &node, const tensor::TensorPtr &weight, const void *quant_datas,
                            size_t new_size, TypeId new_data_type) {
  CHECK_NULL_RETURN(quant_datas);
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
  auto input_node = cnode->input(input_index + kPrimOffset);
  if (input_node->isa<mindspore::Parameter>()) {
    tensor::TensorPtr input_tensor = quant::GetNodeTensor(input_node);
    if (input_tensor != nullptr) {
      auto quantization_params = input_tensor->quant_params();
      if (!quantization_params.empty()) {
        auto quantization_param = quantization_params.front();
        auto axis_attr = quantization_param->GetAttr(kChannelAxis);
        if (axis_attr != nullptr) {
          if (axis_attr->isa<Int64Imm>()) {
            auto axis = axis_attr->cast<Int64ImmPtr>()->value();
            MS_LOG(INFO) << "Quantization param axis is " << axis;
            return axis;
          }
          MS_LOG(WARNING) << "Quantization param axis_attr is not int64";
        }
      }
    }
  }
  auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  if (primitive->name() == ops::kNameMatMulFusion || primitive->name() == ops::kNameMatMul ||
      primitive->name() == ops::kNameBatchMatMul) {
    return GetMatMulPreferredDim(primitive, input_index, dims);
  } else if (primitive->name() == ops::kNameConv2dTransposeFusion) {
    return GetDeConvPreferredDim(primitive, dims);
  } else if (primitive->name() == ops::kNameGather) {
    return GetGatherPreferredDim(cnode);
  } else if (primitive->name() == "FFN") {
    // For FFN MatMul, transpose is false
    return dims.size() - 1;
  }
  // The first index.
  return 0;
}

int GetFollowedNodePreferredDim(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::vector<int> &dims) {
  auto manager = mindspore::Manage(func_graph, true);
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
    return 0;
  }
  if (node_users.size() > 1) {
    MS_LOG(WARNING) << "The cnode dont has only one followed node";
    return 0;
  }
  auto node_user = node_users.begin();
  if (!utils::isa<CNodePtr>(node_user->first)) {
    MS_LOG(WARNING) << "The followed op: " << node_user->first->fullname_with_scope() << " is not cnode";
    return 0;
  }
  auto node_user_cnode = utils::cast<CNodePtr>(node_user->first);
  return GetPreferredDim(node_user_cnode, node_user->second - 1, dims);
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

bool CheckFollowedNodeInSet(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                            const std::set<PrimitivePtr> &support_primitive_types) {
  auto manager = mindspore::Manage(func_graph, true);
  auto node_users = manager->node_users()[cnode];
  if (node_users.empty()) {
    MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
    return false;
  }
  for (auto &node_user : node_users) {
    if (!utils::isa<CNodePtr>(node_user.first)) {
      MS_LOG(INFO) << "The followed op: " << node_user.first->fullname_with_scope() << " is not cnode";
      return false;
    }
    auto node_user_cnode = utils::cast<CNodePtr>(node_user.first);
    if (!CheckNodeInSet(node_user_cnode, support_primitive_types)) {
      return false;
    }
  }
  return true;
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
    if (opt::CheckPrimitiveType(cnode, prim::kPrimUpdateState)) {
      continue;
    }
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
    auto ret = ConvertCNodeFp32ToFp16(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << " convert cnode fp32 to fp16.";
      return ret;
    }
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

      tensor::TensorPtr input_tensor = quant::GetNodeTensor(input);
      MS_CHECK_TRUE_MSG(input_tensor != nullptr, RET_NULL_PTR, "Get node tensor failed.");
      auto quant_params = input_tensor->quant_params();
      tensor_ptr->set_quant_param(quant_params);

      param_node->set_default_param(tensor_ptr);
      param_node->set_abstract(tensor_ptr->ToAbstract());
    }
  }
  return RET_OK;
}

bool IsPerchannelWeight(const std::vector<schema::QuantParamT> &quant_params, const tensor::TensorPtr &weight,
                        int preferred_dim) {
  auto dims = weight->shape();
  return (static_cast<int>(quant_params.size()) == dims[preferred_dim]);
}

QuantizationParamPtr ConvertQuantParamTToQuantizationParam(const std::vector<schema::QuantParamT> &quant_params) {
  if (quant_params.empty()) {
    return nullptr;
  }
  QuantizationParam quantization(quant::kLinearQuant);
  std::vector<ValuePtr> scale_list;
  std::vector<ValuePtr> zeroPoint_list;
  std::vector<ValuePtr> min_list;
  std::vector<ValuePtr> max_list;
  std::vector<ValuePtr> varCorr_list;
  std::vector<ValuePtr> meanCorr_list;
  std::vector<ValuePtr> numBits_list;
  std::vector<ValuePtr> narrowRange_list;
  std::vector<ValuePtr> dstDtype_list;
  std::vector<ValuePtr> roundType_list;
  std::vector<ValuePtr> multiplier_list;
  for (auto quant_param : quant_params) {
    scale_list.push_back(MakeValue(quant_param.scale));
    zeroPoint_list.push_back(MakeValue(quant_param.zeroPoint));
    min_list.push_back(MakeValue(quant_param.min));
    max_list.push_back(MakeValue(quant_param.max));
    varCorr_list.push_back(MakeValue(quant_param.varCorr));
    meanCorr_list.push_back(MakeValue(quant_param.meanCorr));
    numBits_list.push_back(MakeValue(quant_param.numBits));
    narrowRange_list.push_back(MakeValue(quant_param.narrowRange));
    dstDtype_list.push_back(MakeValue(quant_param.dstDtype));
    roundType_list.push_back(MakeValue(quant_param.roundType));
    multiplier_list.push_back(MakeValue(quant_param.multiplier));
  }
  quantization.AddAttr(quant::kScaleList, std::make_shared<ValueList>(scale_list));
  quantization.AddAttr(quant::kZeroPointList, std::make_shared<ValueList>(zeroPoint_list));
  quantization.AddAttr(quant::kMinList, std::make_shared<ValueList>(min_list));
  quantization.AddAttr(quant::kMaxList, std::make_shared<ValueList>(max_list));
  quantization.AddAttr(quant::kVarCorrList, std::make_shared<ValueList>(varCorr_list));
  quantization.AddAttr(quant::kMeanCorrList, std::make_shared<ValueList>(meanCorr_list));
  quantization.AddAttr(quant::kNumBitList, std::make_shared<ValueList>(numBits_list));
  quantization.AddAttr(quant::kNarrowRangeList, std::make_shared<ValueList>(narrowRange_list));
  quantization.AddAttr(quant::kDstDtypeList, std::make_shared<ValueList>(dstDtype_list));
  quantization.AddAttr(quant::kRoundTypeList, std::make_shared<ValueList>(roundType_list));
  quantization.AddAttr(quant::kMultiplierList, std::make_shared<ValueList>(multiplier_list));
  return std::make_shared<mindspore::QuantizationParam>(quantization);
}

std::vector<schema::QuantParamT> ConvertQuantizationParamToQuantParamT(const QuantizationParamPtr &quantization_param) {
  std::vector<schema::QuantParamT> quant_params;
  if (quantization_param == nullptr) {
    return quant_params;
  }
  auto scale_list_attr = quantization_param->GetAttr(quant::kScaleList);
  auto zero_point_list_attr = quantization_param->GetAttr(quant::kZeroPointList);
  auto min_list_attr = quantization_param->GetAttr(quant::kMinList);
  auto max_list_attr = quantization_param->GetAttr(quant::kMaxList);
  auto var_corr_list_attr = quantization_param->GetAttr(quant::kVarCorrList);
  auto mean_corr_list_attr = quantization_param->GetAttr(quant::kMeanCorrList);
  auto num_bits_list_attr = quantization_param->GetAttr(quant::kNumBitList);
  auto narrow_range_list_attr = quantization_param->GetAttr(quant::kNarrowRangeList);
  auto dst_dtype_list_attr = quantization_param->GetAttr(quant::kDstDtypeList);
  auto round_type_list_attr = quantization_param->GetAttr(quant::kRoundTypeList);
  auto multiplier_list_attr = quantization_param->GetAttr(quant::kMultiplierList);
  if (scale_list_attr != nullptr && zero_point_list_attr != nullptr && min_list_attr != nullptr &&
      max_list_attr != nullptr && var_corr_list_attr != nullptr && mean_corr_list_attr != nullptr &&
      num_bits_list_attr != nullptr && narrow_range_list_attr != nullptr) {
    auto scales = GetValue<std::vector<double>>(scale_list_attr);
    auto zero_points = GetValue<std::vector<int32_t>>(zero_point_list_attr);
    auto mins = GetValue<std::vector<double>>(min_list_attr);
    auto maxs = GetValue<std::vector<double>>(max_list_attr);
    auto var_corrs = GetValue<std::vector<float>>(var_corr_list_attr);
    auto mean_corrs = GetValue<std::vector<float>>(mean_corr_list_attr);
    auto num_bits_list = GetValue<std::vector<int32_t>>(num_bits_list_attr);
    auto narrow_range_list = GetValue<std::vector<bool>>(narrow_range_list_attr);
    auto dst_dtype_list = GetValue<std::vector<int32_t>>(dst_dtype_list_attr);
    auto round_type_list = GetValue<std::vector<int32_t>>(round_type_list_attr);
    auto multiplier_list = GetValue<std::vector<int32_t>>(multiplier_list_attr);
    for (size_t index = 0; index < scales.size(); ++index) {
      schema::QuantParamT quant_param;
      quant_param.scale = scales.at(index);
      quant_param.zeroPoint = zero_points.at(index);
      quant_param.min = mins.at(index);
      quant_param.max = maxs.at(index);
      quant_param.varCorr = var_corrs.at(index);
      quant_param.meanCorr = mean_corrs.at(index);
      quant_param.numBits = num_bits_list.at(index);
      quant_param.narrowRange = narrow_range_list.at(index);
      quant_param.dstDtype = dst_dtype_list.at(index);
      quant_param.roundType = round_type_list.at(index);
      quant_param.multiplier = multiplier_list.at(index);
      quant_param.inited = true;
      quant_params.push_back(quant_param);
    }
  }
  return quant_params;
}

int RemoveInputNodeQuantParam(const CNodePtr &cnode, size_t index) {
  if (cnode->size() <= index) {
    MS_LOG(ERROR) << "index out of range, cnode input size is: " << cnode->size() << ", but index: " << index;
    return RET_ERROR;
  }
  auto input_node = cnode->input(index);
  CHECK_NULL_RETURN(input_node);
  auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (IsGraphInput(input_node)) {
    if (cnode_primitive->HasAttr(quant::kGraphInputQuantParam)) {
      cnode_primitive->EraseAttr(quant::kGraphInputQuantParam);
    }
  } else if (input_node->isa<mindspore::CNode>()) {
    if (cnode_primitive->HasAttr(quant::kQuantParam)) {
      cnode_primitive->EraseAttr(quant::kQuantParam);
    }
  } else if (input_node->isa<mindspore::Parameter>() || input_node->isa<mindspore::ValueNode>()) {
    auto input_tensor = quant::GetNodeTensor(input_node);
    CHECK_NULL_RETURN(input_tensor);
    input_tensor->set_quant_param({});
  } else {
    MS_LOG(ERROR) << input_node->fullname_with_scope() << " index: " << index << " is not support "
                  << input_node->type_name() << " type.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<schema::QuantParamT> GetInputNodeQuantParam(const CNodePtr &cnode, size_t index, size_t multi_ouput_index) {
  if (cnode->size() <= index) {
    MS_LOG(WARNING) << "index out of range, cnode input size is: " << cnode->size() << ", but index: " << index;
    return {};
  }
  auto input_node = cnode->input(index);
  MS_CHECK_TRUE_MSG(input_node != nullptr, {}, "Anf node nullptr.");
  auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, {}, "Primitive is nullptr.");
  if (IsGraphInput(input_node)) {
    auto quantization_param_value = cnode_primitive->GetAttr(quant::kGraphInputQuantParam);
    if (quantization_param_value == nullptr) {
      MS_LOG(WARNING) << input_node->fullname_with_scope() << " quant param Not exist.";
      return {};
    }
    auto quantization_param = quantization_param_value->cast<mindspore::QuantizationParamPtr>();
    MS_CHECK_TRUE_MSG(quantization_param != nullptr, {}, "Graph input quant param Not exist.");
    return quant::ConvertQuantizationParamToQuantParamT(quantization_param);
  } else if (input_node->isa<mindspore::CNode>()) {
    auto input_cnode = input_node->cast<mindspore::CNodePtr>();
    auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    MS_CHECK_TRUE_MSG(input_cnode_primitive != nullptr, {}, "Primitive is nullptr.");
    if (!input_cnode_primitive->HasAttr(quant::kQuantParam)) {
      MS_LOG(WARNING) << input_node->fullname_with_scope() << " dont have quant param.";
      return {};
    }
    auto quantization_param_value = input_cnode_primitive->GetAttr(quant::kQuantParam);
    MS_CHECK_TRUE_MSG(quantization_param_value != nullptr, {}, "quantization_param_value is nullptr.");
    auto quantization_param_list = GetValue<std::vector<QuantizationParamPtr>>(quantization_param_value);
    if (quantization_param_list.size() <= multi_ouput_index) {
      MS_LOG(WARNING) << "This node's input node: " << input_cnode->fullname_with_scope()
                      << "'s output quant_params size: " << quantization_param_list.size()
                      << ", but index: " << multi_ouput_index;
      return {};
    }
    // multi-output
    return quant::ConvertQuantizationParamToQuantParamT(quantization_param_list.at(multi_ouput_index));
  } else if (input_node->isa<mindspore::Parameter>() || input_node->isa<mindspore::ValueNode>()) {
    tensor::TensorPtr input_tensor = quant::GetNodeTensor(input_node);
    MS_CHECK_TRUE_MSG(input_tensor != nullptr, {}, "Get node tensor failed.");
    auto quantization_params = input_tensor->quant_params();
    if (quantization_params.empty()) {
      MS_LOG(WARNING) << input_node->fullname_with_scope() << " quantization param is empty.";
      return {};
    }
    auto quantization_param = quantization_params.front();
    return quant::ConvertQuantizationParamToQuantParamT(quantization_param);
  } else {
    MS_LOG(ERROR) << cnode->fullname_with_scope() << " input node with index: " << index
                  << " Not supported for quant param";
  }
  return {};
}

STATUS SetInputNodeQuantParam(const CNodePtr &cnode, size_t index,
                              const std::vector<schema::QuantParamT> &quant_param) {
  auto input_node = cnode->input(index);
  MS_CHECK_TRUE_MSG(input_node != nullptr, RET_NULL_PTR, "Anf node nullptr.");
  if (IsGraphInput(input_node)) {
    auto cnode_primitive = GetValueNode<PrimitivePtr>(cnode->input(kPrimIndex));
    MS_CHECK_TRUE_MSG(cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
    auto quantization_param = quant::ConvertQuantParamTToQuantizationParam(quant_param);
    cnode_primitive->AddAttr(quant::kGraphInputQuantParam, quantization_param);
  } else if (input_node->isa<mindspore::CNode>()) {
    auto input_cnode = input_node->cast<mindspore::CNodePtr>();
    auto input_cnode_primitive = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    MS_CHECK_TRUE_MSG(input_cnode_primitive != nullptr, RET_NULL_PTR, "Primitive is nullptr.");
    auto quantization_param = ConvertQuantParamTToQuantizationParam(quant_param);
    std::vector<ValuePtr> quantization_list{quantization_param};
    input_cnode_primitive->AddAttr(quant::kQuantParam, std::make_shared<ValueList>(quantization_list));
  } else if (input_node->isa<mindspore::Parameter>() || input_node->isa<mindspore::ValueNode>()) {
    tensor::TensorPtr input_tensor = quant::GetNodeTensor(input_node);
    MS_CHECK_TRUE_MSG(input_tensor != nullptr, RET_NULL_PTR, "Get node tensor failed.");
    auto quantization_param = quant::ConvertQuantParamTToQuantizationParam(quant_param);
    CHECK_NULL_RETURN(quantization_param);
    input_tensor->set_quant_param(std::vector<std::shared_ptr<mindspore::QuantizationParam>>{quantization_param});
  } else {
    MS_LOG(WARNING) << input_node->fullname_with_scope() << " Not supported type.";
    return RET_ERROR;
  }
  return RET_OK;
}

tensor::TensorPtr GetNodeTensor(const AnfNodePtr &node) {
  // Only Parameter or ValueNode Node has tensor
  if (node->isa<Parameter>()) {
    auto parameter = node->cast<ParameterPtr>();
    if (parameter->default_param() != nullptr) {
      return parameter->default_param()->cast<tensor::TensorPtr>();
    }
  } else if (node->isa<ValueNode>()) {
    return node->cast<ValueNodePtr>()->value()->cast<tensor::TensorPtr>();
  }
  return nullptr;
}

std::vector<schema::QuantParamT> CloneQuantParam(const std::vector<schema::QuantParamT> &src) {
  MS_CHECK_TRUE_MSG(!src.empty(), {}, "Src is empty.");
  std::vector<schema::QuantParamT> dst;
  for (auto &quant_param : src) {
    schema::QuantParamT quant_param_clone;
    quant_param_clone.scale = quant_param.scale;
    quant_param_clone.zeroPoint = quant_param.zeroPoint;
    quant_param_clone.numBits = quant_param.numBits;
    quant_param_clone.narrowRange = quant_param.narrowRange;
    quant_param_clone.meanCorr = quant_param.meanCorr;
    quant_param_clone.varCorr = quant_param.varCorr;
    quant_param_clone.dstDtype = quant_param.dstDtype;
    quant_param_clone.min = quant_param.min;
    quant_param_clone.max = quant_param.max;
    quant_param_clone.roundType = quant_param.roundType;
    quant_param_clone.multiplier = quant_param.multiplier;
    dst.push_back(quant_param_clone);
  }
  return dst;
}

int CalBiasQuantParams(const std::vector<schema::QuantParamT> &active_params,
                       const std::vector<schema::QuantParamT> &weight_params,
                       std::vector<schema::QuantParamT> *bias_params) {
  std::vector<double> input_scales;
  std::vector<double> filter_scales;
  std::vector<double> bias_scales;
  size_t sizeX = active_params.size();
  for (size_t i = 0; i < sizeX; i++) {
    input_scales.emplace_back(active_params[i].scale);
  }
  size_t sizeY = weight_params.size();
  if (sizeX != sizeY) {
    if (sizeX > 1 && sizeY > 1) {
      MS_LOG(ERROR) << "input and filter's scale count cannot match!";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < sizeY; i++) {
    filter_scales.emplace_back(weight_params[i].scale);
  }
  size_t size = std::max(sizeX, sizeY);
  for (size_t i = 0; i < size; i++) {
    auto scaleX = sizeX > 1 ? input_scales[i] : input_scales[0];
    auto scaleY = sizeY > 1 ? filter_scales[i] : filter_scales[0];
    bias_scales.push_back(scaleX * scaleY);
  }
  MS_ASSERT(!bias_scales.empty());

  // set bias quant param
  for (double bias_scale : bias_scales) {
    schema::QuantParamT quant_param;
    if (bias_scale == 0) {
      MS_LOG(WARNING) << "bias_scale is 0, and set bias_scale to 1.";
      quant_param.scale = 1;
    } else {
      quant_param.scale = bias_scale;
    }
    quant_param.numBits = k32Bit;
    quant_param.zeroPoint = 0;
    quant_param.inited = true;
    bias_params->push_back(quant_param);
  }
  return RET_OK;
}

bool IsAntiQuantModeNodes(const AnfNodePtr &node) {
  CHECK_NULL_RETURN(node);
  if (!utils::isa<CNodePtr>(node) || !opt::CheckPrimitiveType(node, prim::kPrimMul)) {
    MS_LOG(INFO) << "The node is not Mul node";
    return false;
  }
  auto add_node = node->cast<CNodePtr>()->input(kIndexOne);
  if (!utils::isa<CNodePtr>(add_node) || !opt::CheckPrimitiveType(add_node, prim::kPrimAdd)) {
    MS_LOG(INFO) << "The node is not Add node";
    return false;
  }
  auto ascend_antiquant_node = add_node->cast<CNodePtr>()->input(kIndexOne);
  if (!utils::isa<CNodePtr>(ascend_antiquant_node) ||
      !opt::CheckPrimitiveType(ascend_antiquant_node, prim::kPrimAntiQuant)) {
    MS_LOG(INFO) << "The node is not AscendAntiquant node";
    return false;
  }
  return true;
}

STATUS GetScaleZpFromAntiQuantModeNodes(const AnfNodePtr &node, ParameterPtr *scale_param_node,
                                        ParameterPtr *zp_param_node) {
  CHECK_NULL_RETURN(node);
  CHECK_NULL_RETURN(scale_param_node);
  CHECK_NULL_RETURN(zp_param_node);

  if (!utils::isa<CNodePtr>(node) || !opt::CheckPrimitiveType(node, prim::kPrimMul)) {
    return RET_ERROR;
  }
  auto add_node = node->cast<CNodePtr>()->input(kIndexOne);
  auto scale_param = node->cast<CNodePtr>()->input(kIndexTwo);
  if (opt::CheckPrimitiveType(scale_param, prim::kPrimLoad)) {
    scale_param = scale_param->cast<CNodePtr>()->input(kIndexOne);
  }
  *scale_param_node = scale_param->cast<ParameterPtr>();
  CHECK_NULL_RETURN(*scale_param_node);
  if (!utils::isa<CNodePtr>(add_node) || !opt::CheckPrimitiveType(add_node, prim::kPrimAdd)) {
    return RET_ERROR;
  }
  auto zp_param = add_node->cast<CNodePtr>()->input(kIndexTwo);
  if (opt::CheckPrimitiveType(zp_param, prim::kPrimLoad)) {
    zp_param = zp_param->cast<CNodePtr>()->input(kIndexOne);
  }
  *zp_param_node = zp_param->cast<ParameterPtr>();
  CHECK_NULL_RETURN(*zp_param_node);
  return RET_OK;
}

STATUS RemoveAntiQuantModeNodes(const FuncGraphPtr &func_graph, const AnfNodePtr &node, int index) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(node);

  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
  }
  CHECK_NULL_RETURN(manager);

  if (!utils::isa<CNodePtr>(node)) {
    MS_LOG(ERROR) << "The node : " << node->fullname_with_scope() << ", it is not cnode";
    return lite::RET_ERROR;
  }
  auto cnode = node->cast<CNodePtr>();
  CHECK_NULL_RETURN(cnode);

  auto mul_node = cnode->input(index);

  if (!utils::isa<CNodePtr>(mul_node) || !opt::CheckPrimitiveType(mul_node, prim::kPrimMul)) {
    MS_LOG(WARNING) << "In AntiQuant mode, the node : " << cnode->fullname_with_scope() << " is not mul node";
    return RET_OK;
  }
  auto add_node = mul_node->cast<CNodePtr>()->input(kIndexOne);
  if (!opt::CheckPrimitiveType(add_node, prim::kPrimAdd)) {
    MS_LOG(WARNING) << "In AntiQuant mode, the node : " << add_node->fullname_with_scope() << " is not add node";
    return RET_OK;
  }
  auto ascend_antiquant_node = add_node->cast<CNodePtr>()->input(kIndexOne);
  if (!opt::CheckPrimitiveType(ascend_antiquant_node, prim::kPrimAntiQuant)) {
    MS_LOG(WARNING) << "In AntiQuant mode, the node : " << ascend_antiquant_node->fullname_with_scope()
                    << " is not antiquant node";
    return RET_OK;
  }

  manager->Replace(mul_node, ascend_antiquant_node->cast<CNodePtr>()->input(1));
  return lite::RET_OK;
}

std::vector<std::vector<int64_t>> ExtractStrategy(const ValuePtr &stra) {
  if (stra == nullptr) {
    return {};
  }

  auto var = stra->cast<ValueTuplePtr>();
  if (var == nullptr) {
    return {};
  }
  std::vector<std::vector<int64_t>> strategy;
  MS_LOG(INFO) << "Extract information: strategy " << stra->ToString();
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    for (uint64_t index = 0; index < elements.size(); ++index) {
      std::vector<int64_t> dim;
      if (elements[index]->isa<ValueSequence>()) {
        auto value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
  }

  return strategy;
}

std::vector<schema::QuantParamT> CalQuantParamWithMinMax(const tensor::TensorPtr &min_value,
                                                         const tensor::TensorPtr &max_value, bool symmetric) {
  std::vector<schema::QuantParamT> quant_params;
  // Ascend fake quant transform support PerLayer && PerChannel quant param
  if (min_value->ElementsNum() != max_value->ElementsNum()) {
    MS_LOG(ERROR) << "min value size not equal max value size";
    return {};
  }
  int size = min_value->ElementsNum();
  auto min_data = reinterpret_cast<float *>(min_value->data_c());
  auto max_data = reinterpret_cast<float *>(max_value->data_c());
  for (int i = 0; i < size; i++) {
    float real_min = *(min_data + i);
    float real_max = *(max_data + i);
    schema::QuantParamT quant_param;
    int bit_num = k8Bit;

    MS_LOG(DEBUG) << "min: " << real_min << " max: " << real_max << " bit_num: " << bit_num << " symmetric"
                  << symmetric;
    auto ret = CalQuantizationParams(&quant_param, real_min, real_max, bit_num, symmetric);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to calculate quant params";
      return {};
    }
    MS_LOG(INFO) << "quant param scale: " << quant_param.scale << " zp: " << quant_param.zeroPoint;
    quant_params.push_back(quant_param);
  }
  return quant_params;
}

std::vector<schema::QuantParamT> GetQuantParamWithFakeQuantNode(const CNodePtr &fake_quant_node, bool symmetric) {
  tensor::TensorPtr min_value;
  tensor::TensorPtr max_value;
  auto min_input = fake_quant_node->input(kFakeQuantMinIndex + kPrimOffset);
  if (utils::isa<ParameterPtr>(min_input) && min_input->cast<ParameterPtr>()->has_default() &&
      min_input->cast<ParameterPtr>()->default_param() != nullptr) {
    min_value = min_input->cast<ParameterPtr>()->default_param()->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(ERROR) << "Quant param get min value failed";
    return {};
  }
  auto max_input = fake_quant_node->input(kFakeQuantMaxIndex + kPrimOffset);
  if (utils::isa<ParameterPtr>(max_input) && max_input->cast<ParameterPtr>()->has_default() &&
      max_input->cast<ParameterPtr>()->default_param() != nullptr) {
    max_value = max_input->cast<ParameterPtr>()->default_param()->cast<tensor::TensorPtr>();
  } else {
    MS_LOG(ERROR) << "Quant param get max value failed";
    return {};
  }
  auto quant_params = CalQuantParamWithMinMax(min_value, max_value, symmetric);
  return quant_params;
}

}  // namespace mindspore::lite::quant
