/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "extendrt/mindir_loader/mindir_model/mindir_model.h"
#include "utils/ms_utils_secure.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "extendrt/mindir_loader/mindir_model/kernel_mod_util.h"
#include "src/litert/kernel_exec.h"
#include "extendrt/mindir_loader/mindir_model/inner_kernel.h"
#include "extendrt/mock/lite_runtime/populate/base_operator_populate_register.h"

#include "src/litert/kernel_registry.h"

namespace mindspore::infer::mindir {
#define IS_LITTLE_ENDIAN (uint8_t)1U

bool MindirModel::ModelVerify() const { return true; }

int MindirModel::ConvertTensors(std::vector<mindspore::lite::Tensor *> *lite_tensors) {
  if (lite_tensors == nullptr) {
    MS_LOG(ERROR) << "lite tensors is null.";
    return mindspore::lite::RET_NULL_PTR;
  }

  uint32_t tensor_count = this->all_mindir_tensors_.size();
  auto model_input_indices = this->graph_.input_indices_;
  auto model_output_indices = this->graph_.output_indices_;

  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto src_tensor = this->all_mindir_tensors_[i];
    auto *dst_tensor = ConvertTensor(src_tensor);
    if (dst_tensor == nullptr) {
      MS_LOG(ERROR) << "Convert new " << i << "th tensor failed!";
      return mindspore::lite::RET_NULL_PTR;
    }

    if (mindspore::lite::IsContain(model_input_indices, i)) {
      dst_tensor->set_category(mindspore::lite::Category::GRAPH_INPUT);
    }
    if (mindspore::lite::IsContain(model_output_indices, i)) {
      // a tensor is as both input and output, would be treated as an input.
      if (!dst_tensor->IsGraphInput()) {
        dst_tensor->set_category(mindspore::lite::Category::GRAPH_OUTPUT);
      }
    }

    auto ret = CheckTensorValid(dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Check " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }

    lite_tensors->emplace_back(dst_tensor);
  }
  return mindspore::lite::RET_OK;
}

std::string MindirModel::GetModelPath() const { return this->model_path_; }

mindspore::kernel::KernelExec *MindirModel::FindBackendKernel(const std::vector<mindspore::lite::Tensor *> &in_tensors,
                                                              const std::vector<mindspore::lite::Tensor *> &out_tensors,
                                                              const LiteGraph::Node *node, lite::InnerContext *context,
                                                              TypeId prefer_data_type) {
  if (select_lite_kernel_) {
    return FindLiteKernel(in_tensors, out_tensors, node, context, prefer_data_type);
  }
  std::shared_ptr<kernel::InnerKernel> inner_kernel =
    mindspore::kernel::KernelModUtil::GetInnerKernel(in_tensors, out_tensors, node, context);
  kernel::KernelExec *kernel_exec = new kernel::KernelExec(inner_kernel);
  auto desc = kernel_exec->desc();
  desc.data_type = in_tensors.front()->data_type();
  kernel_exec->set_desc(desc);
  return kernel_exec;
}

mindspore::kernel::KernelExec *MindirModel::FindLiteKernel(const std::vector<mindspore::lite::Tensor *> &in_tensors,
                                                           const std::vector<mindspore::lite::Tensor *> &out_tensors,
                                                           const LiteGraph::Node *node, lite::InnerContext *context,
                                                           TypeId prefer_data_type) {
  mindspore::kernel::KernelExec *kernel_exec = nullptr;
  auto op_type_str = node->op_type_;
  auto op_type = BaseOperatorPopulateRegistry::GetInstance()->TypeStrToType(op_type_str);
  auto parame_gen = BaseOperatorPopulateRegistry::GetInstance()->GetParameterCreator(op_type);
  if (parame_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr.";
    return nullptr;
  }
  OpParameter *op_parameter = parame_gen(node->base_operator_.get());
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, kNumberTypeInt32, NHWC, op_type, "", kernel::kBuiltin};
  auto ret = lite::KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context, nullptr, desc,
                                                                op_parameter, &kernel_exec, node->primitive_);
  if (ret != lite::RET_OK || kernel_exec == nullptr) {
    MS_LOG(ERROR) << "find lite kernel failed with code " << ret << ", node: " << node->name_
                  << ", type: " << node->op_type_;
    return nullptr;
  }
  return kernel_exec;
}

mindspore::lite::Tensor *MindirModel::ConvertTensor(TensorProtoWrap mindir_tensor_wrap) {
  auto mindir_tensor = mindir_tensor_wrap.tensor_proto();
  auto data_type = MindirModelUtil::ProtoTypeToTypeId(mindir_tensor.data_type());
  std::vector<int> shape;
  for (int i = 0; i < mindir_tensor.dims_size(); i++) {
    shape.push_back(mindir_tensor.dims(i));
  }
  auto format = Format::NCHW;
  mindspore::lite::NodeType node_type;
  if (mindir_tensor.has_raw_data() || mindir_tensor.has_external_data()) {
    node_type = mindspore::lite::NodeType_ValueNode;
  } else {
    node_type = mindspore::lite::NodeType_CNode;
  }
  auto category = TensorCategory(node_type, mindir_tensor.dims_size(), data_type, mindir_tensor.raw_data().size());
  auto *lite_tensor = new mindspore::lite::Tensor(data_type, shape, format, category);
  lite_tensor->set_tensor_name(mindir_tensor_wrap.name());
  if (this->LoadTensorData(lite_tensor, mindir_tensor) != RET_OK) {
    MS_LOG(WARNING) << "MindirModel: Convert tensor failed, load tensor data failed, tensor data will be empty.";
  }
  return lite_tensor;
}

int MindirModel::LoadTensorData(mindspore::lite::Tensor *lite_tensor, const mind_ir::TensorProto &mindir_tensor) {
  if (mindir_tensor.has_raw_data()) {
    return memcpy_s(lite_tensor->MutableData(), lite_tensor->Size(), mindir_tensor.raw_data().data(),
                    mindir_tensor.raw_data().size());
  }
  if (mindir_tensor.has_external_data()) {
    std::string file = this->GetModelPath() + "/" + mindir_tensor.external_data().location();
    // Read file
    std::basic_ifstream<char> fid(file, std::ios::in | std::ios::binary);
    if (!fid) {
      MS_LOG(ERROR) << "Open file '" << file << "' failed, please check the correct of the file.";
      return RET_OK;
    }
    fid.seekg(0, std::ios_base::end);
    size_t file_size = static_cast<size_t>(fid.tellg());
    fid.clear();
    fid.seekg(0);
    auto plain_data = std::make_unique<char[]>(file_size);
    constexpr uint8_t is_little_endian = 1;
    constexpr int byte_order_index = 0;
    fid.read(plain_data.get(), file_size);
    fid.close();
    // if byte order is not same return false
    if ((plain_data[byte_order_index] == is_little_endian) != common::IsLittleByteOrder()) {
      MS_LOG(ERROR) << "The byte order of export MindIr device and load MindIr device is not same!";
      return mindspore::lite::RET_ERROR;
    }
    const uint8_t *data = reinterpret_cast<const uint8_t *>(plain_data.get());
    auto ret =
      common::huge_memcpy(reinterpret_cast<uint8_t *>(lite_tensor->MutableData()), lite_tensor->Size(),
                          data + mindir_tensor.external_data().offset(), mindir_tensor.external_data().length());
    if (ret != 0) {
      MS_LOG(ERROR) << "Build parameter occur memcpy_s error.";
      return mindspore::lite::RET_OK;
    }
    return mindspore::lite::RET_OK;
  }
  return mindspore::lite::RET_NOT_SUPPORT;
}

int MindirModel::CheckTensorValid(lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (dst_tensor->data_type() == kObjectTypeTensorType) {
    return mindspore::lite::RET_OK;
  }
  if (dst_tensor->IsGraphInput() || dst_tensor->IsGraphOutput()) {
    return mindspore::lite::RET_OK;
  }
  if (dst_tensor->IsConst() == false && dst_tensor->data() != nullptr) {
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

void MindirModel::Free() {
  if (this->buf != nullptr) {
    delete[](this->buf);
    this->buf = nullptr;
  }
  auto nodes_size = this->graph_.all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->graph_.all_nodes_[i];
    auto *primitive_ptr = reinterpret_cast<ops::BaseOperator *>(const_cast<void *>(node->primitive_));
    delete primitive_ptr;
    node->primitive_ = nullptr;
  }
}

void MindirModel::Destroy() {
  Free();

  this->all_mindir_tensors_.clear();

  auto nodes_size = this->graph_.all_nodes_.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = this->graph_.all_nodes_[i];
    MS_ASSERT(node != nullptr);
    delete node;
  }
  this->graph_.all_nodes_.clear();

  auto sub_graph_size = this->graph_.sub_graphs_.size();
  for (size_t i = 0; i < sub_graph_size; ++i) {
    auto sub_graph = this->graph_.sub_graphs_[i];
    delete sub_graph;
  }
}
}  // namespace mindspore::infer::mindir
