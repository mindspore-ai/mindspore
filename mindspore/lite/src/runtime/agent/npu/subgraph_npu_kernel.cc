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

#include "src/runtime/agent/npu/subgraph_npu_kernel.h"
#include <set>
#include "include/errorcode.h"
#include "src/runtime/agent/npu/npu_executor.h"
#include "include/graph/operator.h"
#include "include/graph/graph.h"
#include "src/tensor.h"
#include "include/graph/model.h"
#include "include/hiai_ir_build.h"
#include "include/HiAiModelManagerType.h"
#include "include/version.h"
#include "src/common/utils.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
#include "mindspore/lite/src/runtime/kernel/npu/npu_kernel.h"
#include "src/runtime/agent/npu/npu_manager.h"
namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

static std::set<mindspore::schema::PrimitiveType> npu_specific_weight_nodes = {
  schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_Conv2dTransposeFusion, schema::PrimitiveType_ScaleFusion,
  schema::PrimitiveType_BatchNorm,    schema::PrimitiveType_FullConnection,        schema::PrimitiveType_InstanceNorm};

SubGraphNpuKernel::~SubGraphNpuKernel() {
  subgraph_input_op_.clear();
  subgraph_output_op_.clear();
  out_tensor_sorted_.clear();
  for (auto op : op_buffer_) {
    delete op;
  }
  if (executor_ != nullptr) {
    delete executor_;
  }
  op_buffer_.clear();
}

std::shared_ptr<domi::ModelBufferData> SubGraphNpuKernel::BuildIRModel() {
  ge::Graph graph("NPUGraph");

  auto ret = BuildNPUInputOp();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build NPU input operator failed.";
    return nullptr;
  }
  ret = BuildNPUOutputOp();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build NPU output operator failed.";
    return nullptr;
  }
  graph.SetInputs(subgraph_input_op_).SetOutputs(subgraph_output_op_);
  ge::Model model(GetOMModelName(), mindspore::lite::Version());
  model.SetGraph(graph);
  domi::HiaiIrBuild ir_build;
  auto om_model_buff = std::make_shared<domi::ModelBufferData>();
  if (om_model_buff == nullptr) {
    MS_LOG(ERROR) << "OM model buffer is nullptr.";
    return nullptr;
  }
  if (!ir_build.CreateModelBuff(model, *om_model_buff)) {
    MS_LOG(ERROR) << "Create model buffer failed.";
    return nullptr;
  }
  if (!ir_build.BuildIRModel(model, *om_model_buff)) {
    MS_LOG(ERROR) << "Build IR model failed.";
    ir_build.ReleaseModelBuff(*om_model_buff);
    return nullptr;
  }
  return om_model_buff;
}

int SubGraphNpuKernel::Run() {
  return reinterpret_cast<lite::NPUExecutor *>(this->executor_)
    ->Run(in_tensors_, out_tensor_sorted_, out_nodes_, nodes_);
}

int SubGraphNpuKernel::BuildNPUInputOp() {
  int count = 0;
  subgraph_input_op_.clear();
  op_buffer_.clear();
  for (auto node : this->nodes_) {
    std::vector<ge::Operator *> node_input_op;
    for (auto in_tensor : node->in_tensors()) {
      if (IsSubGraphInputTensor(in_tensor)) {
        auto tensor_name = node->name() + "_" + std::to_string(count++);
        hiai::op::Data *data;
        data = mindspore::lite::ConverterToNPUData(in_tensor, tensor_name);
        subgraph_input_op_.push_back(*data);
        node_input_op.push_back(data);
        op_buffer_.push_back(data);
        continue;
      }

      bool is_weight_tensor = true;
      for (auto in_kernel : node->in_kernels()) {
        if (IsContain(in_kernel->out_tensors(), in_tensor)) {
          if (in_kernel->desc().arch == mindspore::kernel::kNPU) {
            // input come from npu
            auto npu_op = reinterpret_cast<NPUKernel *>(in_kernel)->GetNPUOp();
            if (npu_op != nullptr) {
              node_input_op.push_back(npu_op);
              is_weight_tensor = false;
              break;
            } else {
              MS_LOG(ERROR) << in_kernel->type_str() << "NPU Operator is nullptr.";
              return RET_ERROR;
            }
          } else {
            MS_LOG(ERROR) << "The input of the intermediate node comes from the CPU";
            return RET_ERROR;
          }
        }
      }

      // weight tensor
      if (is_weight_tensor) {
        if (npu_specific_weight_nodes.find(node->Type()) == npu_specific_weight_nodes.end()) {
          auto name = node->name() + "_" + std::to_string(count++);
          auto weight_const = new (std::nothrow) hiai::op::Const(node->name() + "_" + std::to_string(count++));
          if (weight_const == nullptr) {
            MS_LOG(ERROR) << "New weight const failed.";
            return RET_ERROR;
          }
          auto weight_tensor = mindspore::lite::ConverterToNPUTensor(in_tensor);
          weight_const->set_attr_value(weight_tensor);
          node_input_op.push_back(weight_const);
          op_buffer_.push_back(weight_const);
        }
      }
    }
    // set input to NPU
    int ret = reinterpret_cast<NPUKernel *>(node)->SetNPUInputs(node->in_tensors(), node->out_tensors(), node_input_op);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << node->name() << " set npu inputs failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

bool SubGraphNpuKernel::IsSubGraphInputTensor(lite::Tensor *inputs) { return IsContain(this->in_tensors(), inputs); }

std::vector<ge::Operator> SubGraphNpuKernel::GetNPUNodes(const vector<kernel::LiteKernel *> &nodes) {
  std::vector<ge::Operator> ops;
  ops.reserve(nodes.size());
  for (int i = 0; i < nodes.size(); i++) {
    ops.push_back(*reinterpret_cast<NPUKernel *>(nodes[i])->GetNPUOp());
  }
  return ops;
}

int SubGraphNpuKernel::BuildNPUOutputOp() {
  subgraph_output_op_.clear();
  subgraph_output_op_ = GetNPUNodes(out_nodes_);
  out_tensor_sorted_.resize(out_tensors_.size());
  int i = 0;
  for (auto node : out_nodes_) {
    for (auto tensor : node->out_tensors()) {
      if (std::find(out_tensors_.begin(), out_tensors_.end(), tensor) != out_tensors_.end())
        this->out_tensor_sorted_[i++] = tensor;
    }
  }
  if (subgraph_output_op_.empty()) {
    MS_LOG(ERROR) << "NPU subgraph output op is empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::string SubGraphNpuKernel::GetOMModelName() { return this->name_ + ".om"; }

int SubGraphNpuKernel::Init() {
  if (!is_compiled_) {
    name_ = "kNpuSubGraph" + std::to_string(npu_manager_->index());
    auto model_buffer_data = BuildIRModel();
    if (model_buffer_data == nullptr) {
      MS_LOG(ERROR) << "Build IR model failed.";
      return RET_ERROR;
    }

    MS_ASSERT(npu_manager_ != nullptr);

    npu_manager_->AddModel(model_buffer_data, GetOMModelName(), context_->GetNpuInfo().frequency_);

    executor_ = new (std::nothrow) mindspore::lite::NPUExecutor(GetOMModelName(), npu_manager_);

    if (executor_ == nullptr) {
      MS_LOG(ERROR) << "Create NPUExecutor failed.";
      return RET_ERROR;
    }
    is_compiled_ = true;
  }
  return RET_OK;
}

int SubGraphNpuKernel::Prepare() {
  if (executor_->Prepare(nodes_) != RET_OK) {
    MS_LOG(ERROR) << "NPU executor prepare failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
