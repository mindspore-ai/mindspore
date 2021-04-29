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
#define _STUB
#include "src/train/train_export.h"
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <utility>
#include <map>
#include <set>
#include "schema/inner/model_generated.h"
#include "src/train/train_utils.h"

namespace mindspore {
namespace lite {

std::vector<uint8_t> TrainExport::CreateData(const mindspore::lite::Tensor *tensor) {
  uint8_t *tensor_data = reinterpret_cast<uint8_t *>(tensor->data_c());
  auto size = tensor->Size();
  std::vector<uint8_t> data(tensor_data, tensor_data + size);
  return data;
}

std::unique_ptr<schema::TensorT> TrainExport::CreateTensor(const mindspore::lite::Tensor *tensor,
                                                           schema::Tensor *scTensor) {
  auto tensorT = std::make_unique<schema::TensorT>();
  tensorT->nodeType = scTensor->nodeType();
  tensorT->dataType = tensor->data_type();
  tensorT->dims = tensor->shape();
  tensorT->format = tensor->format();
  tensorT->name = tensor->tensor_name();
  tensorT->refCount = 0;
  tensorT->offset = 0;
  tensorT->enableHuffmanCode = false;
  if ((tensorT->nodeType == NodeType_ValueNode) && (scTensor->data() != nullptr) && (scTensor->data()->size() > 0)) {
    tensorT->data = CreateData(tensor);
  }
  for (auto quant_param : tensor->quant_params()) {
    auto quantParamT = std::make_unique<schema::QuantParamT>();
    quantParamT->scale = quant_param.scale;
    quantParamT->zeroPoint = quant_param.zeroPoint;
    quantParamT->min = 0;
    quantParamT->max = 0;
    quantParamT->narrowRange = true;
    quantParamT->numBits = quant_param.bitNum;
    quantParamT->inited = quant_param.inited;
    quantParamT->varCorr = quant_param.var_corr;
    quantParamT->meanCorr = quant_param.mean_corr;
    quantParamT->dstDtype = quant_param.dstDtype;
    quantParamT->roundType = quant_param.roundType;
    quantParamT->multiplier = quant_param.multiplier;
    tensorT->quantParams.emplace_back(std::move(quantParamT));
  }
  tensorT->quantClusters = tensor->quant_clusters();
  return tensorT;
}

mindspore::lite::Model::Node *TrainExport::FindNode(const mindspore::kernel::LiteKernel *kernel) {
  auto nodes = model_->all_nodes_;
  auto it = std::find_if(nodes.begin(), nodes.end(),
                         [&kernel](mindspore::lite::Model::Node *n) { return (kernel->name() == n->name_); });
  if (it == nodes.end()) {
    return nullptr;
  }
  return *it;
}

std::unique_ptr<schema::CNodeT> TrainExport::CreateCNode(const mindspore::kernel::LiteKernel *kernel,
                                                         std::vector<uint32_t> inputIndex,
                                                         std::vector<uint32_t> outputIndex) {
  auto cnodeT = std::make_unique<schema::CNodeT>();
  cnodeT->inputIndex = inputIndex;
  cnodeT->outputIndex = outputIndex;
  cnodeT->name = kernel->name();
  cnodeT->quantType = schema::QuantType_QUANT_NONE;
  // find kernel in model
  auto *node = FindNode(kernel);
  if (node == nullptr) {
    MS_LOG(ERROR) << "cannot find kernel " + kernel->name() + " in model";
    return nullptr;
  }
  auto primitive = reinterpret_cast<schema::Primitive *>(const_cast<void *>(node->primitive_));
  cnodeT->primitive = std::unique_ptr<schema::PrimitiveT>(primitive->UnPack());
  return cnodeT;
}

int TrainExport::Export(const std::vector<mindspore::kernel::LiteKernel *> &kernels,
                        const std::vector<mindspore::lite::Tensor *> &tensors,
                        const std::vector<std::string> &output_names) {
  std::map<size_t, size_t> remap;
  std::vector<size_t> map_index;
  std::set<size_t> out_set;
  int tensor_idx = 0;
  auto meta_graph = std::make_unique<schema::MetaGraphT>();
  meta_graph->fmkType = 3;
  meta_graph->name = model_->name_;
  meta_graph->version = model_->version_;
  for (const auto kernel : kernels) {
    std::vector<uint32_t> in_idx, out_idx;
    for (const auto tensor : kernel->in_tensors()) {
      size_t id = TSFindTensor(tensors, tensor);
      if (id == tensors.size()) {
        MS_LOG(ERROR) << "cannot find tensor " + tensor->ToString() + " in model";
        return RET_ERROR;
      }
      auto it = remap.find(id);
      if (it == remap.end()) {
        remap[id] = tensor_idx;
        in_idx.push_back(tensor_idx);
        map_index.push_back(id);
        tensor_idx++;
      } else {
        in_idx.push_back(it->second);
      }
    }
    for (const auto tensor : kernel->out_tensors()) {
      size_t id = TSFindTensor(tensors, tensor);
      if (id == tensors.size()) {
        MS_LOG(ERROR) << "cannot find tensor " + tensor->ToString() + " in model";
        return RET_ERROR;
      }
      out_set.insert(id);
      auto it = remap.find(id);
      if (it == remap.end()) {
        remap[id] = tensor_idx;
        map_index.push_back(id);
        out_idx.push_back(tensor_idx);
        out_set.insert(tensor_idx);
        tensor_idx++;
      } else {
        out_idx.push_back(it->second);
        out_set.insert(it->second);
      }
    }
    auto cnode = CreateCNode(kernel, in_idx, out_idx);
    meta_graph->nodes.emplace_back(std::move(cnode));
  }
  for (auto id : map_index) {
    mindspore::lite::Tensor *tensor = tensors.at(id);
    schema::Tensor *scTensor = model_->all_tensors_.at(id);
    auto tensorT = CreateTensor(tensor, scTensor);
    // find a tensor which is not an output
    if (out_set.find(id) == out_set.end()) {
      if ((tensorT->nodeType == NodeType_ValueNode) && (tensorT->data.size() == 0)) {
        meta_graph->inputIndex.push_back(remap[id]);
      }
    }
    // find output tensor
    if (std::find(output_names.begin(), output_names.end(), tensor->tensor_name()) != output_names.end()) {
      meta_graph->outputIndex.push_back(remap[id]);
    }
    meta_graph->allTensors.emplace_back(std::move(tensorT));
  }
  auto graph = meta_graph.release();
  int err = SaveToFile(graph, file_name_);
  if (err != RET_OK) {
    MS_LOG(ERROR) << "failed to save flatbuffer file " << file_name_;
  }
  delete graph;
  return err;
}

int TrainExport::SaveToFile(const schema::MetaGraphT *graph, const std::string &outputPath) {
  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, graph);
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  int size = builder.GetSize();
  auto content = builder.GetBufferPointer();
  if (content == nullptr) {
    MS_LOG(ERROR) << "GetBufferPointer nullptr";
    return RET_ERROR;
  }
  if (access((outputPath + ".ms").c_str(), F_OK) == 0) {
    chmod((outputPath + ".ms").c_str(), S_IWUSR);
  }
  std::ofstream output(outputPath + ".ms", std::ofstream::binary);
  if (!output.is_open()) {
    MS_LOG(ERROR) << "Can not open output file: " << outputPath << ".ms";
    return RET_ERROR;
  }
  output.write((const char *)content, size);
  output.close();
  chmod((outputPath + ".ms").c_str(), S_IRUSR);
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore
