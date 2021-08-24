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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include "schema/inner/model_generated.h"
#include "src/lite_kernel.h"
#include "src/lite_model.h"
#include "include/train/train_cfg.h"

namespace mindspore {
#ifndef _STUB
namespace schema {
struct CNodeT;
struct TensorT;
struct MetaGraphT;
}  // namespace schema
#endif
namespace lite {

class TrainExport {
 public:
  explicit TrainExport(const std::string file_name) : file_name_(file_name) {}
  virtual ~TrainExport();
  int ExportNet(const std::vector<mindspore::kernel::LiteKernel *> &kernels,
                const std::vector<mindspore::lite::Tensor *> &tensors, const std::vector<std::string> &output_names,
                const Model *model, QuantizationType quant_type);
  int ExportInit(const std::string model_name, std::string version);
  int SaveToFile();
  void set_connect(const std::unordered_map<size_t, size_t> &map) { connect_ = map; }
  int LoadModel(void *buf, size_t buf_size);
  int AddTransformNode();

 protected:
  virtual std::vector<uint8_t> CreateData(const mindspore::lite::Tensor *tensor);

 private:
  std::string file_name_;
  schema::MetaGraphT *meta_graph_ = nullptr;
  std::vector<size_t> out_idx_;
  std::map<size_t, size_t> remap_;
  std::unordered_map<size_t, size_t> connect_;  // connection map (backbone tenor id-> head tensor id)
  bool IsNodeNonDepend(const std::unique_ptr<schema::CNodeT> &node, const std::vector<size_t> &sinked_tensor_idxes);
  int TopologicalSort();
  void PrepareRemap(int offset);
  Model::Node *FindNode(const mindspore::kernel::LiteKernel *kernel, const Model *model);
  std::unique_ptr<schema::TensorT> CreateTensor(const Tensor *tensor, schema::Tensor *scTensor);
  std::unique_ptr<schema::CNodeT> CreateCNode(const mindspore::kernel::LiteKernel *kernel,
                                              std::vector<uint32_t> inputIndex, std::vector<uint32_t> outputIndex,
                                              const Model *model);
  int IsInputTensor(const schema::TensorT &t);
  int CreateAndAddCNode(const mindspore::kernel::LiteKernel *kernel, std::vector<uint32_t> inputIndex,
                        std::vector<uint32_t> outputIndex, const Model *model);
  std::unique_ptr<schema::CNodeT> CreateTransformNode(std::vector<uint32_t> inputIndex,
                                                      std::vector<uint32_t> outputIndex, size_t id);
  std::unique_ptr<schema::TensorT> CreateTransformTensor(size_t id);
  std::unique_ptr<schema::TensorT> CreateTransformConst(size_t last_id);
  int AddTransform();
  bool NeedQuantization(const mindspore::lite::Tensor *tensor);
  virtual int QuantTensorData(schema::TensorT *dest_tensor, const mindspore::lite::Tensor *src_tensor);
  mindspore::schema::QuantType GetNodeQuantType(const mindspore::kernel::LiteKernel *kernel);
  void TagQuantizedNodes();
  QuantizationType quant_type_;
};
};  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
