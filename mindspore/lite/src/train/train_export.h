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
#include "schema/inner/model_generated.h"
#include "src/lite_kernel.h"
#include "src/lite_model.h"

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
  TrainExport(const std::string file_name, const mindspore::lite::Model *model)
      : model_(model), file_name_(file_name) {}
  virtual ~TrainExport() {}
  int Export(const std::vector<mindspore::kernel::LiteKernel *> &kernels,
             const std::vector<mindspore::lite::Tensor *> &tensors, const std::vector<std::string> &output_names);

 protected:
  virtual std::vector<uint8_t> CreateData(const mindspore::lite::Tensor *tensor);

 private:
  const Model *model_;
  std::string file_name_;
  mindspore::lite::Model::Node *FindNode(const mindspore::kernel::LiteKernel *kernel);
  std::unique_ptr<schema::TensorT> CreateTensor(const mindspore::lite::Tensor *tensor, schema::Tensor *scTensor);
  std::unique_ptr<schema::CNodeT> CreateCNode(const mindspore::kernel::LiteKernel *kernel,
                                              std::vector<uint32_t> inputIndex, std::vector<uint32_t> outputIndex);

  bool NeedQuantization(const mindspore::lite::Tensor *tensor);
  virtual int QuantTensorData(schema::TensorT *dest_tensor, const mindspore::lite::Tensor *src_tensor);
  mindspore::schema::QuantType GetNodeQuantType(const mindspore::kernel::LiteKernel *kernel);
};
};  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_EXPORT_H_
