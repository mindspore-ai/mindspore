/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_TF_BUFFER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_TF_BUFFER_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "dataset/engine/data_buffer.h"
#include "./example.pb.h"
#include "dataset/engine/datasetops/source/tf_client.h"

namespace mindspore {
namespace dataset {
// This TFBuffer is the buffer type for dealing with tf record data.
class TFBuffer : public DataBuffer {
 public:
  // constructor
  TFBuffer(uint32_t id,                    // In: The id for this buffer
           DataBuffer::BufferFlags flags,  // In: The flags for this buffer
           const std::shared_ptr<StorageClient>
             &storage_client);  // In: The storage client that is related to this buffer type

  // destructor
  ~TFBuffer() override;

  // Name: print()
  // Description: A function that prints info
  void Print(std::ostream &out,              // In: The output stream to print to
             bool show_all) const override;  // In: T/F if it should print everything

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const TFBuffer &tf_buffer) {
    tf_buffer.Print(out, false);  // Show meta info only
    return out;
  }

  // Name: load()
  // Description: populates the DataBuffer with data.
  //              Overrides base-class method.
  Status Load() override;

 private:
  std::ifstream cur_reader_;
  FileInfo cur_f_info_;

  std::shared_ptr<StorageClient> storage_client_;  // The storage client for populating the buffer initially.

  // Name: ParseSingleExample()
  // Description: Drives the calls to TFClient for fetching the tf_file info from
  //              the tf_file files.  Returns a single row of data from the tf_file
  //              files.
  Status ParseSingleExample(dataengine::Example *ptr);

  // Name: LoadFeature()
  // Description: Given the column type of the tf record and the values list,
  //              constructs the tensor and returns it.
  Status LoadFeature(const dataengine::Feature::KindCase &column_list_type,
                     const dataengine::Feature &column_values_list, const ColDescriptor &current_col,
                     std::shared_ptr<Tensor> *out_tensor);

  Status LoadBytesList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                       std::string *element_str);

  Status LoadFloatList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                       uint32_t *num_elements, std::unique_ptr<float[]> *float_array);

  Status LoadIntList(const ColDescriptor &current_col, const dataengine::Feature &column_values_list,
                     uint32_t *num_elements, std::unique_ptr<int64_t[]> *int_array);

  Status CreateTensorShapeForColumn(const ColDescriptor &current_col, uint32_t num_elements,
                                    TensorShape *current_shape);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_TF_BUFFER_H_
