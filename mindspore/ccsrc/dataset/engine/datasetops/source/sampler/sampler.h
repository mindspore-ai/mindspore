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
#ifndef DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SAMPLER_H_
#define DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SAMPLER_H_

#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "dataset/core/tensor.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/data_schema.h"
#include "dataset/engine/datasetops/dataset_op.h"

namespace mindspore {
namespace dataset {
//  RandomAccessOp is a base class that all data-producing leaf operators
//  must inherit from if those leaf operator wish to support sampling.
class RandomAccessOp {
 public:
  // Sampler get numRows from StorageOp
  // @param int64_t num - return number of rows, normally num of samples
  // @return - The error code return
  virtual Status GetNumSamples(int64_t *num_samples) const {
    // CI complains num_samples not used if the following line is not added
    CHECK_FAIL_RETURN_UNEXPECTED(num_samples != nullptr, "num_samples == nullptr");
    RETURN_STATUS_UNEXPECTED("function GetNumSamples needs to overridden to support this sampler");
  }

  // Sampler get number of rows in the dataset!
  // @param int64_t num - return number of rows for this dataset
  // @return - The error code return
  virtual Status GetNumRowsInDataset(int64_t *num_rows) const {
    // CI complains num_rows not used if the following line is not added
    CHECK_FAIL_RETURN_UNEXPECTED(num_rows != nullptr, "num_rows == nullptr");
    RETURN_STATUS_UNEXPECTED("function GetNumRowsInDataset needs to overridden to support this sampler");
  }

  // sampler gets label , imageIds from storageOp, this function is unique to PK
  // @param std::map<int64_t, std::vector<int64_t>> * map
  // @return - The error code return
  virtual Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *map) const {
    RETURN_STATUS_UNEXPECTED("GetClassIds needs to be override to support PK");
  }

  // default destructor
  virtual ~RandomAccessOp() = default;
};

class Sampler : public DatasetOp {
 public:
  // @param int64_t samplesPerBuffer: Num of Sampler Ids to fetch via 1 GetNextBuffer call
  explicit Sampler(int64_t samples_per_buffer = std::numeric_limits<int64_t>::max());

  // default destructor
  ~Sampler() = default;

  // Get a list of sample ids.
  // @note It is Sampler responsibility to make sure that the id is not out of bound.
  // @param std::unique_ptr<DataBuffer> pBuffer - Buffer to be returned to StorageOp
  // @param int32_t workerId - not meant to be used
  // @return - The error code return
  Status GetNextBuffer(std::unique_ptr<DataBuffer> *out_buffer) override = 0;

  // return all ids in one epoch as a numpy array, then call reset
  Status GetAllIdsThenReset(py::array *data);

  // for next epoch of sampleIds
  // @return - The error code return
  Status Reset() override = 0;

  // setter function for num_rows_
  Status SetNumRowsInDataset(int64_t num_rows);

  // setter function for num_samples_
  Status SetNumSamples(int64_t num_samples);

  int64_t num_samples() { return num_samples_; }

  // first handshake between StorageOp and Sampler. This func will call getNumRows and getNumSamples
  // @param op - StorageOp pointer, pass in so Sampler can call getNumSamples() and get ClassIds()
  // @return
  virtual Status HandshakeRandomAccessOp(const RandomAccessOp *op);

  // initialize sampler and perform checks on certain vars
  virtual Status InitSampler() { return Status::OK(); }

  // Not meant to be called
  // @return
  int32_t num_workers() const final { return 0; }

  // Not meant to be called
  // @return
  int32_t num_consumers() const final { return 0; }

  // Not meant to be called
  // @return
  int32_t num_producers() const final { return 0; }

  // Not meant to be called!
  // @return - The error code return
  Status operator()() final { RETURN_STATUS_UNEXPECTED("Functor not supported in Sampler"); }

  // Adds a sampler to become our child.
  // @param std::shared_ptr<DatasetOp> - The sampler to add as a child.
  // @return - The error code returned.
  Status AddChild(std::shared_ptr<DatasetOp> child);

  // A helper function to create a int64_t 1-D Tensor specifically used to hold sampleIds for Sampler
  // @param std::shared_ptr<Tensor>* sampleIds
  // @param int64_t numElements - must be a non 0 number
  // @return - The error code returned.
  Status CreateSamplerTensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements);

  void Print(std::ostream &out, bool show_all) const override;

  friend std::ostream &operator<<(std::ostream &out, const Sampler &sampler) {
    sampler.Print(out, false);
    return out;
  }

  // Checks if this sampler has a child sampler.
  // @return - tre if there is a child sampler, false otherwise.
  bool HasChildSampler();

  // Uses id as an index for the list of ids generated by the child sampler, and gets the
  // associated id.
  // @param int64_t* out_associated_id - Out parameter, contains the associated id.
  // @param int64_t id - The id used as an index to get the associated child id.
  // @return - The error code returned.
  Status GetAssociatedChildId(int64_t *out_associated_id, int64_t id);

 protected:
  // Number of rows of data from the place this sampler is sampling from. If this sampler
  // has a child sampler, num_rows_ is the number of ids the child sampler will
  // output. Otherwise, num_rows_ is the number of rows in the dataset.
  int64_t num_rows_;

  // Number of ids this sampler will return.
  int64_t num_samples_;

  // The max number of ids a DataBuffer returned by this sampler will contain.
  int64_t samples_per_buffer_;
  std::unique_ptr<ColDescriptor> col_desc_;
  std::unique_ptr<DataBuffer> child_ids_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SAMPLER_H_
