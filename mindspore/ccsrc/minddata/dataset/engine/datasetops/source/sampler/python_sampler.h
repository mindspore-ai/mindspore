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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_

#include <limits>
#include <memory>

#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace mindspore {
namespace dataset {
class PythonSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param num_samples - the number of samples to draw.  Value of 0 means to sample all of the
  //                      data from the dataset.
  // @param py_sampler_instance - the python instance of the sampler
  // @param int64_t samples_per_tensor - Num of Sampler Ids to fetch via 1 GetNextSample call
  explicit PythonSamplerRT(int64_t num_samples, py::object py_sampler_instance,
                           int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~PythonSamplerRT() = default;

  // Initialize the sampler.
  // @return Status
  Status InitSampler() override;

  // for next epoch of sampleIds
  // @return Status The status code returned
  Status ResetSampler() override;

  // Op calls this to get next Sample that contains all the sampleIds
  // @param TensorRow to be returned to corresponding Dataset Op
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(TensorRow *out) override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

 private:
  bool need_to_reset_;  // Whether Reset() should be called before calling GetNextSample()

  py::object py_sampler_instance;  // The handle to the py_sampler python object
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_PYTHON_SAMPLER_H_
