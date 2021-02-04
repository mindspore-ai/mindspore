/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/wait_post.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace dataset {
#pragma GCC visibility push(hidden)

class GeneratorOp : public PipelineOp, public RandomAccessOp {
 public:
  class Builder {
   public:
    /// Builder constructor.  Creates the builder object.
    /// \note No default args
    /// \return This is a constructor.
    Builder();

    ~Builder() = default;

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetGeneratorFunction(py::function generator_function) {
      build_generator_function_ = generator_function;
      return *this;
    }

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetColumnNames(const std::vector<std::string> &column_names) {
      build_column_names_ = column_names;
      return *this;
    }

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetColumnTypes(const std::vector<DataType> &column_types) {
      build_column_types_ = column_types;
      return *this;
    }

    /// Setter method.
    /// \return Builder setter method returns reference to the builder.
    Builder &SetPrefetchSize(int32_t prefetch_size) {
      build_prefetch_size_ = prefetch_size;
      return *this;
    }

    /// The builder "build" method creates the final object.
    /// \return shared_ptr to the new GeneratorOp object
    Status Build(std::shared_ptr<GeneratorOp> *);

   private:
    // The builder saves all GeneratorOp construction arguments internally.
    // The following are the arguments.
    py::function build_generator_function_;
    std::vector<std::string> build_column_names_;
    std::vector<DataType> build_column_types_;

    int32_t build_prefetch_size_ = 0;
    int32_t build_buffer_size_;
    int32_t build_op_connector_size_;

    Status SanityCheck();
  };

  GeneratorOp(py::function generator_function, std::vector<std::string> column_names,
              std::vector<DataType> column_types, int32_t prefetch_size, int32_t buffer_size, int32_t connector_size,
              std::shared_ptr<SamplerRT> sampler);

  ~GeneratorOp() = default;

  /// A print method typically used for debugging
  /// \param out - The output stream to write output to
  /// \param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;

  /// << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out - reference to the output stream being overloaded
  /// \param generator_op - reference to the GeneratorOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const GeneratorOp &generator_op) {
    generator_op.Print(out, false);
    return out;
  }

  /// Class functor operator () override.
  /// All DatasetOps operate by launching a thread (see ExecutionTree). This class functor will
  /// provide the master loop that drives the logic for performing the work.
  /// \return Status The status code returned
  Status operator()() override;

  /// Overrides base class reset method.  When an operator does a reset, it cleans up any state
  /// info from it's previous execution and then initializes itself so that it can be executed
  /// again.
  /// \return Status The status code returned
  Status Reset() override;

  /// Op name getter
  /// \return Name of the current Op
  std::string Name() const override { return "GeneratorOp"; }

 private:
  py::function generator_function_;
  std::vector<std::string> column_names_;
  std::vector<DataType> column_types_;
  int32_t prefetch_size_;
  int32_t buffer_size_;
  int64_t generator_counter_;

  py::object generator_;
  int32_t buffer_id_;

  WaitPost wp_;

  Status PyRowToTensorRow(py::object py_data, TensorRow *tensor_row);

  Status FillBuffer(TensorQTable *tt);

  /// Private function for computing the assignment of the column name map.
  /// \return - Status
  Status ComputeColMap() override;

  /// Initialize Sampler, calls sampler->Init() within
  /// \return Status The status code returned
  Status InitSampler();

  /// Create new Generator object from the generator function
  /// \return Status The status code returned
  Status CreateGeneratorObject();

  /// Initialize GeneratorOp
  /// \return Status The status code returned
  Status Init();
};

#pragma GCC visibility pop
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_GENERATOR_OP_H_
