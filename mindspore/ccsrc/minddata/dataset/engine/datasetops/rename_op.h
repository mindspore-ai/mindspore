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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// forward declare
class DataBuffer;

class RenameOp : public PipelineOp {
 public:
  //  The nested builder class inside of the RenameOp is used to help manage all of
  //  the arguments for constructing it.  Use the builder by setting each argument
  //  with the provided set methods, and then finally call the build method to execute
  //  the actual construction.
  class Builder {
   public:
    // Builder constructor.  Creates the builder object.
    // @note No default args
    // @return This is a constructor.
    Builder();

    // Default destructor
    ~Builder() = default;

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetInColNames(const std::vector<std::string> &in_col_names) {
      builder_in_columns_ = in_col_names;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOutColNames(const std::vector<std::string> &out_col_names) {
      builder_out_columns_ = out_col_names;
      return *this;
    }

    // Setter method.
    // @return Builder setter method returns reference to the builder.
    Builder &SetOpConnectorSize(int32_t op_connector_size) {
      builder_op_connector_size_ = op_connector_size;
      return *this;
    }

    // The builder "build" method creates the ZipOp dataset Operator.
    // @return shared_ptr to the new RenameOp object
    Status Build(std::shared_ptr<RenameOp> *);

   private:
    std::vector<std::string> builder_in_columns_;
    std::vector<std::string> builder_out_columns_;
    int32_t builder_op_connector_size_;

    Status SanityCheck() const;
  };

  // Constructor for RenameOp
  // @param in_col_names names of columns to rename
  // @param out_col_names names of columns after rename
  // @param op_connector_size connector size
  RenameOp(const std::vector<std::string> &in_col_names,   // In: Col names to consume
           const std::vector<std::string> &out_col_names,  // In: Col names to produce
           int32_t op_connector_size);

  // Destructor
  ~RenameOp();

  Status EofReceived(int32_t) override;

  Status EoeReceived(int32_t) override;

  // Print function for Rename
  // @param out output stream to print to
  // @param show_all if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RenameOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kRenameOp; }

 protected:
  // Rename core functionality
  // Computing the assignment of the new column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Variable to store the input column names
  std::vector<std::string> in_columns_;

  // Variable to store the output column names
  std::vector<std::string> out_columns_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
