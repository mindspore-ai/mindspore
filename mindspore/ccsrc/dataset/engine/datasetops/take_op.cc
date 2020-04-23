/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <utility>

#include "common/utils.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/take_op.h"
#include "dataset/engine/db_connector.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {
// Builder constructor. Creates the builder object.
TakeOp::Builder::Builder(int32_t count) : build_max_takes_(count) {}

Status TakeOp::Builder::SanityCheck() const {
  if (build_max_takes_ <= 0) {
    std::string err_msg("Take count must be greater than 0.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

// The builder "build" method creates the final object.
Status TakeOp::Builder::Build(std::shared_ptr<TakeOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<TakeOp>(build_max_takes_);
  return Status::OK();
}

// Constructor of the TakeOp.
TakeOp::TakeOp(int32_t count) : PipelineOp(0), max_takes_(count), take_count_(0) {}

// A print method typically used for debugging
void TakeOp::Print(std::ostream &out, bool show_all) const {
  // Call base class printer first
  PipelineOp::Print(out, show_all);

  // Then display our own stuff
  out << "TakeOp:"
      << "\nCurrent take count: " << take_count_ << "\nMax take count: " << max_takes_;
}

// This function will be call muti times to returns the buffer, when meet required max take count or meet
// EOF buffer then this will stop.
Status TakeOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
  if (child_.empty()) {
    RETURN_STATUS_UNEXPECTED("TakeOp can't be the leaf node.");
  }

  std::unique_ptr<DataBuffer> buf;

  bool last_repeat = !BitTest(op_ctrl_flags_, kDeOpRepeated) || BitTest(op_ctrl_flags_, kDeOpLastRepeat);
  if (take_count_ == max_takes_) {
    if (state_ == OpState::kDeOpRunning) {
      MS_LOG(INFO) << "meet max count and push-back eoe buffer.";
      auto eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
      *p_buffer = std::move(eoe_buffer);
      state_ = OpState::kDeOpIdle;

      // Reset the count and drain
      if (!last_repeat) {
        take_count_ = 0;
        RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, true));
        while (!buf->eoe() && !buf->eof()) {
          RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, true));
        }
      }
    } else {
      MS_LOG(INFO) << "meet max count and push-back eof buffer.";
      auto eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
      *p_buffer = std::move(eof_buffer);
      take_count_ = 0;
    }
    return Status::OK();
  }
  RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&buf, worker_id, true));
  // Loop until non EOE is received
  if (buf->eoe()) {
    take_count_ = 0;
    *p_buffer = std::move(buf);
    return Status::OK();
  }

  // Check if the last buf is next eof
  if (buf->eof()) {
    *p_buffer = std::move(buf);
    return Status::OK();
  }

  // Get buffer and push back when take_count is still small
  if (take_count_ < max_takes_) {
    RETURN_IF_NOT_OK(FillBuffer(&buf, p_buffer));
  }
  return Status::OK();
}

// Function FillBuffer mainly prepare the buffer for returning
Status TakeOp::FillBuffer(std::unique_ptr<DataBuffer> *buffer, std::unique_ptr<DataBuffer> *data_buffer) {
  int32_t buffer_size = (*buffer)->NumRows();
  if (take_count_ + buffer_size < max_takes_) {
    *data_buffer = std::move(*buffer);
    take_count_ = take_count_ + buffer_size;
  } else {
    MS_LOG(INFO) << "In last buffer: Push one buffer.";
    std::unique_ptr<TensorQTable> new_tensor_table = std::make_unique<TensorQTable>();
    while (take_count_ < max_takes_) {
      TensorRow new_row;
      RETURN_IF_NOT_OK((*buffer)->PopRow(&new_row));
      take_count_++;
      new_tensor_table->push_back(new_row);
    }
    (*buffer)->set_tensor_table(std::move(new_tensor_table));
    *data_buffer = std::move(*buffer);
  }
  return Status::OK();
}

// Class functor operator () override.
// Most dataset ops operate by launching a thread (see ExecutionTree).
// However, the TakeOp is defined as a inlined operator, so it is invalid to launch the
// functor since this op runs inlined inside another operator.  The function is overloaded to
// ensure that it is not called by mistake (it will generate an error).
Status TakeOp::operator()() { RETURN_STATUS_UNEXPECTED("Logic error. TakeOp is an inlined operator."); }

Status TakeOp::PrepareNodePostAction() {
  RETURN_IF_NOT_OK(PipelineOp::PrepareNodePostAction());
  tree_->AddToRepeatStack(shared_from_this());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
