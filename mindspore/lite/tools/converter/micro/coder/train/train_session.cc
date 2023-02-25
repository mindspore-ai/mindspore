/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/micro/coder/train/train_session.h"
#include <map>
#include <utility>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "include/errorcode.h"
#include "tools/converter/micro/coder/utils/train_utils.h"
#include "tools/converter/micro/coder/train/train_generator.h"

namespace mindspore::lite::micro {
int CoderTrainSession::Build() {
  int ret = CoderSession::Build();
  MS_CHECK_RET_CODE(ret, "code session build failed.");
  MS_CHECK_RET_CODE(CompileTrainCoders(), "CompileTrainCoders failed");
  MS_CHECK_RET_CODE(coder_graph_->CompileTrainOutputs(train_op_coders_), "CompileTrainOutputs failed!");
  MS_CHECK_RET_CODE(coder_graph_->CompileEvalOutputs(train_op_coders_), "CompileEvalOutputs failed!");
  MS_CHECK_RET_CODE(CompileEvalCoders(coder_graph_->GetEvalOutputsMap()), "CompileTrainCoders failed.");
  return RET_OK;
}

int CoderTrainSession::Run(const std::string model_name) {
  MS_LOG(INFO) << "start run op coders";
  int ret = Preprocess();
  MS_CHECK_RET_CODE(ret, "preprocess failed");

  ret = DoCode();
  MS_CHECK_RET_CODE(ret, "do code failed");

  PassArgsToContext(model_name);
  MS_LOG(INFO) << "run op coders success";
  return RET_OK;
}

int CoderTrainSession::GenerateCode() {
  MS_LOG(INFO) << "CoderSession::GenerateCode start";
  auto generator = std::make_shared<TrainGenerator>(std::move(context_), code_blocks_with_flag_);
  MS_CHECK_PTR(generator);

  int ret = generator->GenerateCode();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "generate code failed";
  }
  MS_LOG(INFO) << "CoderSession::GenerateCode done";
  return ret;
}
int CoderTrainSession::DoCode() {
  int ret = RET_OK;
  size_t last_idx = context_->code_blocks().size();
  for (const auto &op_coder : op_coders_) {
    MS_CHECK_PTR(op_coder);
    MS_LOG(DEBUG) << "code: " << op_coder->name();
    ret = op_coder->DoCode(this->context_.get());
    MS_CHECK_RET_CODE(ret, "do coder " << op_coder->name() << " failed");
    auto code_blocks = context_->code_blocks();
    auto cur_indx = code_blocks.size();
    MS_CHECK_TRUE_MSG(cur_indx > last_idx, RET_ERROR, "append code failed.");
    bool is_train_only =
      std::find(eval_op_coders_.begin(), eval_op_coders_.end(), op_coder.get()) == eval_op_coders_.end();
    for (; last_idx < cur_indx; last_idx++) {
      code_blocks_with_flag_.emplace_back(code_blocks.at(last_idx), is_train_only);
    }
  }
  return ret;
}

int CoderTrainSession::UpdateCodeBlocksWithFlag() {
  auto code_blocks = context_->code_blocks();
  MS_CHECK_TRUE_MSG(code_blocks.size() == code_blocks_with_flag_.size(), RET_ERROR, "code blocks size is unmatched.");
  for (size_t i = 0; i < code_blocks.size(); i++) {
    code_blocks_with_flag_.at(i).first = code_blocks.at(i);
  }
  return RET_OK;
}

int CoderTrainSession::PassArgsToContext(const std::string model_name) {
  int ret = CoderSession::PassArgsToContext(model_name);
  MS_CHECK_RET_CODE(ret, "PassArgsToContext failed");
  if (Configurator::GetInstance()->debug_mode()) {
    ret = UpdateCodeBlocksWithFlag();
    MS_CHECK_RET_CODE(ret, "update code_blocks_with_flag_ failed.");
  }
  context_->set_graph_train_outputs(coder_graph_->train_output_tensors());
  context_->set_graph_eval_outputs(coder_graph_->eval_output_tensors());
  context_->set_model_name(model_name);
  return ret;
}

void CoderTrainSession::FindEvalCoders(OperatorCoder *coder) {
  if (coder == nullptr) {
    return;
  }
  if (std::find(eval_op_coders_.begin(), eval_op_coders_.end(), coder) ==
      eval_op_coders_.end()) {  // kernel is not already in vector
    for (auto in_coder : coder->input_ops()) {
      FindEvalCoders(in_coder);
    }
    if (!IsLossCoder(coder)) {
      eval_op_coders_.emplace_back(coder);
    }
  }
}

int CoderTrainSession::CompileTrainCoders() {
  train_op_coders_.clear();
  (void)std::transform(op_coders_.begin(), op_coders_.end(), std::back_inserter(train_op_coders_),
                       [](const std::unique_ptr<OperatorCoder> &coder) { return coder.get(); });
  return RET_OK;
}

int CoderTrainSession::CompileEvalCoders(const std::map<std::string, std::vector<Tensor *>> &eval_outputs_map) {
  eval_op_coders_.clear();
  for (const auto &item : eval_outputs_map) {
    std::string kernel_name = item.first;
    auto iter = std::find_if(train_op_coders_.begin(), train_op_coders_.end(),
                             [&kernel_name](const OperatorCoder *coder) { return (coder->name() == kernel_name); });
    MS_CHECK_TRUE_MSG(iter != train_op_coders_.end(), RET_ERROR, "can't find output coder in Eval mode.");
    MS_CHECK_TRUE_MSG(*iter != nullptr, RET_ERROR, "find output coder in Eval mode.");
    (void)FindEvalCoders(*iter);
  }
  if (eval_op_coders_.empty()) {
    eval_op_coders_ = train_op_coders_;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::micro
