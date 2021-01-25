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
#include "debug/rdr/somas_recorder.h"
#include "backend/optimizer/somas/somas.h"
namespace mindspore {
void SomasRecorder::Export() {
  if (filename_.empty()) {
    filename_ = directory_ + module_ + "_" + tag_;
  }
  std::string filename = filename_ + "_somas_after_allocate_" + std::to_string(graph_id_) + "_" + timestamp_ + ".ir";
  somas_reuse_util_ptr_->DumpSomasInfoIR(filename);
  std::string mem_filename = filename_ + "_somas_mem_info_" + std::to_string(graph_id_) + "_" + timestamp_ + ".ir";
  somas_reuse_util_ptr_->DumpSomasMemoryIR(mem_filename);
}

bool SomasRecorder::GenString() { return true; }
}  // namespace mindspore
