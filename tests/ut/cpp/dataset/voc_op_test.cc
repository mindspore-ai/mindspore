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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common/common.h"
#include "common/utils.h"
#include "dataset/core/client.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/datasetops/source/voc_op.h"
#include "dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "dataset/util/de_error.h"
#include "dataset/util/path.h"
#include "dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

std::shared_ptr<BatchOp> Batch(int batch_size = 1, bool drop = false, int rows_per_buf = 2);

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

std::shared_ptr<VOCOp> CreateVOC(int64_t num_wrks, int64_t rows, int64_t conns, std::string path,
                                 bool shuf = false, std::unique_ptr<Sampler> sampler = nullptr,
                                 int64_t num_samples = 0, bool decode = false) {
  std::shared_ptr<VOCOp> so;
  VOCOp::Builder builder;
  Status rc = builder.SetNumWorkers(num_wrks).SetDir(path).SetRowsPerBuffer(rows)
                     .SetOpConnectorSize(conns).SetSampler(std::move(sampler))
                     .SetNumSamples(num_samples).SetDecode(decode).Build(&so);
  return so;
}

class MindDataTestVOCSampler : public UT::DatasetOpTesting {
 protected:
};
