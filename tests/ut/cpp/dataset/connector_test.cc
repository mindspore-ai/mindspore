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

#include <fcntl.h>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>


#include "common/common.h"
#include "minddata/dataset/engine/connector.h"
#include "minddata/dataset/util/task_manager.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestConnector : public UT::Common {
public:

  MindDataTestConnector();

  // Test scenario: single producer, single consumer.
  // This means there is only one queue in the connector.
  Status Run_test_0();

  // Test scenario: multiple producers, multiple cosumers
  // A chain of three layer of thread groups connected by two Connectors between
  // two layer. You can set different num of threads on layer 1 and 2, and layer 3
  // that does the serialization to _ouput vector needs to be single thread.
  // A random sleep/delay can be introduced for each thread. See run().
  Status Run_test_1();

  void SetSleepMilliSec(uint32_t ms) { sleep_ms_ = ms; }

private:
  std::unique_ptr<TaskGroup> tg_;
  uint32_t last_input_;
  uint32_t sleep_ms_ = 0;
  std::vector<uint32_t> input_;
  WaitPost wp;

  // This worker loop is to be called by a single thread. It will pop my_conn Connector
  // and populate output vector
  Status SerialWorkerPull(
                          int tid,
                          std::shared_ptr<Connector<uint32_t>> my_conn,
                          std::vector<uint32_t> *output
                          );

  // This worker loop read from input_ vector that have complete list of tasks/elements.
  // The assignment from the elements in input_ to each worker is ensured in RoundRobin,
  // i.e., tid-0 will pick input_[0], tid-1 will pick input_[1], so-on circular.
  Status FirstWorkerPush(
                         int tid,
                         std::shared_ptr<Connector<uint32_t> > my_conn,
                         int start_in,
                         int offset);

  // This worker loop read from a Connector and put the result into another Connector.
  Status MidWorkerJob(
                      int tid,
                      std::shared_ptr<Connector<uint32_t> > from_conn,
                      std::shared_ptr<Connector<uint32_t> > to_conn);

  Status ValidateOutput(const std::vector<uint32_t> &output);

  uint32_t GenRand(int max);

  // Put the current thread to sleep mode for MaxDue milliseconds.
  // (Imitating nondeterministic processing time)
  void GoToSleep(int max_dur);
};


// Test0 : single producer, single consumer which means there is only one queue in the connector
TEST_F(MindDataTestConnector, Test0) {
  MS_LOG(INFO) << "MindDataTestConnector Test0: single producer, single consumer.";
  Status rc = this->Run_test_0();
  ASSERT_TRUE(rc.IsOk());
  rc = TaskManager::GetMasterThreadRc();
  ASSERT_TRUE(rc.IsOk());
}

// Test1: multiple producers, multiple consumers without random delay
// A chain of three layer of thread groups connected by two Connectors between
// two layer.
TEST_F(MindDataTestConnector, Test1) {
  MS_LOG(INFO) << "MindDataTestConnector Test1.";
  Status rc = this->Run_test_1();
  ASSERT_TRUE(rc.IsOk());
  rc = TaskManager::GetMasterThreadRc();
  ASSERT_TRUE(rc.IsOk());
}

// Test1: multiple producers, multiple consumers with random delay after push/pop
// A chain of three layer of thread groups connected by two Connectors between
// two layer.
TEST_F(MindDataTestConnector, Test2) {
  MS_LOG(INFO) << "MindDataTestConnector Test2.";
  this->SetSleepMilliSec(30);
  Status rc = this->Run_test_1();
  ASSERT_TRUE(rc.IsOk());
  rc = TaskManager::GetMasterThreadRc();
  ASSERT_TRUE(rc.IsOk());
}



// Implementation of MindDataTestConnector class and the helper functions.
MindDataTestConnector::MindDataTestConnector() : tg_(new TaskGroup()) {
  last_input_ = 150;
  for (int i = 1; i <= last_input_; i++) {
    input_.push_back(i);
  }
  wp.Register(tg_.get());
}

Status MindDataTestConnector::Run_test_0() {
  Status rc;
  std::vector<uint32_t> output;
  wp.Clear();
  auto my_conn = std::make_shared<Connector<uint32_t>>(1,  // num of producers
                                                      1,  // num of consumers
                                                      10);  // capacity of each queue
  MS_ASSERT(my_conn != nullptr);

  rc = my_conn->Register(tg_.get());
  RETURN_IF_NOT_OK(rc);

  // Spawn a thread to read input_ vector and put it in my_conn
  rc = tg_->CreateAsyncTask("Worker Push",
                            std::bind(&MindDataTestConnector::FirstWorkerPush,
                                      this,  // passing this instance
                                      0,  // id = 0 for this simple one to one case
                                      my_conn,  // the connector
                                      0,  // start index to read from the input_ list
                                      1));  // the offset to read the next index
  RETURN_IF_NOT_OK(rc);

  // Spawn another thread to read from my_conn and write to _output vector.
  rc = tg_->CreateAsyncTask("Worker Pull",
                            std::bind(&MindDataTestConnector::SerialWorkerPull,
                                      this,
                                      0,
                                      my_conn,
                                      &output));
  RETURN_IF_NOT_OK(rc);
  // Wait for the threads to finish.
  rc = wp.Wait();
  EXPECT_TRUE(rc.IsOk());
  tg_->interrupt_all();
  tg_->join_all(Task::WaitFlag::kNonBlocking);
  my_conn.reset();
  return ValidateOutput(output);
}

Status MindDataTestConnector::Run_test_1() {
  std::vector<uint32_t> output;
  Status rc;
  wp.Clear();

  // number of threads in each layer
  int l1_threads = 15;
  int l2_threads = 20;
  int l3_threads = 1;

  // Capacity for the first and second connectors
  int conn1_qcap = 5;
  int conn2_qcap = 10;

  auto conn1 = std::make_shared<Connector<uint32_t>>(l1_threads,  // num of producers
                                                     l2_threads,  // num of consumers
                                                     conn1_qcap);  // the cap of each queue

  auto conn2 = std::make_shared<Connector<uint32_t>>(l2_threads,
                                                     l3_threads,
                                                     conn2_qcap);

  rc = conn1->Register(tg_.get());
  RETURN_IF_NOT_OK(rc);
  rc = conn2->Register(tg_.get());
  RETURN_IF_NOT_OK(rc);

  // Instantiating the threads in the first layer
  for (int i = 0; i < l1_threads; i++) {
    rc = tg_->CreateAsyncTask("First Worker Push",
                              std::bind(&MindDataTestConnector::FirstWorkerPush,
                                        this,  // passing this instance
                                        i,  // thread id in this group of thread
                                        conn1,  // the connector
                                        i,  // start index to read from the input_ list
                                        l1_threads));   // the offset to read the next index
    RETURN_IF_NOT_OK(rc);
  }

  // Instantiating the threads in the 2nd layer
  for (int i = 0; i < l2_threads; i++) {
    rc = tg_->CreateAsyncTask("Mid Worker Job",
                              std::bind(&MindDataTestConnector::MidWorkerJob,
                                        this,  // passing this instance
                                        i,  // thread id in this group of thread
                                        conn1,  // the 1st connector
                                        conn2));    // the 2nd connector
    RETURN_IF_NOT_OK(rc);
  }

  // Last layer doing serialization to one queue to check if the order is preserved
  rc = tg_->CreateAsyncTask("Worker Pull",
                            std::bind(&MindDataTestConnector::SerialWorkerPull,
                                      this,
                                      0,  // thread id = 0, since it's the only one
                                      conn2,  // popping the data from conn2
                                      &output));
  RETURN_IF_NOT_OK(rc);
  // Wait for the threads to finish.
  rc = wp.Wait();
  EXPECT_TRUE(rc.IsOk());
  tg_->interrupt_all();
  tg_->join_all(Task::WaitFlag::kNonBlocking);
  conn1.reset();
  conn2.reset();

  return ValidateOutput(output);
}

Status MindDataTestConnector::SerialWorkerPull(
                                               int tid,
                                               std::shared_ptr<Connector<uint32_t>> my_conn,
                                               std::vector<uint32_t> *output
                                       ) {
  Status rc;
  TaskManager::FindMe()->Post();
  while (1) {
    uint32_t res;
    rc = my_conn->Pop(tid, &res);
    RETURN_IF_NOT_OK(rc);

    output->push_back(res);

    // Emulate different processing time for each thread
    if (sleep_ms_ != 0) {
      GoToSleep(sleep_ms_);
    }

    // Signal master thread after it processed the last_input_.
    // This will trigger the MidWorkerJob threads to quit their worker loop.
    if (res == last_input_) {
      MS_LOG(INFO) << "All data is collected.";
      wp.Set();
      break;
    }
  }
  return Status::OK();
}

Status MindDataTestConnector::FirstWorkerPush(
                                      int tid,
                                      std::shared_ptr<Connector<uint32_t> > my_conn,
                                      int start_in,
                                      int offset)  {
  TaskManager::FindMe()->Post();
  MS_ASSERT(my_conn != nullptr);
  Status rc;
  for (int i = start_in; i < input_.size(); i += offset) {
    rc = my_conn->Push(tid, input_[i]);

    // Emulate different processing time for each thread
    if (sleep_ms_ != 0)
      GoToSleep(sleep_ms_);
  }
  return Status::OK();
}

// This worker loop read from a Connector and put the result into another Connector.
Status MindDataTestConnector::MidWorkerJob(
                                   int tid,
                                   std::shared_ptr<Connector<uint32_t> > from_conn,
                                   std::shared_ptr<Connector<uint32_t> > to_conn) {
  MS_ASSERT((from_conn != nullptr) && (to_conn != nullptr));
  Status rc;
  TaskManager::FindMe()->Post();
  while (1) {
    uint32_t el;
    rc = from_conn->Pop(tid, &el);
    RETURN_IF_NOT_OK(rc);

    // Emulate different processing time for each thread
    if (sleep_ms_ != 0) {
      GoToSleep(sleep_ms_);
    }
    rc = to_conn->Push(tid, el);
    RETURN_IF_NOT_OK(rc);
  }
  return Status::OK();
}

Status MindDataTestConnector::ValidateOutput(const std::vector<uint32_t> &output) {
  int prev = 0;
  for (auto el : output) {
    if (prev >= el) {
      return Status(StatusCode::kMDUnexpectedError, "Output vector are not in-order.");
    }
    prev = el;
  }
  return Status::OK();
}

uint32_t MindDataTestConnector::GenRand(int max) {
  uint32_t r_int = 0;
  if (max == 0) {
    return r_int;
  }

  // open urandom not random
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd > 0) {
    if (read(fd, &r_int, sizeof(uint32_t)) != sizeof(uint32_t)) {
      r_int = max / 2;
    }
  }
  (void)close(fd);  // close it!

  return r_int % max;
}

// Put the current thread to sleep mode for MaxDue milliseconds.
// (Imitating nondeterministic processing time)
void MindDataTestConnector::GoToSleep(int max_dur) {
  uint32_t duration = GenRand(max_dur);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration));
}
