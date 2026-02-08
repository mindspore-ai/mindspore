/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/message_queue.h"

#include <string>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/msg.h>
#endif

#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/type_id.h"

namespace mindspore::dataset {
#if !defined(_WIN32) && !defined(_WIN64)
MessageQueue::MessageQueue(key_t key)
    : mtype_(0), shm_id_(-1), shm_size_(0), key_(key), release_flag_(true), state_(MessageState::kInit) {
  // create message queue id first
  int msg_queue_id = msgget(key, IPC_CREAT | 0600);
  if (msg_queue_id < 0) {
    MS_LOG(EXCEPTION) << "Create send message by key: " << std::to_string(key)
                      << " failed. Errno: " << std::to_string(errno);
  }
  msg_queue_id_ = msg_queue_id;

  auto ret = memset_s(err_msg_, kWorkerErrorMsgSize, 0, kWorkerErrorMsgSize);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memset_s failed. err code: " << std::to_string(ret);
  }
}

MessageQueue::MessageQueue(key_t key, int msg_queue_id)
    : mtype_(0),
      shm_id_(-1),
      shm_size_(0),
      key_(key),
      msg_queue_id_(msg_queue_id),
      release_flag_(true),
      state_(MessageState::kInit) {
  auto ret = memset_s(err_msg_, kWorkerErrorMsgSize, 0, kWorkerErrorMsgSize);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memset_s failed. err code: " << std::to_string(ret);
  }
}

MessageQueue::~MessageQueue() { ReleaseQueue(); }

void MessageQueue::SetReleaseFlag(bool flag) { release_flag_ = flag; }

void MessageQueue::ReleaseQueue() {
  if (release_flag_ && msg_queue_id_ != -1 && msgget(key_, kMsgQueuePermission) == msg_queue_id_) {
    if (msgctl(msg_queue_id_, IPC_RMID, 0) == -1) {
      MS_LOG(ERROR) << "Delete msg queue id: " << msg_queue_id_ << " failed.";
    }
    state_ = MessageState::kReleased;
    MS_LOG(INFO) << "Delete msg queue id: " << msg_queue_id_ << " success.";
    msg_queue_id_ = -1;
  }
}

Status MessageQueue::GetOrCreateMessageQueueID() {
  msg_queue_id_ = msgget(key_, kMsgQueuePermission);
  if (msg_queue_id_ < 0) {
    if (state_ != MessageState::kReleased) {
      // create message queue id
      int msg_queue_id_ = msgget(key_, IPC_CREAT | kMsgQueuePermission);
      if (msg_queue_id_ < 0) {
        RETURN_STATUS_UNEXPECTED("Create send message by key: " + std::to_string(key_) +
                                 " failed. Errno: " + std::to_string(errno));
      }
      MS_LOG(INFO) << "Create send message queue id: " << std::to_string(msg_queue_id_)
                   << " by key: " << std::to_string(key_) << " success.";
    } else {
      RETURN_STATUS_UNEXPECTED("Message queue key: " + std::to_string(key_) + " had been released. " +
                               "There may be an error in the independent dataset process, you can enable INFO log " +
                               "to view detailed error messages.");
    }
  }
  state_ = MessageState::kRunning;
  return Status::OK();
}

MessageState MessageQueue::MessageQueueState() { return state_; }

Status MessageQueue::MsgSnd(int64_t mtype, int shm_id, uint64_t shm_size) {
  if (msg_queue_id_ == -1) {
    RETURN_STATUS_UNEXPECTED("Exec msgsnd failed. Msg queue id: " + std::to_string(msg_queue_id_) + " invalid.");
  }
  mtype_ = mtype;
  if (shm_id != -1) {
    shm_id_ = shm_id;
    shm_size_ = shm_size;
  }
  if (msg_queue_id_ >= 0 && msgsnd(msg_queue_id_, this, sizeof(MessageQueue) - sizeof(mtype), 0) != 0) {
    if (msgget(key_, kMsgQueuePermission) < 0) {
      MS_LOG(INFO) << "The msg_queue_id: " << std::to_string(msg_queue_id_) << " had been released.";
      return Status::OK();
    }
    RETURN_STATUS_UNEXPECTED("Exec msgsnd failed. Msg queue id: " + std::to_string(msg_queue_id_) +
                             ", mtype: " + std::to_string(mtype) + ", shm_id: " + std::to_string(shm_id) +
                             ", shm_size: " + std::to_string(shm_size));
  }
  MS_LOG(DEBUG) << "Exec msgsnd success, msg queue id: " << msg_queue_id_ << ", mtype: " << mtype
                << ", shm_id: " << shm_id << ", shm_size: " << shm_size << ", err status: " << GetErrorStatus();
  return Status::OK();
}

Status MessageQueue::MsgRcv(int64_t mtype) {
  if (msg_queue_id_ >= 0 && msgrcv(msg_queue_id_, this, sizeof(MessageQueue) - sizeof(mtype), mtype, 0) <= 0) {
    if (msgget(key_, kMsgQueuePermission) < 0) {
      state_ = MessageState::kReleased;
      if (errno == kMsgQueueClosed) {  // the message queue had been closed
        MS_LOG(INFO) << "The msg_queue_id: " << std::to_string(msg_queue_id_) << " had been closed.";
      }
      if (errno == kMsgQueueInterrupted) {
        MS_LOG(INFO) << "The msg_queue_id: " << std::to_string(msg_queue_id_) << " had been interrupted.";
      }
      return Status::OK();
    }
    RETURN_STATUS_UNEXPECTED("Exec msgrcv failed. Msg queue id: " + std::to_string(msg_queue_id_) +
                             ", mtype: " + std::to_string(mtype) + ", errno: " + std::to_string(errno));
  }
  MS_LOG(DEBUG) << "Exec msgrcv success, msg queue id: " << msg_queue_id_ << ", mtype: " << mtype
                << ", shm_id: " << shm_id_ << ", shm_size: " << shm_size_ << ", err status: " << GetErrorStatus();
  return Status::OK();
}

int MessageQueue::MsgRcv(int64_t mtype, int msgflg) {
  auto ret = msgrcv(msg_queue_id_, this, sizeof(MessageQueue) - sizeof(mtype), mtype, msgflg);
  if (ret > 0) {
    MS_LOG(DEBUG) << "Exec msgrcv success, msg queue id: " << msg_queue_id_ << ", mtype: " << mtype
                  << ", shm_id: " << shm_id_ << ", shm_size: " << shm_size_ << ", err status: " << GetErrorStatus();
  }
  return ret;
}

// two error status case:
//
// case 1 - contains c layer info with cpp file and line number
//
// E    RuntimeError: Exception thrown from user defined Python function in dataset.
// E
// E    ------------------------------------------------------------------
// E    - Python Call Stack:
// E    ------------------------------------------------------------------
// E    map operation: [PyFunc] failed. The corresponding data file is: ../train-0000-of-0001.data. Error description:
// E    ValueError: Traceback (most recent call last):
// E      File "/home/user/mindspore/dataset/transforms/py_transforms_util.py", line 63, in compose
// E        args = transform(*args)
// E      File "/home/user/mindspore/dataset/transforms/transforms.py", line 85, in __call__
// E        return self._execute_py(*input_tensor_list)
// E      File "/home/user/mindspore/dataset/transforms/transforms.py", line 946, in _execute_py
// E        return util.random_choice(img, self.transforms)
// E      File "/home/user/mindspore/dataset/transforms/py_transforms_util.py", line 171, in random_choice
// E        return random.choice(transforms)(img)
// E      File "/home/user/mindspore/dataset/vision/transforms.py", line 97, in __call__
// E        return super().__call__(*input_tensor_list)
// E      File "/home/user/mindspore/dataset/transforms/transforms.py", line 85, in __call__
// E        return self._execute_py(*input_tensor_list)
// E      File "/home/user/mindspore/dataset/vision/transforms.py", line 4279, in _execute_py
// E        self.fill_value, Border.to_python_type(self.padding_mode))
// E      File "/home/user/mindspore/dataset/vision/py_transforms_util.py", line 490, in random_crop
// E        top, left, height, width = _input_to_factor(img, size)
// E      File "/home/user/mindspore/dataset/vision/py_transforms_util.py", line 472, in _input_to_factor
// E        raise ValueError("Crop size {} is larger than input image size {}.".format(size, (img_height, img_width)))
// E    ValueError: Crop size (5000, 5000) is larger than input image size (2268, 4032).
// E
// E    ------------------------------------------------------------------
// E    - Dataset Pipeline Error Message:
// E    ------------------------------------------------------------------
// E    [ERROR] Execute user Python code failed, check 'Python Call Stack' above.
// E
// E    ------------------------------------------------------------------
// E    - C++ Call Stack: (For framework developers)
// E    ------------------------------------------------------------------
// E    mindspore/ccsrc/minddata/dataset/engine/datasetops/map_op/map_job.h(57).
//
// ../../../build/package/mindspore/dataset/engine/iterators.py:260: RuntimeError
//
// case 2 - just python stack info
//
// E    RuntimeError: Exception thrown from user defined Python function in dataset.
// E
// E    ------------------------------------------------------------------
// E    - Python Call Stack:
// E    ------------------------------------------------------------------
// E    ZeroDivisionError: Traceback (most recent call last):
// E      File "/home/user/mindspore/dataset/transforms/py_transforms_util.py", line 199, in __call__
// E        result = self.transform(*args)
// E      File "/home/user/mindspore/tests/ut/python/dataset/test_formatted_exception.py", line 125, in batch_func
// E        fake_data = 1/zero
// E    ZeroDivisionError: division by zero
// E
// E    ------------------------------------------------------------------
// E    - Dataset Pipeline Error Message:
// E    ------------------------------------------------------------------
// E    [ERROR] Execute user Python code failed, check 'Python Call Stack' above.
//
// ../../../build/package/mindspore/dataset/engine/iterators.py:260: RuntimeError

Status MessageQueue::SerializeStatus(const Status &status) {
  // StatusCode : 4bytes
  // line_of_code: 4bytes
  // file_name : 4bytes + data
  // err_description : 4bytes + data
  auto ret = memset_s(err_msg_, kWorkerErrorMsgSize, 0, kWorkerErrorMsgSize);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == EOK, "memset_s failed. err code: " + std::to_string(ret));

  MS_LOG(INFO) << "Begin serialize status: " << status.ToString();

  // StatusCode
  int32_t offset = 0;
  auto status_code = status.StatusCode();
  auto ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &status_code, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the status code of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  // line_of_code
  int line_of_code = status.GetLineOfCode();
  ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &line_of_code, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the line number of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  if (line_of_code != -1) {
    // file_name
    const auto vec_file_name = status.GetFileName();
    std::string file_name(vec_file_name);
    if (offset + kFourBytes + vec_file_name.size() >= kWorkerErrorMsgSize) {
      file_name = file_name.substr(0, kWorkerErrorMsgSize - kFourBytes - offset);
    }
    int file_name_len = file_name.size();
    ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &file_name_len, kFourBytes);
    CHECK_FAIL_RETURN_UNEXPECTED(
      ret_code == EOK, "memcpy_s the file name length of Status failed. err code: " + std::to_string(ret_code));
    offset += kFourBytes;

    ret_code = memcpy_s(err_msg_ + offset, file_name_len, file_name.data(), file_name_len);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                 "memcpy_s the file name of Status failed. err code: " + std::to_string(ret_code));
    offset += file_name_len;

    if (vec_file_name.size() != file_name_len) {
      err_msg_[kWorkerErrorMsgSize - 1] = '\0';
      return Status::OK();
    }
  }

  // err_description
  auto vec_err_description = status.GetErrDescription();
  std::string err_description(vec_err_description);
  if (offset + kFourBytes + vec_err_description.size() >= kWorkerErrorMsgSize) {
    err_description = err_description.substr(0, kWorkerErrorMsgSize - kFourBytes - offset);
  }
  int err_description_len = err_description.size();
  ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &err_description_len, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(
    ret_code == EOK, "memcpy_s the err description len of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  ret_code = memcpy_s(err_msg_ + offset, err_description_len, err_description.data(), err_description_len);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the err description of Status failed. err code: " + std::to_string(ret_code));

  err_msg_[kWorkerErrorMsgSize - 1] = '\0';

  MS_LOG(INFO) << "End serialize status.";
  return Status::OK();
}

Status MessageQueue::SerializeStatus(const int32_t &status_code, const int32_t &line_of_code,
                                     const std::string &filename, const std::string &err_desc) {
  // StatusCode : 4bytes
  // line_of_code: 4bytes
  // file_name : 4bytes + data
  // err_description : 4bytes + data
  auto ret = memset_s(err_msg_, kWorkerErrorMsgSize, 0, kWorkerErrorMsgSize);
  CHECK_FAIL_RETURN_UNEXPECTED(ret == EOK, "memset_s failed. err code: " + std::to_string(ret));

  MS_LOG(INFO) << "Begin serialize status: " << err_desc;

  // StatusCode
  int32_t offset = 0;
  auto ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &status_code, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the status code of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  // line_of_code
  ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &line_of_code, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the line number of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  // file_name
  std::string file_name = filename;
  auto ori_file_name_len = file_name.size();
  if (offset + kFourBytes + static_cast<int32_t>(ori_file_name_len) >= kWorkerErrorMsgSize) {
    file_name = file_name.substr(0, kWorkerErrorMsgSize - kFourBytes - offset);
  }
  int file_name_len = file_name.size();
  ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &file_name_len, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the file name length of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  ret_code = memcpy_s(err_msg_ + offset, file_name_len, file_name.data(), file_name_len);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the file name of Status failed. err code: " + std::to_string(ret_code));
  offset += file_name_len;

  if (ori_file_name_len != file_name_len) {
    err_msg_[kWorkerErrorMsgSize - 1] = '\0';
    return Status::OK();
  }

  // err_description
  std::string err_description = err_desc;
  if (offset + kFourBytes + static_cast<int32_t>(err_description.size()) >= kWorkerErrorMsgSize) {
    err_description = err_description.substr(0, kWorkerErrorMsgSize - kFourBytes - offset);
  }
  int err_description_len = err_description.size();
  ret_code = memcpy_s(err_msg_ + offset, kFourBytes, &err_description_len, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(
    ret_code == EOK, "memcpy_s the err description len of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  ret_code = memcpy_s(err_msg_ + offset, err_description_len, err_description.data(), err_description_len);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the err description of Status failed. err code: " + std::to_string(ret_code));

  err_msg_[kWorkerErrorMsgSize - 1] = '\0';

  MS_LOG(INFO) << "End serialize status.";
  return Status::OK();
}

Status MessageQueue::DeserializeStatus() {
  StatusCode status_code = StatusCode::kSuccess;
  int line_of_code = -1;
  std::string file_name = "";
  std::string err_description = "";

  MS_LOG(INFO) << "Begin deserialize status.";

  // status_code
  int32_t offset = 0;
  auto ret_code = memcpy_s(&status_code, kFourBytes, err_msg_ + offset, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the status code of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  // line_of_code
  ret_code = memcpy_s(&line_of_code, kFourBytes, err_msg_ + offset, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the line number of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  if (line_of_code != -1) {
    // file_name
    int file_name_len = 0;
    ret_code = memcpy_s(&file_name_len, kFourBytes, err_msg_ + offset, kFourBytes);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                 "memcpy_s the file name len of Status failed. err code: " + std::to_string(ret_code));
    offset += kFourBytes;

    file_name.resize(file_name_len, '\0');
    ret_code = memcpy_s(file_name.data(), file_name_len, err_msg_ + offset, file_name_len);
    CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                                 "memcpy_s the file name of Status failed. err code: " + std::to_string(ret_code));
    offset += file_name_len;

    if (offset >= kWorkerErrorMsgSize) {
      return Status(status_code, line_of_code, file_name.c_str());
    }
  }

  // err_description
  int err_description_len = 0;
  ret_code = memcpy_s(&err_description_len, kFourBytes, err_msg_ + offset, kFourBytes);
  CHECK_FAIL_RETURN_UNEXPECTED(
    ret_code == EOK, "memcpy_s the err description len of Status failed. err code: " + std::to_string(ret_code));
  offset += kFourBytes;

  err_description.resize(err_description_len, '\0');
  ret_code = memcpy_s(err_description.data(), err_description_len, err_msg_ + offset, err_description_len);
  CHECK_FAIL_RETURN_UNEXPECTED(ret_code == EOK,
                               "memcpy_s the err description of Status failed. err code: " + std::to_string(ret_code));

  Status ret;
  if (line_of_code != -1) {
    ret = Status(status_code, line_of_code, file_name.c_str(), err_description);
  } else {
    ret = Status(status_code, err_description);
  }
  MS_LOG(INFO) << "End deserialize status: " << ret;
  return ret;
}

bool MessageQueue::GetErrorStatus() {
  int *status_code = reinterpret_cast<int *>(err_msg_);
  if (*status_code != 0) {
    return true;
  }
  return false;
}

void MessageQueue::ClearErrMsg() {
  auto ret = memset_s(err_msg_, kWorkerErrorMsgSize, 0, kWorkerErrorMsgSize);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memset_s failed. err code: " << std::to_string(ret);
  }
}
#endif
}  // namespace mindspore::dataset
