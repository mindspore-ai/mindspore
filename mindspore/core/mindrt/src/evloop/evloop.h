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

#ifndef __LITEBUS_EVLOOP_H__
#define __LITEBUS_EVLOOP_H__

#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <functional>
#include <list>
#include <mutex>
#include <queue>
#include <sys/eventfd.h>
#include <semaphore.h>
#include "timer/duration.h"

namespace mindspore {

/*
 * max epoll set size
 */
constexpr auto EPOLL_SIZE = 4096;

/*
 * epoll event max size
 */
constexpr auto EPOLL_EVENTS_SIZE = 64;

typedef void (*EventHandler)(int fd, uint32_t events, void *data);

typedef struct EventData {
  EventHandler handler;
  void *data;
  int fd;

} EventData;

class EvLoop {
 public:
  EvLoop() {
    efd = -1;
    stopLoop = 0;
    queueEventfd = -1;
    loopThread = 0;
  };
  EvLoop(const EvLoop &) = delete;
  EvLoop &operator=(const EvLoop &) = delete;

  bool Init(const std::string &threadName);

  int AddFuncToEvLoop(std::function<void()> &&func);

  int AddFdEvent(int fd, uint32_t events, EventHandler handler, void *data);
  int ModifyFdEvent(int fd, uint32_t events);
  int DelFdEvent(int fd);
  void Finish();

  ~EvLoop();

  int EventLoopCreate(void);
  void StopEventLoop();

  void EventLoopDestroy();

  void EventFreeDelEvents();
  void AddDeletedEvents(EventData *eventData);
  int FindDeletedEvent(const EventData *tev);

  void HandleEvent(const struct epoll_event *events, int nevent);

  void DeleteEvent(int fd);
  EventData *FindEvent(int fd);
  void AddEvent(EventData *eventData);
  void CleanUp();

  int efd;
  int stopLoop;
  std::mutex loopMutex;
  sem_t semId;
  pthread_t loopThread;
  int queueEventfd;
  std::mutex queueMutex;
  std::queue<std::function<void()>> queue;

  std::mutex eventsLock;
  // fd,EventData
  std::map<int, EventData *> events;

  // Just to be safe, let's use a list to preserve deleted events rather than a map. Because the caller may
  // delete events on the same fd twice in once epoll_wait
  std::map<int, std::list<EventData *>> deletedEvents;
};

}  // namespace mindspore

#endif
