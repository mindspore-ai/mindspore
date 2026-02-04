#include <iostream>
#include <map>
#include <cstdint>

extern "C" {

void MS_DbgOnStepBegin(uint32_t step, int rank, std::map<uint32_t, void *> ctx) {
  std::cout << "Call Mock function: MS_DbgOnStepBegin" << std::endl;
}

void MS_DbgOnStepEnd() { std::cout << "Call Mock function: MS_DbgOnStepEnd" << std::endl; }
}
