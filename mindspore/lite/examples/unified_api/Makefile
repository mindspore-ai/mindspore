BASE_DIR=$(realpath ../../../../)
APP:=bin/net_runner
INF_APP:=bin/infer
LMSTLIB:=-lmindspore-lite-train -lminddata-lite
LMSLIB:=-lmindspore-lite
MSDIR:=$(realpath package-$(TARGET)/lib)
ifneq ("$(wildcard $(MSDIR)/libhiai.so)","")
  LHIAILIB:=-lhiai_ir_build  -lhiai_ir -lhiai
else
  LHIAILIB:=
endif

SRC:=src/net_runner.cc
OBJ:=$(SRC:.cc=.o)

INF_SRC:=src/inference.cc
INF_OBJ:=$(INF_SRC:.cc=.o)

CFLAGS := -Ofast -std=c++17  \
	-I . \
        -I ./msl/runtime \
        -I ./msl/runtime/minddata \
        -I ./msl/tools/third_party/flatbuffers/include


ifeq ($(TARGET),arm64)
CXX :=  ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
CFLAGS  += --target=aarch64-none-linux-android21 --gcc-toolchain=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fdata-sections -ffunction-sections 
LDFLAGS := --target=aarch64-none-linux-android21 --gcc-toolchain=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot -Wl,--gc-sections 
LDFLAGS += -L$(MSDIR) $(LMSLIB) -pthread -llog -latomic -lm  $(LHIAILIB) -Wl,-rpath,$(MSDIR)
else
CFLAGS += -g
LDFLAGS := -L$(MSDIR) $(LMSLIB) -lpthread -Wl,-rpath,$(MSDIR)
endif  
LD := ${CXX}


all:$(APP) $(INF_APP)

$(APP): $(OBJ)
	@mkdir -p bin
	$(LD) $(OBJ) $(LMSTLIB) $(LDFLAGS) -o $@

$(INF_APP): $(INF_OBJ)
	@mkdir -p bin
	$(LD) $(INF_OBJ) $(LDFLAGS) -o $@


clean:
	rm -rf src/*.o bin/ 


mrproper: 
	rm -rf package* msl src/*.o bin/ model/*.mindir model/*.ms model/*.so* model/converter_lite

%.o:%.cc
	$(CXX) $(CFLAGS) -c $< -o $@
