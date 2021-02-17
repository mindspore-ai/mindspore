BASE_DIR=$(realpath ../../../../)
APP:=bin/net_runner
MSLIB:=mindspore-lite
LMDLIB:=-lminddata-lite -ljpeg 
LHIAILIB:=-lhiai_ir_build  -lhiai_ir -lhiai
MSDIR:=$(realpath package-$(TARGET)/lib)

SRC:=src/net_runner.cc
OBJ:=$(SRC:.cc=.o)

CFLAGS := -Ofast -std=c++17  \
	-I . \
        -I ./msl \
        -I ./msl/minddata \
        -I ./msl/third_party/flatbuffers/include


ifeq ($(TARGET),arm64)
CXX :=  ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++
CFLAGS  += --target=aarch64-none-linux-android21 --gcc-toolchain=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot -fdata-sections -ffunction-sections 
LDFLAGS := --target=aarch64-none-linux-android21 --gcc-toolchain=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot -Wl,--gc-sections 
LDFLAGS += -L$(MSDIR) -l$(MSLIB) $(LMDLIB) -pthread -llog -latomic -lm  $(LHIAILIB)
else
CFLAGS += -g
LDFLAGS := -L$(MSDIR) -l$(MSLIB) $(LMDLIB) -lpthread -Wl,-rpath,$(MSDIR)
endif  
LD := ${CXX}


all:$(APP)

$(APP): $(OBJ) $(MSDIR)/lib$(MSLIB).so
	@mkdir -p bin
	$(LD) $(OBJ) $(LDFLAGS) -o $@

clean:
	rm -rf src/*.o bin/ 


mrproper: 
	rm -rf package* msl src/*.o bin/ model/*.mindir model/*.ms model/*.so model/converter_lite

%.o:%.cc
	$(CXX) $(CFLAGS) -c $< -o $@
