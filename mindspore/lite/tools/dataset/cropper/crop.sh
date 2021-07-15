#!/bin/bash

usage()
{
  echo "Usage:"
  echo "bash crop.sh -p <path-to-mindspore-directory> <source-file> [<more-source-files>] \\"
  echo "bash crop.sh -h \\"
  echo ""
  echo "Options:"
  echo "    -p path to mindspore directory"
  echo "    -h print usage"

}

# check and set options
checkopts()
{
  while getopts ':p:h' opt
  do
    case "${opt}" in
      p)
        MINDSPORE_PATH="$(cd "${OPTARG}" &> /dev/null && pwd )"
        ;;
      h)
        usage
        exit 1
        ;;
      *)
        echo "Unknown option: \"${OPTARG}\""
        usage
        exit 1
    esac
  done
}

checkopts "$@"

# exit if less than 3 args are given by user
if [ $# -lt 3 ]; then
  usage
  exit 1
fi

# exit if mindspore path is not given by user
if [ -z "${MINDSPORE_PATH}" ]; then
  echo -e "\e[31mPlease set MINDSPORE_PATH using -p flag.\e[0m"
  exit 1
fi

ORIGINAL_PATH="$PWD"
FILE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# getting absolute paths for user provided filenames
USER_CODES=""
for i in "${@:OPTIND}";
do
  USER_CODES+="$(cd "$(dirname "${i}" )" &> /dev/null && pwd )/$(basename "${i}") "
done
# exit if user has not given any argument as their code
if [ -z "${USER_CODES}" ]; then
  echo -e "\e[31mPlease provide your file names as arguments.\e[0m"
  exit 1
fi
echo "Provided files: $USER_CODES"

echo "MS PATH: $MINDSPORE_PATH"
echo "CWD: $ORIGINAL_PATH"
echo "File PATH: $FILE_PATH"


cd $FILE_PATH || exit

MD_LIB_FILENAME="libminddata-lite.a"

# locate original MindData lite library
MD_LIB_PATH=`find $MINDSPORE_PATH -name "${MD_LIB_FILENAME}" | head -n 1`
if [ -z "${MD_LIB_PATH}" ]; then
  echo -e "\e[31mMindData lite static library could not be found.\e[0m"
  cd $ORIGINAL_PATH || exit
  exit 1
fi


# extract all objects of static lib to tmp/
mkdir -p tmp
cp $MD_LIB_PATH tmp
cd tmp || exit
# extract objects with identical names by prepending (one or more) '_' to their names
# (this scruipt supports more than 2 duplicate filenames)
DUPLICATES=`ar t "${MD_LIB_FILENAME}" | sort | uniq -d`
for dup in $DUPLICATES;
do
  i=0
  prepend_var="_"
  while :
  do
    i=$((i + 1))
    # check if more duplicates are available (break otherwise)
    error_output=$(ar xN $i "${MD_LIB_FILENAME}" $dup  2>&1)
    if [ -n "$error_output" ]; then
      break
    fi
    mv $dup "${prepend_var}${dup}"
    prepend_var="${prepend_var}_"
  done
done

# extract unique files from static library
UNIQUES=`ar t "${MD_LIB_FILENAME}" | sort | uniq -u`
ar x "${MD_LIB_FILENAME}" ${UNIQUES}
cd ..

# remove unused object files
# write needed depsendencies to tmp/needed_dependencies.txt
python build_lib.py ${USER_CODES}
retVal=$?
if [ $retVal -ne 0 ]; then
  cd $ORIGINAL_PATH || exit
  exit 1
fi

LD_SEP='\n'
EX_SEP=$';'
LD_PATHS=""
EXTERNAL_DEPS=""

# locate external dependencies for MindData lite
LIBJPEG_PATH=`find $MINDSPORE_PATH -name "libjpeg.so*" | head -n 1`
LIBTURBOJPEG_PATH=`find $MINDSPORE_PATH -name "libturbojpeg.so*" | head -n 1`
LIBSECUREC_PATH=`find $MINDSPORE_PATH -name libsecurec.a | head -n 1`

# resolve symbolc links
if [ "$(uname)" == "Darwin" ]; then
  c=$(file -b "$(readlink $LIBJPEG_PATH)")
elif [ "$(expr substr "$(uname -s)" 1 5)" == "Linux" ]; then
  c=$(file -b "$(readlink -f $LIBJPEG_PATH)")
fi
# detect system architecture
IFS="," read -r -a array <<< "$c"
TARGET_ARCHITECTURE=${array[1]##* }
echo "Architecture: $TARGET_ARCHITECTURE"

# exit if $ANDROID_NDK is not set by user for ARM32 or ARM64
if [ "$TARGET_ARCHITECTURE" == "ARM64" ]; then
  if [ -z "${ANDROID_NDK}" ]; then
    echo -e "\e[31mPlease set ANDROID_NDK environment variable.\e[0m"
    cd $ORIGINAL_PATH || exit
    exit 1
  fi
elif [ "$TARGET_ARCHITECTURE" == "ARM32" ]; then
  if [ -z "${ANDROID_NDK}" ]; then
    echo -e "\e[31mPlease set ANDROID_NDK environment variable.\e[0m"
    cd $ORIGINAL_PATH || exit
    exit 1
  fi
  # add LIBCLANG_RT_LIB for ARM32
  LIBCLANG_RT_LIB=`find $ANDROID_NDK -name libclang_rt.builtins-arm-android.a | head -n 1`
  EXTERNAL_DEPS=${EXTERNAL_DEPS}${LIBCLANG_RT_LIB}${EX_SEP}
else
  echo "No need for ANDROID_NDK"
fi
# Note: add .a files only to EXTERNAL_DEPS.
if grep -q 'jpeg' "tmp/needed_dependencies.txt"; then
  LD_PATHS=${LD_PATHS}${LIBJPEG_PATH}${LD_SEP}
  LD_PATHS=${LD_PATHS}${LIBTURBOJPEG_PATH}${LD_SEP}
  EXTERNAL_DEPS=${EXTERNAL_DEPS}${LIBJPEG_PATH}${EX_SEP}
  EXTERNAL_DEPS=${EXTERNAL_DEPS}${LIBTURBOJPEG_PATH}${EX_SEP}
fi
# we always need securec library
EXTERNAL_DEPS=${EXTERNAL_DEPS}${LIBSECUREC_PATH}${EX_SEP}

# create .so lib from remaining object files
cmake -S . -B . \
      -DEXTERNAL_DEPS="${EXTERNAL_DEPS}" \
      -DARCHITECTURE=$TARGET_ARCHITECTURE

# no dependencies to MindData lite
retVal=$?
if [ $retVal -eq 0 ]; then
  make
  echo -e "\e[32mLibrary was built successfully, The new list of MindData-related dependencies is as follows:\e[0m"
  echo -e "\e[36m$LD_PATHS$PWD/libminddata-lite_min.so\e[0m"
fi

rm -rf tmp/

cd $ORIGINAL_PATH || exit
