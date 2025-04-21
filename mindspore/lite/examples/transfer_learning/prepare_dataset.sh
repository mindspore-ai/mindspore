#!/bin/bash

echo "=======Preparing Dataset======="
[ -d "dataset" ] && echo "dataset was already created" && exit 0
PLACES_DATA_PATH=$1
if [ ! -d ${PLACES_DATA_PATH}/val_256/ ]; then
  echo "The path" ${PLACES_DATA_PATH} "does not contain Places validation dataset. Please read the README file!" && exit 1
fi
class_id=0
sp="/-\|"
classes=("4" "98" "6" "7" "10" "15" "17" "70" "26" "30")
echo -n 'Prep class '
for class in "${classes[@]}"; do
  mkdir -p dataset/$class_id
  f=0
  i=1
  echo -n $(($class_id+1)) ' '
  cat scripts/places365_val.txt | grep -w ${class} | awk '{print $1}' | while read line
  do 
    printf "\b${sp:i++%${#sp}:1}"
    convert -colorspace RGB -gravity center -crop '224x224+0+0' ${PLACES_DATA_PATH}/val_256/$line dataset/$class_id/$f.bmp; 
    f=$(($f+1)); 
  done
  printf "\b"
  class_id=$(($class_id+1))
done
echo ' '
