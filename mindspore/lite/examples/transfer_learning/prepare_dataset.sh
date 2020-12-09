#!/bin/bash

echo "=======Preparing Dataset======="
[ -d "dataset" ] && echo "dataset was already created" && exit
echo "Preparing dataset" 
PLACES_DATA_PATH=$1
class_id=0
classes=("4" "98" "6" "7" "10" "15" "17" "70" "26" "30")
for class in "${classes[@]}"; do
  mkdir -p dataset/$class_id
  i=0
  cat scripts/places365_val.txt | grep -w ${class} | awk '{print $1}' | while read line
  do 
    echo converting ${PLACES_DATA_PATH}/val_256/$line to bmp
    convert -colorspace RGB -gravity center -crop '224x224+0+0' ${PLACES_DATA_PATH}/val_256/$line dataset/$class_id/$i.bmp; 
    i=$(($i+1)); 
  done
  class_id=$(($class_id+1))
done
