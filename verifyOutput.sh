#!/bin/bash
apt-get install -y bc
pip install pyglet gym
pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}
if ! mkdir tmp; then
    echo please delete tmp directory before running the script
    exit 0
fi
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`
flag=1
pushd tmp
python3 ../mountain_car.py "--task" "T1" "--train" "1"
python3 ../mountain_car.py "--task" "T2" "--train" "1"
python3 ../mountain_car.py "--task" "T1" "--train" "0"> out1
python3 ../mountain_car.py "--task" "T2" "--train" "0"> out2
if [ ! -f "T1.npy" ]; then
    echo "${red}T1.npy does not get generated.${reset}"
    flag=0
else
	echo "${green}T1.npy generated successfully.${reset}"
fi
if [ ! -f "T2.npy" ]; then
    echo "${red}T2.npy does not get generated.${reset}"
    flag=0
else
   echo "${green}T2.npy generated successfully.${reset}"
fi
if [ ! -f "T1.jpg" ]; then
    echo "${red}T1.jpg does not get generated.${reset}"
    flag=0
else
   echo "${green}T1.jpg generated successfully.${reset}"
fi
if [ ! -f "T2.jpg" ]; then
    echo "${red}T2.jpg does not get generated.${reset}"
    flag=0
else
   echo "${green}T2.jpg generated successfully.${reset}"
fi
var="$(wc -l < out1)"
if [ ! $var = '1' ]; then
	echo "${red}your code should print exactly 1 line${reset}"
	flag=0
fi
var="$(wc -l < out2)"
if [ ! $var = '1' ]; then
	echo "${red}your code should print exactly 1 line${reset}"
	flag=0
fi
while read line; 
do
echo $line
#echo $line | tr -d '\n'
#if [ "$line" -lt "-200" ]  || [ "$line" -gt "0" ] ; then
x1=$(echo "$line < -200" | bc)
x2=$(echo "$line > 0" | bc)
if [ $x1 -eq 1 ] || [ $x2 -eq 1 ] ; then
	echo "${red}generate output between -200 to 0${reset}"
	flag=0
fi
done < out1
while read line;
do
echo $line
#echo $line | tr -d '\n'
#if [ "$line" -lt "-200" ]  || [ "$line" -gt "0" ] ; then
x1=$(echo "$line < -200" | bc)
x2=$(echo "$line > 0" | bc)
if [ $x1 -eq 1 ] || [ $x2 -eq 1 ] ; then
	echo "${red}generate output between -200 to 0${reset}"
	flag=0
fi
done < out2
if [[ flag -eq 1 ]]; then
	echo "${green}verification successful${reset}"
else
	echo "${red}verification unsuccessful ${reset}"
fi
popd
rm -rf tmp
