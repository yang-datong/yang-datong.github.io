#!/bin/bash
data=""
echo -e "\033[32m"

show(){
	clear
	echo -e "请按顺序回忆以下单词："
	echo -e $data | while read i;do echo "$i $RANDOM" ;done|sort -k2n|cut -d" " -f1
	echo -e "\n"
}

while true;do
	read -p "input: " tmp
	if [ "$tmp" == "" ];then
		continue
	elif [ "$tmp" == "dele" ];then
		read -p "dele:" d
		data=$(echo $data | sed "s/${d}//")
		show
		continue
	fi
	data="$data\n$tmp"
	show
done
