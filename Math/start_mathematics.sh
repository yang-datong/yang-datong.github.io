#!/bin/bash

cd ~/Desktop/demo1
#1. open demo.md
./foot.py k
#2. open pdf \ open video
if [ ! -n "$1" ];then
	open *.pdf
	./foot.py i
else
	echo " "
fi

#3. open markdown_preview
./foot.py vim
nvim -c 'MarkdownPreview' ./w微分/demo3.md
