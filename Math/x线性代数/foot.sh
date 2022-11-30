#!/bin/bash
for i in ./*.gif;do
	ffmpeg -i $i -loop 1 demo/$i
done

