# !/usr/bin/python
# coding:utf-8
'''
将四张小图拼成一张大图，方便对比
重点介绍一下subplot(numRows, numCols, plotNum)
1、当三个参数都小于10的时候，逗号可以省略，于是plt.subplot(2, 2, 2) 等价于 plt.subplot(222)
2、numRows, numCols 表示我们要绘制一个多少行 多少列的图，plotNum是每个图位置的编号，从左往右，从上往下依次是1,2,3,...
3、当你需要多图组合时，你可以是用221, 222, 212这种类型 在212的时候假装只有两行一列的把程序骗过去
参考 https://blog.csdn.net/gatieme/article/details/61416645

重点介绍一下plot的颜色形状控制参数
fmt = '[color][marker][line]'
Colors
character	color
'b'	blue
'g'	green
'r'	red
'c'	cyan
'm'	magenta
'y'	yellow
'k'	black
'w'	white

Markers
character	description
'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
'1'	tri_down marker
'2'	tri_up marker
'3'	tri_left marker
'4'	tri_right marker
's'	square marker
'p'	pentagon marker
'*'	star marker
'h'	hexagon1 marker
'H'	hexagon2 marker
'+'	plus marker
'x'	x marker
'D'	diamond marker
'd'	thin_diamond marker
'|'	vline marker
'_'	hline marker

Line Styles
character	description
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style
参考文档https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
@ author: Pruce
@ email: 1756983926@qq.com
@ date: 2018年6月19日
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.xlabel('k') 
plt.ylabel('SSE')
X = range(2,12)
Scores = [56238672.62127426, 44838325.20241091, 29363214.9616623, 23554097.69106624, 17557936.87354927, 15480918.660410887, 17003830.32141292, 12921201.539286045, 11502310.046684416, 10169470.393877456]
plt.plot(X, Scores, '*-')
plt.title('20180716-3001-active')

plt.subplot(222)
plt.xlabel('k')
plt.ylabel('SSE')
X = range(2,12)
Scores = [266534374.66506726, 190634527.61865658, 138300855.64247516, 117261246.56215253, 90589297.72945915, 86352473.37537478, 74765571.3360506, 63392251.12648861, 50122673.761374004, 48680717.17391231]
plt.plot(X, Scores, 'o-')
plt.title('20180716-3001-deposit')

plt.subplot(223)
plt.xlabel('k')
plt.ylabel('SSE')
X = range(2,12)
Scores = [62601564.53508409, 49173480.12529594, 40716607.673059896, 35915690.26506331, 33183045.4523746, 31071662.653303925, 25615234.82365249, 24057477.97104432, 21760117.34216803, 20558646.901328206]
plt.plot(X, Scores, 'p-')
plt.title('20180716-4001-active')

plt.subplot(224)
plt.xlabel('k')
plt.ylabel('SSE')
X = range(2,12)
Scores = [258983524.82411453, 193096195.1842368, 151239604.68733484, 126481568.18275778, 108914189.20533328, 86924695.5555884, 74489099.73874772, 64553843.87808447, 56562800.120242044, 52727262.40811915]
plt.plot(X, Scores, 'p-')
plt.title('20180716-4001-deposit')

plt.show()