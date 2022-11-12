[toc]

**视频**

- 图解： https://www.bilibili.com/video/BV1Qs41167bP/?spm_id_from=333.788.recommend_more_video.-1&vd_source=1ec51cb8123536a0bf872aa061240412
- 李永乐： https://www.bilibili.com/cheese/play/ep49651?csource=Hp_searchresult

# 线性代数

## 向量

- $\displaystyle{向量表示:\left[ \begin{matrix}a \\ b\end{matrix} \right] }$

- $\displaystyle{线性代数中向量的箭头起点位于原点}$

> 二维

<img src="./img1.png"  width=70% />

> 三维

<img src="./img2.png"  width=70% />

### 向量加法

<img src="./img3.png"  width=70% />

<img src="./img4.png"  width=70% />

$\displaystyle{公式如下:}$

$$\left[\begin{matrix} x_1 \\ y_1 \\ \end{matrix}\right] +  \left[\begin{matrix} x_2 \\ y_2 \\ \end{matrix}\right] = \left[\begin{matrix} x_1 + x_2 \\ y_1 + y_2 \\ \end{matrix}\right]$$

### 向量乘法

<img src="./img5.png"  width=70% />

$\displaystyle{公式如下:}$

$$2 \cdot \left[\begin{matrix} x \\ y \\ \end{matrix}\right] = \left[\begin{matrix} 2x \\ 2y \\ \end{matrix}\right]$$

## 线性组合、张成的空间与基

- $\displaystyle{i帽：一个指向正\textcolor{red}{右方}，长度为1的单位向量}$

- $\displaystyle{j帽：一个指向正\textcolor{red}{上方}，长度为1的单位向量}$

<img src="./img6.png"  width=70% />

- $\displaystyle{两个向量标量乘法之和的结果称为这两个向量的线性组合}$

$$\vec{v}与\vec{w}的线性组合=> a\vec{v} + b\vec{w}$$

### 张成的空间

- $\displaystyle{两个向量的张成空间:向量之间的运算获得的所有可能的向量集合}$

- $\displaystyle{\textcolor{red}{给定向量}张成的空间:所有可以表示为\textcolor{red}{给定向量线性组合}的向量集合}$

* $\displaystyle{大部分二维向量来说张成空间是所有二维向量的集合}$

<img src="./img7.png"  width=70% />

- $\displaystyle{当向量共线时张成空间是终点落在一条直线上的向量集合}$

<img src="./img8.png"  width=70% />

- $\displaystyle{三维空间中两个向量张成的空间}$

$\displaystyle{\textcolor{red}{拓展}：当第三个向量没有落在前两个向量张成的空间中，那么进行缩放会得到整个三维空间}$`

<img src="./img9.png"  width=70% />

## 线性变换

$\displaystyle{概念如下:}$

- $\displaystyle{线性有关: 多个向量中移除一个而不减小张成的空间}$

- $\displaystyle{线性无关: 所有向量都给张成的空间增添了新的维度}$

- $\displaystyle{向量空间的一个基: 张成该空间的一个线性无关向量的集合}$

<br>

$\displaystyle{一个变化称为线性变化的\textcolor{red}{满足条件}:}$

1. $\displaystyle{直线在变换后仍然保持为直线}$

2. $\displaystyle{原点必须保持固定}$

<img src="./img10.gif" alt="./img10.gif" width=70% />

$\displaystyle{可以理解为保持网格线平行并等距离分布的变换称为线性变换}$

<font color="#83D1DD"> $\displaystyle{结论:}$ </font>

$\displaystyle{线性变换是将向量作为输入和输出的一类函数}$

### 数学表示线性变换位置

$\displaystyle{存在线性变换如下:}$

<img src="./img12.gif" alt="./img12.gif" width=70% />

$\displaystyle{只需要记录两个基向量(i、j帽)落脚后的位置}$

$\displaystyle{向量\left[\begin{matrix} -1 \\ 2 \\ \end{matrix}\right] 线性变换前: }$
$$\vec v =  -1i + 2j$$

$\displaystyle{从变换网格线可以看到变换后的i帽为\left[\begin{matrix} 1 \\ -2 \\ \end{matrix}\right] ,j帽为\left[\begin{matrix} 3 \\ 0 \\ \end{matrix}\right]}$

<font color="#83D1DD"> $\displaystyle{线性变换后的:}$</font>

$$
\begin{aligned}
&Transformed \quad  \vec{v} =  -1 (Transformed \quad i)+ 2(Transformed \quad j)\\
& = -1 \left[\begin{matrix} 1 \\ -2 \\ \end{matrix}\right] + 2\left[\begin{matrix} 3 \\ 0 \\ \end{matrix}\right] \\
& = \left[\begin{matrix} -1(1) &+ &2(3) \\ -1(-2) &+ &2(0) \\ \end{matrix}\right] \\
& = \left[\begin{matrix} 5 \\ 2 \\ \end{matrix}\right] \\
&  \\
\end{aligned}
$$

## 矩阵

$\displaystyle{二维线性变换后的i帽、j帽组成的坐标包装在2 \times 2的格子中称它为\textcolor{red}{矩阵}(Matrix)}$

$\displaystyle{矩阵代表一个特定的线性变换}$

<img src="./img13.png" alt="./img13.png" width=70% />

#### 矩阵乘法

$$\left[\begin{matrix} a & b \\ c & d \\ \end{matrix}\right] \cdot  \left[\begin{matrix} x \\ y \\ \end{matrix}\right] = x \left[\begin{matrix} a \\ c \\ \end{matrix}\right] + y \left[\begin{matrix} b \\ d \\ \end{matrix}\right] = \left[\begin{matrix} ax + by \\ cx + dy \\ \end{matrix}\right]$$

#### 复合变换

$\displaystyle{如下对整个平面逆时针旋转90度，再进行一次剪切}$

<img src="./img14.gif" alt="./img14.gif" width=70% />

$\displaystyle{计算表示为如下:}$

$$\underbrace{\left[\begin{matrix} 1 & 1 \\ 0 & 1 \\ \end{matrix}\right]}_{剪切矩阵} \bigg(\underbrace {\left[\begin{matrix} 0 & -1 \\ 1 & 0 \\ \end{matrix}\right] }_{旋转矩阵}\left[\begin{matrix} x \\ y \\ \end{matrix}\right] \bigg)$$

$\displaystyle{矩阵乘积需要\textcolor{red}{从右往左}读,即平面先应用右侧矩阵的变换再往左侧应用变换}$

`和复合函数的记号一样比如f(g(x)) 是先计算g(x)然后才是f(x)`

<font color="#83D1DD"> $\displaystyle{结论:}$</font>

$\displaystyle{两个矩阵相乘的几何意义就是两个线性变换的相继作用}$

### 矩阵 \* 矩阵

$\displaystyle{存在如下矩阵:\underbrace{\left[\begin{matrix} 0&2 \\ 1&0 \\ \end{matrix}\right]}_{M_2} \underbrace{\left[\begin{matrix} 1&-2 \\ 1&0 \\ \end{matrix}\right]}_{M_1}= ?}$

:one: $\displaystyle{根据定义i帽的新坐标由M_1的第一类给出，也就是(1,1)}$

$$i:(1,1)\quad j:(-2,0)$$

:two: $\displaystyle{将i帽经过M_2矩阵变换后，就是将向量(1,1)乘上矩阵M_2}$

$$\underbrace{\left[\begin{matrix} 0&2 \\ 1&0 \\ \end{matrix}\right]}_{M_2} \underbrace{\left[\begin{matrix} 1 \\ 1 \\ \end{matrix}\right]}_{i} = 1\left[\begin{matrix} 0 \\ 1 \\ \end{matrix}\right] + 1\left[\begin{matrix} 2 \\ 0 \\ \end{matrix}\right] = \left[\begin{matrix} 2 \\ 1 \\ \end{matrix}\right]$$

:three: $\displaystyle{将j帽经过M_2矩阵变换后，就是将向量(-2,0)乘上矩阵M_2}$

$$\underbrace{\left[\begin{matrix} 0&2 \\ 1&0 \\ \end{matrix}\right]}_{M_2}\underbrace{\left[\begin{matrix} -2 \\ 0 \\ \end{matrix}\right]}_{j} = -2 \left[\begin{matrix} 0 \\ 1 \\ \end{matrix}\right] + 0 = \left[\begin{matrix} 0 \\ -2 \\ \end{matrix}\right]$$

:four: $\displaystyle{最后代入即可}$

$$\left[\begin{matrix} 0&2 \\ 1&0 \\ \end{matrix}\right] \left[\begin{matrix} 1&-2 \\ 1&0 \\ \end{matrix}\right] = \left[\begin{matrix} 2&0 \\ 1&-2 \\ \end{matrix}\right]$$

<font color="#83D1DD"> $\displaystyle{结论:}$</font>

$\displaystyle{可得公式如下:}$

$$\left[\begin{matrix} a&b \\ c&d \\ \end{matrix}\right] \left[\begin{matrix} e&f \\ g&h \\ \end{matrix}\right] = \left[\begin{matrix} ae+bg & af+bh \\ ce+dg &cf+dh \\ \end{matrix}\right]$$

<font color="#83D1DD"> $注意$ </font>

只有在第一个矩阵的`列数`等于第二个矩阵的`行数`时，两个矩阵才能相乘

## 行列式

- $\displaystyle{空间变换情况：}$

  - $\displaystyle{向外拉伸空间}$

  - $\displaystyle{向内挤压空间}$

### 拉伸空间

$\displaystyle{如下矩阵\left[\begin{matrix} 3&0 \\ 0&2 \\ \end{matrix}\right]它将i帽伸长为原来的3倍，将j帽伸长为原来的2倍}$

<img src="./img15.png" alt="./img15.png" width=70% />

- $\displaystyle{从图中可得信息有面积扩展为原来的6倍}$

$\displaystyle{这种特殊的缩放比例,即线性变换对面积产生改变的比例称为这个变换的\textcolor{red}{行列式}}$

<font color="#83D1DD"> $\displaystyle{以上线性变换可表示为:}$</font>

$$det\bigg(\left[\begin{matrix} 3&0 \\ 0&2 \\ \end{matrix}\right]\bigg)  = 6$$

### 挤压空间

$\displaystyle{同样，当面积发生缩小\frac{1}{2}倍时，那么此时的行列式也为\frac{1}{2}}$

<img src="./img16.gif" alt="./img16.gif" width=70% />

<font color="#83D1DD"> $\displaystyle{以上线性变换可表示为:}$</font>

$$det\bigg(\left[\begin{matrix} 0.5 & 0.5 \\ 0.5&0.5 \\ \end{matrix}\right]\bigg) =  \frac{1}{2}$$

$\displaystyle{当det\bigg(\left[\begin{matrix} 4&2 \\ 2&1 \\ \end{matrix}\right]\bigg) = 0时那么则表示该矩阵所代表的变换压缩到了更小维度}$

<img src="./img17.gif" alt="./img17.gif" width=70% />

<font color="#83D1DD">$\displaystyle{结论}$</font>

$\displaystyle{通过判断矩阵的行列式是否为0,即可得到该矩阵所代表的变换是否将空间压缩到了更小维度上}$

<br>

### 空间反转

$\displaystyle{行列式为负数时,此时的线性变为称为该变换反转了空间取向,而面积依旧是放大}$

<img src="./img18.gif" alt="./img18.gif" width=70% />

$\displaystyle{以上线性变换可表示为:}$

$$det\bigg(\left[\begin{matrix} 1 & 2 \\ 1&-1 \\ \end{matrix}\right]\bigg) = -3 $$

<font color="#83D1DD"> $\displaystyle{公式如下}$</font>

$$det\bigg(\left[\begin{matrix} a&b \\ c&d \\ \end{matrix}\right]\bigg) = ad - bc $$

$\displaystyle{好比i帽、j帽为1时的矩阵则行列式为1 => 1 \cdot 1 - 0 \cdot 0 =1}$

$\displaystyle{当b或c其中一项为0时(ad不为0)则矩阵为一个平行四边形,此时面积为ad}$

<img src="./img19.png" alt="./img19.png" width=70% />

$\displaystyle{而当b或c不为0时(ad不为0)则该平行四边形在对角方向上拉伸或压缩了多少}$

### 三维空间面积线性变换

<img src="./img20.gif" alt="./img20.gif" width=70% />

<font color="#83D1DD"> $\displaystyle{公式如下}$ </font>

$$
det\left(\left[\begin{matrix} a&b&c \\ d&e&f \\ g&h&i \\ \end{matrix}\right]\right) = a \cdot  det \left(\left[\begin{matrix} e&f \\ h&i \\ \end{matrix}\right]\right) -b \cdot det \left(\left[\begin{matrix} d&f \\ g&i \\ \end{matrix}\right]\right) + c \cdot det\left(\left[\begin{matrix} d&e \\ g&h \\ \end{matrix}\right]\right)\\
$$

<br>
<br>
<br>

## 线性方程组

$存在如下方程组:$

$$
\begin{aligned}
\begin{cases} &2x+5y+3z = -3 \\ &4x+0y+8z=0 \\&1x+3y+0z = 2 \\ \end{cases} \rightarrow \left[\begin{matrix} 2&5&3 \\ 4&0&8 \\1 & 3 & 0 \\ \end{matrix}\right] \left[\begin{matrix} x \\ y \\ z \end{matrix}\right] = \left[\begin{matrix} -3 \\ 0 \\ 2 \\\end{matrix}\right]
\end{aligned}
$$

A 的逆变换：$A^{(-1)}$

恒等变换：$A \cdot A^{(-1)} =  \left[\begin{matrix}1&0 \\ 0&1 \\ \end{matrix}\right](i帽、j帽不变)$

$$A \vec{x} = \vec{v}  =>  A^{-1}A \vec{x} = A^{-1} \vec{v} => \vec{x} = A^{-1}\vec{v}$$

线性变换 A 没有压缩空间纬度（行列式不为 0），则它存在逆变换

不存在逆变换时。将空间变换压缩为一条直线，向量 v 恰好落与该直线，则该解存在

### 秩(Rank)

- 秩：表示变换后的空间维数

- 列空间:一个变换所有可能的变换结果的集合称为矩阵的“列空间”

- https://www.bilibili.com/video/BV1ns411r7dE/?spm_id_from=333.788.recommend_more_video.0&vd_source=1ec51cb8123536a0bf872aa061240412
