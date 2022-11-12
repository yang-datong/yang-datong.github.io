> 2.

$$
\begin{aligned}
& y' = f'(u) + u' = \arctan{u}^2 \cdot  (\frac{x-2}{3x+2})' \\
& =\arctan{u^2} \cdot  \frac{(x-2)' \cdot (3x+2) - (x-2) \cdot (3x+2)'}{(3x+2)^2} \\
& =\arctan{u^2} \cdot  \frac{(3x+2)-(3x-6)}{(3x+2)^2}\\
& =\arctan{u^2} \cdot  \frac{2+6}{(3x + 2)^2} \\
& x=0 => \arctan{1} \cdot 4 = \frac{\pi}{4} \cdot 2 = \frac{\pi}{2} \\
\end{aligned}
$$

> 3.

$$
\begin{aligned}
& A.不理解 \\
& B.f(x) = \begin{cases} &-x,x \leq 0 \\ &x,x>0 \\ \end{cases} f_-'(0)=-1 ,f'_+(0) = 1 , |f'(0)| = 1\\
& C.set \quad f(x) = x^2 - 1,f(1) = 0 , f'(1) = 2x-1 = 1 \\
& D.f(x) = \begin{cases} &1, x\geq 0 \\ &-1,x<0 \\ \end{cases} , g(x) = \begin{cases} &-1,x\geq 0 \\ &1,x<0 \\ \end{cases} , f(x) +g(x) = 0 \\
\end{aligned}
$$

> 4.

$$
\begin{aligned}
& df({\sqrt{1-x^2}})  = f'(\sqrt{1-x^2} ) \cdot {dx} \\
& set \quad \sqrt{1-x^2} = u, f'(u) = f'(x) \cdot u'    \\
& = \frac{2(\sqrt{1-x^2})}{\sqrt{1-(\sqrt{1-x^2})^2}} \cdot \bigg[(1-x^2)^\frac{1}{2}\bigg]' \\
& = \frac{2(\sqrt{1-x^2})}{\sqrt{1-(\sqrt{1-x^2})^2}}  \cdot \frac{1}{2}(1-x^2)^{-\frac{1}{2}} \cdot-2x\\
& = \frac{2(\sqrt{1-x^2})}{\sqrt{1-(\sqrt{1-x^2})^2}}  \cdot \frac{1}{2} \frac{1}{\sqrt{1-x^2} } \cdot -2x \\
& = \frac{-2x}{\sqrt{1-(|1-x^2|)}} \\
& = \frac{-2x}{|x|}
\end{aligned}
$$

> 5.

$$
\begin{aligned}
& \because |f(x)| \leq x^2 , \therefore f(0) = 0 , f(x) \geq f(0) \\
& \therefore \lim\limits_{x\rightarrow 0}f(x) = f(0) = 0  \\
& \therefore x =0 ，f(x)连续 \\
& \therefore f'(0) = \lim\limits_{x\rightarrow 0}\frac{f(x+0)-f(0)}{x} = 0\\
\end{aligned}
$$

> 6.

$$
\begin{aligned}
& f'(1) = \lim\limits_{x\rightarrow 0}\frac{f(x + 1) - f(1)}{x} = \lim\limits_{x\rightarrow 0}\frac{af(x) - af(0)}{x} \\
& = \lim\limits_{x\rightarrow 0}\frac{a(f(x) - f(0))}{x}  = a \cdot \lim\limits_{x\rightarrow 0}\frac{f(x + 0 ) - f(0)}{x} = a \cdot f'(0)\\
&  = ab \\
\end{aligned}
$$

2

$$
\begin{aligned}
& x=0 => f(1) = af(0) , f'(0) = \frac{d}{dx}\bigg(\frac{f(1)}{a}\bigg) = b \\
& b = \frac{f'(1)a - f(1)a'}{a^2} = \frac{f'(1)a}{a^2} \\
& \therefore  f'(1) = ab \\
\end{aligned}
$$

> 7.

$$
\begin{aligned}
& P'(x_0)  = \lim\limits_{x\rightarrow 0}\frac{f(x + x_0) - f(x_0)}{x} \\
& \because x= x_0为最大实根 \\
& \therefore x \geq x_0 \geq  0  , f(x) \geq  f(x_0) \geq 0 \\
\end{aligned}
$$

> 5.

$$
\begin{aligned}
& \because |f(x)| \leq  x^2   \therefore f(0) = 0 \\
& \lim\limits_{x\rightarrow 0}|f(x)| \leq \lim\limits_{x\rightarrow 0}x^2=0 \\
& \because |f(x) -f(0)| \geq  0 \\
& \therefore \lim\limits_{x\rightarrow 0}f(x) = f(0) = 0 => 连续 \\
&  \\
& f'(0) = \lim\limits_{x\rightarrow 0}\bigg|\frac{f(x) - f(0)}{x}\bigg|  = \lim\limits_{x\rightarrow 0}\bigg|\frac{f(x)}{x}\bigg| \leq \lim\limits_{x\rightarrow 0}|x| = 0\\
& \therefore f'(0) = 0 \\
\end{aligned}
$$

> 6.

$$
\begin{aligned}
& x = 0 : f(1) = af(0) => f(0) = \frac{f(1)}{a}  \\
& f'(0) = \bigg[ \frac{f(1)}{a} \bigg]' = \frac{f'(1)a + f(1)a'}{a^2} = \frac{f'(1)a}{a^2} = b \\
& \therefore f'(1) = ab \\
\end{aligned}
$$

2

$$
\begin{aligned}
& f'(1)  = \lim\limits_{x\rightarrow 0}\frac{f(1 + x) - f(1)}{x} = \lim\limits_{x\rightarrow 0}\frac{af(x) - af(0)}{x} \\
& =\lim\limits_{x\rightarrow 0}\frac{a[f(x) - f(0)]}{x}  = af'(0) = ab\\
\end{aligned}
$$

> 7.

$$
\begin{aligned}
& \because x= x_0为最大实根 \therefore x-x_0 \rightarrow 0 => x\rightarrow x_0 \\
& P'(x_0) = \lim\limits_{x\rightarrow x_0}\frac{f(x  - x_0 + x_0) - f(x_0)}{x-x_0} \\
& =\lim\limits_{x\rightarrow x_0}\frac{f(x)-f(x_0)}{x-x_0}  \\
& \because x_0为最大实根,\therefore f(x) - f(x_0) \geq  0   \\
& \therefore P'(x_0) \geq  0  \\
\end{aligned}
$$

> 1.

$$
\begin{aligned}
& \lim\limits_{x\rightarrow 0}\frac{[f(x_0 + 2x) - f(x_0)] -[f(x_0 -x) -f(x_0) ]}{2x} \\
& = f'(x_0)  + \frac{1}{2}f'(x_0) = 1\\
& \therefore f'(x_0) = \frac{2}{3} \\
\end{aligned}
$$

> 8.

$$
\begin{aligned}
& f(x) = \begin{cases} &3x^2 + x^3 , x>0\\&3x^2 - x^3  ,x<0  \\ \end{cases} \\
& f'(x) = \begin{cases} &6x + 3x^2 , x>0\\&6x - 3x^2  ,x<0  \\ \end{cases} \\
& f''(x) = \begin{cases} &6 + 6x , x>0\\&6 - 6x  ,x<0  \\ \end{cases} \\
\end{aligned}
$$

> 9.

$$
\begin{aligned}
& f'(a) = \lim\limits_{x\rightarrow a}\frac{f(x - a + a) - f(a)}{x-a}  = \lim\limits_{x\rightarrow 0}\frac{f(x) - f(a)}{x-a} > 0\\
& \therefore f(x) - f(a) > 0  ,\quad  x- a >0 或者 f(x) - f(x) < 0 ,x -a <0\\
\end{aligned}
$$

> 10.

$$
\begin{aligned}
& (1,-1)代入y =x^2 + ax + b => -1 = 1 + a + b => a+b = -2 \\
& y' = (x^2 + ax + b)' = 2x + a  , x= 1代入,则斜率k =2 + a  \\
&  \\
& set \quad f'(x) = (2y = -1 + xy^3)' = 2y' = y^3 + 3xy^2 \cdot y' \\
& y'(2 - 3xy^2) = y^3 \\
&(1,-1)代入 :f'(x) = y' = \frac{y^3}{2 - 3xy^2}  = \frac{-1}{2-3} = 1  \\
& \therefore f'(x) = y' = 2+a = 1   \\
& \therefore  a = -1 , b = -1 \\
\end{aligned}
$$

### 填空

>1.

$$\begin{aligned}
& f(2) = \lim\limits_{x\rightarrow 2}f(x)  = \lim\limits_{x\rightarrow 2}(x-2) \cdot \lim\limits_{x\rightarrow 2}\frac{f(x)}{x-2} = 0\\
f'(2)& = \lim\limits_{x\rightarrow 2}\frac{f(x-2 + 2) - f(2)}{x-2} = \lim\limits_{x\rightarrow 2}\frac{f(x) - f(2)}{x-2} \\
& = \lim\limits_{x\rightarrow 2}\frac{f(x)}{x-2}  - \lim\limits_{x\rightarrow 2}\frac{f(2)}{x-2}  = 2 - 0 = 0\\
\end{aligned}$$

>2.

$$\begin{aligned}
& set \quad 切线方程为 y  - y_0 = k(x - x_0) \\
& set\quad 切点(m,\frac{1}{m}) \\
& \therefore  k = y'(m) = -\frac{1}{m^2} \\
& 将k,(-3,1)代入方程 => y - \frac{1}{m} = -\frac{1}{m^2}(x-m) \\
& 1 - \frac{1}{m} = -\frac{1}{m^2} (-3 - m) \\
& 1 - \frac{1}{m} = \frac{3}{m^2} + \frac{1}{m} \\
& m^2 - m = 3 + m  \\
& m^2 - 2m -3 = 0\\
& (m-3)(m+1) = 0 , m = \begin{cases} &3 \\ &-1 \\ \end{cases} \\
& \therefore f(x) = \begin{cases} &y = -\frac{1}{9}(x+3) - 1 \\ &y=-(x+3) -1 \\ \end{cases}
\end{aligned}$$


### 填空
>1.
 $$\begin{aligned}
& f'(2) = \lim\limits_{x\rightarrow 2}\frac{f(x) - f(2)}{x-2} = \lim\limits_{x\rightarrow 2}\frac{f(x)}{x-2} - \lim\limits_{x\rightarrow 2}\frac{f(2)}{x-2} \\
& f(2)  = \lim\limits_{x\rightarrow 2}f(x) = \lim\limits_{x\rightarrow 2}(x-2) \frac{f(x)}{(x-2)} =  0  \\
& \lim\limits_{x\rightarrow 2}\frac{f(x)}{x-2} = 2 \\
& \lim\limits_{x\rightarrow 2}\frac{f(2)}{x-2} = 0 \\
& \therefore f'(2) = 2 - 0 = 2 \\
\end{aligned}$$

>2.
$$\begin{aligned}
& set \quad 切线方程 : y - y_0 = k(x - x_0) \\
& \because y = x^{-1}  => y' = -\frac{1}{x^2} \\
& \therefore k = - \frac{1}{x^2} \\
& set \quad 切点为(m,\frac{1}{m}) , k = -\frac{1}{m^2}\\
& \therefore 方程变为: y - \frac{1}{m} = - \frac{1}{m^2} (x - m) \\
& 将点(-3,1)代入方程: 1 - \frac{1}{m} = - \frac{1}{m^2} (-3 - m) \\
& m^2 - m = 3 + m => m^2-2m-3 = 0 => (m-3)(m+1) = 0 \\
& \therefore 切线方程 = \begin{cases} &y - \frac{1}{3} = -\frac{1}{9}(x-3) ,m = 3 \\ &y + 1 = -1(x+1),m = -1 \\ \end{cases}\\
& = \begin{cases} &9y = -x+6 \\ &y = -x-2 \\ \end{cases}
\end{aligned}$$

>3.
$$\begin{aligned}
& \ln{y}  = \cos{^2x}\ln{\sin{x}} \\
& \ln{y}'= \frac{y'}{y} = 2\cos{x}\sin{x}\ln{\sin{x}} + \cos{^2x} \frac{1}{\sin{x}} \cdot \cos{x} \\
&  = \sin{2x}\ln{\sin{x}} + \frac{\cos{^3x}}{\sin{x}} \\
& \therefore  y' = \ln{y'} \cdot y = [-\sin{2x}\ln{\sin{x}} + \frac{\cos{^3x}}{\sin{x}}] \cdot  (\sin{x})^{\cos{^2x}} \\
\end{aligned}$$

>4.
$$\begin{aligned}
& set \quad 切线方程:  y - y_0 = k(x- x_0)  \\
& f'(x) =  nx^{n - 1} , \frac{df}{dx}|_{x=1} = n  \\
& 点(1,1)代入 : y = n(x-1) + 1 \\
& 点(\xi_n,0)代入 : n(\xi_n - 1)+ 1 = 0 => \xi_n = \frac{n-1}{n}\\
& \therefore \lim\limits_{n\rightarrow \infty}f(\xi_n) = \lim\limits_{n\rightarrow \infty}(1- \frac{1}{n})^n = \frac{1}{e}
\end{aligned}$$


>5.
$$\begin{aligned}
&  \\
&  \\
&  \\
&  \\
&  \\
\end{aligned}$$

>6.

$$\begin{aligned}
& \ln{y} = 3u \\
& (\ln{y})' = \frac{y'}{y} \\
& y' = (3u)' \cdot e^{3u} \\
& (3u)'  = [3f(\ln{x})]' = 3[f(\ln{x})]'  = 3f'(\ln{x}) \cdot \frac{1}{x}\\
& y' = \frac{e^{3f(\ln{x})} \cdot 3f'(\ln{x})}{x} \\
\end{aligned}$$

>7.

$$\begin{aligned}
& \frac{d}{dx}(y\sin{x}) = y'\sin{x} + y\cos{x} \\
& \frac{d}{dx}(\cos{(x-y)}) = -\sin{(x-y)} \cdot (1 - y') \\
& y'\sin{x} + y'\sin{(x-y)}  = -\sin{(x-y)} - y\cos{x}\\
& y' = -\frac{\sin{(x-y) - y\cos{x}}}{\sin{x} + \sin{(x-y)}} \\
& dy = -\frac{\sin{(x-y) - y\cos{x}}}{\sin{x} + \sin{(x-y)}} \cdot dx \\
\end{aligned}$$

>8.

$$\begin{aligned}
& set \quad  t= \frac{1}{n} , f(x) = x^k \\
& \therefore \lim\limits_{n\rightarrow \infty}n[(1+\frac{1}{n})^k - 1] = \lim\limits_{t\rightarrow 0}\frac{(1+t)^k - 1}{t}= \lim\limits_{t\rightarrow 0}\frac{(1+t)^k - 1^k}{t} = \lim\limits_{t\rightarrow 0}\frac{f(1+t)-f(1)}{t}\\
& = f'(1) \\
& \because f'(x) = kx^{k-1} \\
& \therefore f'(1) = k
\end{aligned}$$

2

$$\begin{aligned}
& set \quad  t= \frac{1}{n} \\
& \therefore \lim\limits_{n\rightarrow \infty}n[(1+\frac{1}{n})^k - 1] = \lim\limits_{t\rightarrow 0}\frac{(1+t)^k - 1}{t} \\
& 根据等价无穷小可得: (1 + t)^k - 1 = kt \\
& \therefore =k \\
\end{aligned}$$

简答题

>1.

$$\begin{aligned}
& 将x=0代入 : \\
& e^y  = e => y = 1 \\
 一阶导 : &e^yy' + y  + xy' = 0  \\
& 将x= 0 ,y = 1代入: \\
& ey' + 1 = 0  => y' = - \frac{1}{e}\\
二阶导:&e^yy'y' + e^y y'' + y' + y' +xy'' = 0  \\
& 将x= 0 , y =1 , y' = \frac{1}{e}代入: \\
& e \cdot -\frac{1}{e} \cdot -\frac{1}{e} + ey'' \cdot  - \frac{2}{e} = 0 \\
& y'' = \frac{1}{e^2}  \\
\end{aligned}$$


>2.

$$\begin{aligned}
& x= 0 : f(y) = f(0)f(y) , \therefore f(0) = 1 \\
& f'(y) = \lim\limits_{x\rightarrow 0}\frac{f(x + y ) - f(y)}{x} = \lim\limits_{x\rightarrow 0}\frac{f(x)f(y) - f(y)}{x} = \lim\limits_{x\rightarrow 0}\frac{f(y)(f(x) - 1)}{x} \\
& = \lim\limits_{x\rightarrow 0}\frac{f(y)(1 + xg(x) - 1)}{x} = \lim\limits_{x\rightarrow 0}\frac{f(y) \cdot 1}{1} = \lim\limits_{x\rightarrow 0}f(y)\\
\end{aligned}$$


>3.
$$\begin{aligned}
& set \quad f(x) = x^n + x^{n-1} + \dots + x^2 + x - 1  \\
& f(0) =  -1 , f(1)  = n - 1 \\
& 根据零点定理x_0 \in (0,1)区间必有y(x_0) = 0 \\
& \therefore 可知至少存在一个实根 \\
& f'(x) = nx^{n-1} + (n-1)x^{n-2} + \dots + 2x + 1 \\
& \therefore f'(x)  > 0
\end{aligned}$$

>4.

$$\begin{aligned}
& 0 \leq f(0) \leq  0  \therefore  f(0) = 0 \\
& \lim\limits_{x\rightarrow 0}f(x) = f(0) = 0 \\
& \therefore f(x)在x处连续 \\
& f'(0) =  \lim\limits_{x\rightarrow 0}\frac{f(x) - f(0)}{x} = \lim\limits_{x\rightarrow 0}\frac{f(x)}{x} \\
& \because x \leq  f(x) \leq  x^3 + x  \\
& \therefore \frac{x}{x} \leq \frac{f(x)}{x} \leq x^2 + 1 \\
& \because \lim\limits_{x\rightarrow 0}\frac{x}{x} = \lim\limits_{x\rightarrow 0}x^2 + 1 = 1 \\
& \therefore \lim\limits_{x\rightarrow 0}\frac{f(x)}{x} = 1  \\
& \therefore f'(0) = 1 \\
\end{aligned}$$



>5.

$$\begin{aligned}
set \quad f(x) & = (1+x)\ln{^2(1+x)} - x^2 \\
& f(0) = 0 \\
set \quad f'(x) &= \ln{^2(1+x)} + (1+x) \cdot 2\ln{(1+x)} \cdot \frac{1}{1+x} - 2x \\
& = \ln{^2(1+x)} + 2\ln{(1+x)} - 2x\\
& f'(0) = 0 \\
set \quad f''(x)&= 2\ln{(1+x)} \frac{1}{1+x} + \frac{2}{1+x} - 2  \\
& = \frac{2[\ln{(1+x) +1}]}{1+x}  - 2 \\
& f''(0)  = 0 \\
set \quad f'''(x)& = \frac{(2[\frac{1}{1+x}])(1+x) - 2[\ln{(1+x)+1}]}{(1+x)^2} \\
& = \frac{ - 2\ln{(1+x)}}{(1+x)^2} \\
& f'''(0) < 0 \\
& 由斜率为负数,可知f(x)在x \in (0,1)单调递减,即f(x) < f(0) \\
& \therefore (1+x)\ln{^2(1+x)} < x^2
\end{aligned}$$



<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
