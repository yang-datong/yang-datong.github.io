## 选择题

> 1.  $\displaystyle{set \quad 函数f(x) 在点x_0处可导，且\lim\limits_{x\rightarrow 0}\frac{f(x_0 + 2x) - f(x_0  -x)}{2x} = 1 , f'(x_0)=?}$

$$
\begin{aligned}
& set \quad a = x_0 \\
& \lim\limits_{x\rightarrow 0}\frac{f(a + 2x) - f(a - x)}{2x} = \lim\limits_{x\rightarrow 0}\frac{f(a + 2x) - f(a) -[f(a- x) - f(a)]}{2x} \\
& = f'(a)  + \frac{1}{2}f'(a) = 1 \\
& \therefore -\frac{1}{2} f'(a) = 1 => f'(a)  = \frac{2}{3} \\
\end{aligned}
$$

> 2.  $\displaystyle{exitst \quad y = f(\frac{x-2}{3x+2}) , f'(x) = \arctan{x^2},则\frac{dy}{dx}\bigg|_{x=0} = ?}$

$$
\begin{aligned}
& y' = \arctan{(\frac{x-2}{3x+2})^2} (\frac{x-2}{3x+2})' = \arctan{(\frac{x-2}{3x+2})^2} \frac{(x-2)'(3x+2) - (x-2)(3x+2)'}{(3x+2)^2} \\
& =\arctan{(\frac{x-2}{3x+2})^2} \frac{(3x+2) - 3(x-2)}{(3x+2)^2} = \arctan{(\frac{x-2}{3x+2})^2} \frac{8}{(3x+2)^2} \\
& \therefore  y'(0) = \arctan{1} \cdot 2 = \frac{\pi}{2}   \\
\end{aligned}
$$

> 3.

$$
\begin{aligned}
 A.&f(x) = x在点x=0可导=> f'(x) = \lim\limits_{ \Delta{x} \rightarrow 0}\frac{f( \Delta{x} +x) - f(x)}{ \Delta{x}} =  1\\
& 但|f(x)| = |x|在点x=0不可导 => |f'(x)| = \frac{x}{|x|} \\
B.& set \quad f(x) = \begin{cases} &-1 , x\leq 0 \\ &1 , x>0 \\ \end{cases} \quad |f'(0)| = 1 , f'(0)不连续即不可导\\
C.& set \quad set \quad f(x) = x , f(0) = 0 , f'(0) = 1\\
D.& set \quad f(x) = \begin{cases} &1,x\leq 0 \\ &-1 ,x>0 \\ \end{cases}  , g(x)  = \begin{cases} &-1,x\leq 0 \\ &1 ,x>0 \\ \end{cases} ;\frac{d}{dx}[f(x) + g(x)]=0[yes]\\
\end{aligned}
$$

> 4. $\displaystyle{f'(x) = \frac{2x}{\sqrt{1-x^2}} , 则df(\sqrt{1-x^2})=?}$

$$
\begin{aligned}
f'(\sqrt{1-x^2})& = \frac{2(\sqrt{1-x^2})}{\sqrt{1-(\sqrt{1-x^2})^2}} \cdot (\sqrt{1-x^2})'  \\
& = \frac{2\sqrt{1-x^2}}{|x|} \cdot \frac{1}{2}(1 - x^2)^{-\frac{1}{2}} \cdot (1-x^2)'\\
& = \frac{2\sqrt{1-x^2}}{|x|} \cdot \frac{1}{2}\frac{1}{\sqrt{1-x^2}} \cdot -2x\\
& = - \frac{2x}{|x|} \\
& \therefore df(\sqrt{1-x^2}) =  - \frac{2x}{|x|}  \cdot dx\\
\end{aligned}
$$

> 5.  $\displaystyle{set \quad f(x) 在区间(-\xi,\xi)内由定义，若当x \in (-\xi,\xi)是，恒有|f(x)| \leq x^2,x=0是f(x)的?}$

$$
\begin{aligned}
& \because \lim\limits_{x\rightarrow 0}|f(x)| \leq \lim\limits_{x\rightarrow 0}x^2  = 0\\
& \therefore \lim\limits_{x\rightarrow 0}|f(x)| \leq  0  \\
& \because |f(x)| \leq x^2  => f(0)  = 0 \\
& \therefore \lim\limits_{x\rightarrow 0}|f(x)| \leq f(0) \\
&  \\
& \because |f(x)| \leq x^2  => |f(x)| \geq  0 \\
& \therefore |f(x)| \geq f(0) \\
&  \\
& 综上可知 \begin{cases} &|f(x)| \geq f(0) \\ &\lim\limits_{x\rightarrow 0} | f(x)| \leq f(0) \\ \end{cases} \\
& \therefore |\lim\limits_{x\rightarrow 0}f(x)| = f(0) = 0 => 故f(x)在x=0处连续 \\
&  \\
& f'(0) = \lim\limits_{x\rightarrow 0}\bigg|\frac{f(x) - f(0)}{x}\bigg| = \lim\limits_{x\rightarrow 0}|\frac{f(x)}{x}| \\
& \because |f(x)| \leq x^2 => \bigg|\frac{f(x)}{x}\bigg| \leq |x| \\
& \therefore f'(0)=\lim\limits_{x\rightarrow 0}\bigg|\frac{f(x)}{x}\bigg| = 0\\
\end{aligned}
$$

> 6.  $\displaystyle{set \quad f(x+1) = af(x)总成立，f'(0) = b , a , b 为非零常数，则f(x) 在 x= 1处 ?}$

$$
\begin{aligned}
& f'(1) = \lim\limits_{x\rightarrow 0}\frac{f(x + 1) - f(1)}{x}  = \lim\limits_{x\rightarrow 0}\frac{af(x) - af(0)}{x} = a \lim\limits_{x\rightarrow 0}\frac{f(x) - f(0)}{x} = ab\\
\end{aligned}
$$

> 7.  $\displaystyle{set \quad 多项式P(x) = x^4 + a_3x^3 + a_2x^2 + a_1x + a_0,又设x= x_0 , 是它的最大实根,则P(x_0)?}$

$$
\begin{aligned}
& P'(x_0) = \lim\limits_{x\rightarrow x_0} \frac{P(x-x_0 + x_0) - P(x_0)}{x-x_0} = \lim\limits_{x\rightarrow x_0}\frac{P(x) - P(x_0)}{x-x_0} \\
&  \\
& \therefore if \quad x > x_0 > 0  ; then \quad P(x) \geq  P(x_0) > 0 \\
& \therefore P'(x_0) \geq  0  \\
\end{aligned}
$$

> 8.  $\displaystyle{f(x) = 3x^2 + x^2 |x| ,则使f^{(n)}(0)存在的最高阶n=?}$

$$
\begin{aligned}
&  \\
& \because set \quad g(x) = |x| => g'(x) = \frac{x}{|x|} = \pm{1} \\
& \therefore f'(0) = \begin{cases} &6x + 3x^2 , x \geq 0 \\ &6x-3x^2 , x<0 \\ \end{cases} \\
& f''(0) = \begin{cases} &6 + 6x , x \geq 0 \\ &6-6x , x<0 \\ \end{cases} \\
& f'''(0) = \begin{cases} &6 , x \geq 0 \\ &-6 , x<0 \\ \end{cases}(此时不可导) \\
& \therefore 最高阶数为2
\end{aligned}
$$

> 9.  $\displaystyle{set \quad f'(a) > 0 , 则\exists \delta > 0 ,有 }$

$$
\begin{aligned}
& f'(a) = \lim\limits_{x\rightarrow a}\frac{f(x) - f(a)}{x-a} \\
& \because f'(a) > 0  \\
& \therefore \begin{cases} &f(x) - f(a) >0 并且 x- a > 0\\ &f(x) - f(a) < 0并且x - a < 0 \\ \end{cases} \\
\end{aligned}
$$

> 10. $\displaystyle{set \quad 曲线y = x^2 + ax + b和2y = -1 + xy^3 在点(1,-1)处相切，其中a,b是常数}$

$$
\begin{aligned}
& y' = 2x + a  \\
& \therefore 在切点(1,-1)处 斜率k = 2 + a \\
& 对切线两边分别求导: \\
 2y' &= y^3 + 3xy^2 \cdot y' \\
& => 2y' - 3xy^2y' = y^3 \\
& => y' =  \frac{y^3}{(2 - 3xy^2)} \\
& 代入(1,-1) => y' = 1 \\
& \therefore k = 2+a = 1  \\
& 将(1,-1) , a= -1代入曲线: \\
& \therefore a = -1 , b = -1 \\
\end{aligned}
$$

## 填空

> 1.  $\displaystyle{f(x) 在x=2处连续 ， 且\lim\limits_{x\rightarrow 2}\frac{f(x)}{x-2} = 2 , f'(2) = ?}$

$$
\begin{aligned}
& \because f(x)在x=2处连续 \\
& \therefore  \lim\limits_{x\rightarrow 2}f(x) = f(2) = \lim\limits_{x\rightarrow 2}(x-2) \cdot \lim\limits_{x\rightarrow 2}\frac{f(x)}{(x-2)} = 0 \\
&  \\
& \therefore f'(2) = \lim\limits_{x\rightarrow 2}\frac{f(x) - f(2)}{x-2} =2 -  \lim\limits_{x\rightarrow 2} \frac{f(2)}{x-2}  = 2- 0 =  2\\
\end{aligned}
$$

> 2.  $\displaystyle{给定曲线y = \frac{1}{x} , 则过点(-3 ,1)的切线方程为?}$

$$
\begin{aligned}
& set \quad 切线方程: y - 1 = k(x + 3) \\
& set \quad 切点: (m,\frac{1}{m}) \\
& y' = -x^{-2} = - \frac{1}{x^2} \\
& 代入切线方程可得: \\
& \frac{1}{m} - 1 = -\frac{1}{m^2}(m + 3) \\
& m - m^2 = -m - 3 \\
& m^2 - 2m  -3 = 0 \\
& m = \begin{cases} &3 \\ &-1 \\ \end{cases} \\
& \therefore 切线方程为 \begin{cases} &9y = -x + 6 , k = -\frac{1}{9}\\ &y = -x-2 ,k=-1\\ \end{cases} \\
\end{aligned}
$$

> 3.  $\displaystyle{set \quad y = (\sin{x})^{\cos{^2x}} , y' =?}$

$$
\begin{aligned}
\because (\ln{y})' &= \frac{y'}{y} = (\cos{^2x} \ln{\sin{x}})' = 2\cos{x} \cdot -\sin{x}\ln{\sin{x}} + \cos{^2x}\frac{\cos{x}}{\sin{x}} \\
& = -2\cos{x}\sin{x}\ln{\sin{x}} + \frac{\cos{^3x}}{\sin{x}}\\
& =-\sin{2x}\ln{\sin{x}} + \frac{\cos{^3x}}{\sin{x}} \\
& \therefore y' = y (\ln{y})'  =\bigg(-\sin{2x}\ln{\sin{x}} + \frac{\cos{^3x}}{\sin{x}}\bigg)\sin{x}^{\cos{^2x}} \\
\end{aligned}
$$

> 4.  $\displaystyle{set \quad 曲线f(x) = x^n 在点(1,1)处的切线与x轴的交点为(\xi,0),则f(\xi_n) =?}$

$$
\begin{aligned}
& f'(x) = nx^{n-1} => f'(1) = n \\
& set \quad 切线方程: y- y_0 = k(x - x_0)  \\
& 将k = n , (1,1)代入方程: \\
& y = n(x-1)+1 \\
& 将(\xi_n,0)代入方程: \\
& n\xi_n - n = -1 \\
& \xi_n = \frac{n-1}{n} \\
& \lim\limits_{n\rightarrow \infty}f(\xi_n) = \lim\limits_{n\rightarrow \infty}(\frac{n-1}{n})^n = \lim\limits_{n\rightarrow \infty}(1 - \frac{1}{n})^n = \frac{1}{e} \\
\end{aligned}
$$

<font color=red>------------------------------------------------------------------------------------------------------------------------</font>

### <center>完全不懂</center>

> 5.  $\displaystyle{set \quad  y = \sin{^3x} , 则y^{(n)} = ?}$

$$
\begin{aligned}
\sin{^3x} &= \frac{1}{2}(1-\cos{2x})\sin{x}  = \frac{1}{2}(\sin{x} - \sin{x}\cos{2x})\\
& = \frac{1}{2}\sin{x} - \frac{1}{2}\sin{x}\cos{2x} \\
& = \frac{1}{2}\sin{x} - \frac{1}{2} \cdot \frac{1}{2}(\sin{3x} - \sin{x}) \\
& = \frac{1}{2}\sin{x} - \frac{1}{4}\sin{3x} + \frac{1}{4}\sin{x} \\
& = \frac{3}{4}\sin{x}-\frac{1}{4}\sin{3x}\\
(\sin{^3x})^n& = \frac{3}{4}(\sin{x})^n - \frac{1}{4}(\sin{3x})^n\\
& =\frac{3}{4}\sin{(x+\frac{n\pi}{2})} - \frac{3^n}{4}\sin{(3x+\frac{n\pi}{2})} \\
\end{aligned}
$$

<font color=red>------------------------------------------------------------------------------------------------------------------------</font>

> 6.  $\displaystyle{set \quad y = e^{3u} , u  = f(t) , t= \ln{x} , 其中f(u)可微分,dy = ?}$

$$
\begin{aligned}
y' &= e^{3u} \cdot (3u)' = e^{3f(\ln{x})} \cdot 3f'(t) \frac{1}{x} \\
& =\frac{3e^{3f(\ln{x})}f'(\ln{x})}{x}  \\
\end{aligned}
$$

> 7.  $\displaystyle{set \quad y\sin{x} - \cos{(x-y)} = 0 , dy = ?}$

$$
\begin{aligned}
[y\sin{x} - \cos{(x-y)}]' &= y'\sin{x} + y\cos{x} - (-\sin{(x-y)})(1-y') \\
& = y'\sin{x} + y\cos{x} + \sin{(x-y)}(1-y') \\
& = y'\sin{x} + y\cos{x} + \sin{(x-y)} - \sin{(x-y)}y' = 0 \\
y'&= \frac{y\cos{x} + \sin{(x-y)}}{\sin{(x-y) - \sin{x}}} \\
dy&= \frac{y\cos{x} + \sin{(x-y)}}{\sin{(x-y) - \sin{x}}} \cdot dx \\
\end{aligned}
$$

> 8.  $\displaystyle{set \quad  k 为常数,\lim\limits_{n\rightarrow \infty}n[(1+\frac{1}{n})^k -1] = ?}$

方法一:

$$
\begin{aligned}
& 根据等阶无穷小=> \lim\limits_{x\rightarrow \infty}[(x+1)^n - 1] = nx\\
& set \quad t= \frac{1}{n} \\
& \therefore \lim\limits_{n\rightarrow \infty}n[(1+\frac{1}{n})^k - 1] = \lim\limits_{t\rightarrow 0}\frac{1}{t}[(1+t)^k - 1] = \frac{kt}{t} = k \\
\end{aligned}
$$

方法二:

$$
\begin{aligned}
& set \quad t = \frac{1}{n} \\
& set \quad f(x) = x^k \\
& \therefore \lim\limits_{n\rightarrow \infty}n[(1+\frac{1}{n})^k - 1] = \lim\limits_{t\rightarrow 0}\frac{1}{t}[(1+t)^k - 1] = \lim\limits_{t\rightarrow 0}\frac{f(1 + t) - f(1)}{t} = f'(1) \\
& \because f'(x) = kx^{(k-1)} \\
& \therefore f'(1) = k \\
\end{aligned}
$$

> 9.  $\displaystyle{方程x^y = y^x 确定x = x(y) , 则\frac{dx}{dy} = ?}$

$$
\begin{aligned}
& \frac{dy}{dx} = y' => \frac{dx}{dy} = \frac{1}{y'} \\
& 对方程取对数进行求导: \\
& (y\ln{x})' = (x\ln{y})' \\
& y'\ln{x} + \frac{y}{x} = \ln{y} + x \frac{y'}{y} \\
& y'\ln{x} - \frac{xy'}{y} = \ln{y} - \frac{y}{x} \\
& y'xy\ln{x} + y^2 = xy\ln{y} + x^2y' \\
& y'= \frac{xy\ln{y} - y^2}{xy\ln{x} - x^2}\\
& \frac{dy}{dx} = \frac{x^2 - xy\ln{x}}{y^2 - xy\ln{y}} \\
\end{aligned}
$$

## 简答题

> 1.  $\displaystyle{set \quad  函数y = y(x) 由方程e^y + xy = e所确定,求y''(0)}$

$$
\begin{aligned}
set \quad& g(x) => e^y + xy - e  = 0\\
& 代入x=0 :e^y = e => y = 1 \\
g'(x)&=> e^yy' + y +  xy' = 0\\
& 代入x=0,y=1:ey' + 1 =0 => y' = - \frac{1}{e} \\
g''(x)&=>e^yy'y' + e^yy'' + y' + y' + xy'' =0   \\
& 代入x=0,y=1,y' = - \frac{1}{e} : e \cdot \frac{1}{e^2}+ ey'' - \frac{2}{e} =0 \\
& y'' = (-\frac{1}{e}  + \frac{2}{e} ) \frac{1}{e} = \frac{1}{e^2}\\
\end{aligned}
$$

> 2.  $\displaystyle{set \quad f(x)在(-\infty , + \infty )上有定义,且对任何 x,y 有 f(x+y) = f(x)f(y) , f(x) = 1+ xg(x)}\\

$$
\begin{aligned}
& f'(x_0) =  \lim\limits_{ \Delta{x}\rightarrow 0}\frac{f( \Delta{x}+x_0) - f(x_0)}{ \Delta{x}} = \lim\limits_{ \Delta{x}\rightarrow 0}\frac{f(x_0)[f( \Delta{x}) - 1]}{ \Delta{x}} = \lim\limits_{ \Delta{x}\rightarrow 0}\frac{f(x_0) \Delta{x}g(x)}{ \Delta{x}}  \\
& =\lim\limits_{ \Delta{x}\rightarrow 0}f(x_0)  \\
\end{aligned}
$$

<font color=red>------------------------------------------------------------------------------------------------------------------------</font>

<center> 不会 </center>

> 3.  $\displaystyle{证明:方程x^n + x^{n-1} + \dots + x^2 + x = 1在(0,1)内必有唯一实根x_n ,求\lim\limits_{n\rightarrow \infty}x_n}$

$$
\begin{aligned}
& set \quad f(x) = x^n + x^{n-1} + \dots + x^2 + x - 1  \\
& f(0) = -1 , f(1) =  n -1 \\
& if \quad  n \rightarrow \infty  ; then f(1) > 0 \\
& 根据零点定理: x_n  \in (0,1) 必有f(x_n) = 0 \\
&  \\
\end{aligned}
$$

<font color=red>------------------------------------------------------------------------------------------------------------------------</font>

> 4.  $\displaystyle{set \quad 函数f(x)在[-1,1]上有定义，且满足x \leq f(x) \leq x^3 + x , -1 \leq x \leq 1 } \\ \displaystyle{ 证明f'(0) } 存在,且f'(0) = 1$

$$
\begin{aligned}
& \because 0 \leq f(0) \leq  0  \\
& \therefore f(0) = 0 \\
& \because \lim\limits_{x\rightarrow 0}x \leq \lim\limits_{x\rightarrow 0}f(x) \leq \lim\limits_{x\rightarrow 0}(x^3 + x) \\
& \therefore \lim\limits_{x\rightarrow 0}f(x) = f(0)  =0 \\
& \\
& f'(0)  = \lim\limits_{x\rightarrow 0}\frac{f(x) - f(0)}{x} = \lim\limits_{x\rightarrow 0}\frac{f(x)}{x}\\
& \because \lim\limits_{x\rightarrow 0}\frac{x}{x} \leq \lim\limits_{x\rightarrow 0}\frac{f(x)}{x} \leq \lim\limits_{x\rightarrow 0}\frac{x^3 + x}{x} => 1 \leq \lim\limits_{x\rightarrow 0}\frac{f(x)}{x} \leq \lim\limits_{x\rightarrow 0}(x^2 + 1 )\\
& \therefore f'(0) = 1 \\
\end{aligned}
$$

> 5.  $\displaystyle{set \quad x \in (0,1) ,证明(1+x)\ln{^2 (1+x)} < x^2}$

$$
\begin{aligned}
& set \quad f(x) = (1+x)\ln{^2(1+x)} - x^2 \\
f'(x)& = (1+x)'\ln{^2(1+x)} + (1+x)2\ln{(1+x)}\frac{1}{1+x}(1+x)' - 2x \\
& =\ln{^2(1+x)} + 2\ln{(1+x)} - 2x \\
& f'(0) = 0 \\
f''(x)& = 2\ln{(1+x)}\frac{1}{1+x} + 2 \frac{1}{x+1} - 2  \\
& = \frac{2\ln{(1+x)} +2}{x+1} - 2\\
& f''(0) = 0\\
f'''(x)& = \frac{(2 \frac{1}{x+1})(x+1) - [2\ln{(1+x)}+2]}{(x+1)^2} \\
& = \frac{- 2\ln{(1+x)}}{(x+1)^2}\\
& f'''(0) = 0 并且 f'''(x) < 0 \\
& \therefore 函数f(x)在x \in (0,1)处单调递减 \\
& \therefore f(x) < 0 => (1+x)\ln{^2(1+x)} < x^2\\
\end{aligned}
$$
