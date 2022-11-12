## 选择题
> 1. $\displaystyle{下列函数相同的是}$

$$\begin{aligned}
& A. \quad \lg{x^2} = 2\lg{x},x \leq 0 定义域不同[error] \\
& B. \quad x=\pm ,\sqrt{x^2} = +  [error]\\
& C. \quad f(x) = g(x) = x\sqrt[3]{x-1}  \\
& D. x \neq 1 [error]\\
\end{aligned}$$

>2. $\displaystyle{exitst=>f(x+1) = \frac{x}{x+1} ,求f^{-1}(x+1)}$

$$\begin{aligned}
& f(x) = \frac{x-1}{x} , x = \frac{x-1}{f(x)}\\
& => set \quad  f(x) = y , xy = x-1 , x - xy = 1 , x(1-y)=1 \\
& \therefore  x= \frac{1}{1-y} \\
& \therefore f^{-1}(x) = \frac{1}{1-x} \\
& \therefore f^{-1}(x+1)  = \frac{1}{1-(x+1)} = \frac{1}{-x}\\
\end{aligned}$$

> 3.  $\displaystyle{命题：}$

$\displaystyle{A.\quad if=>\lim\limits_{n\rightarrow \infty}|U_n| = a,则\lim\limits_{n\rightarrow \infty}U_n = a}$
$$\begin{aligned}
& set \quad  U_n = -1 \\
& \lim\limits_{n\rightarrow \infty}U_n = no \quad exitst [no]
\end{aligned}$$

$\displaystyle{B.\quad set=>\{x_n\}为任意数列,\lim\limits_{n\rightarrow \infty}y_n = 0 , 则\lim\limits_{n\rightarrow \infty}x_ny_n = 0}$
$$\begin{aligned}
& set	\quad x_n =  n , y_n = \frac{1}{n} \\
& \therefore \lim\limits_{n\rightarrow \infty}x_ny_n = 1 [no]
\end{aligned}$$

$\displaystyle{C.\quad if \quad \lim\limits_{n\rightarrow \infty}x_ny_n = 0 , 则必有\lim\limits_{n\rightarrow \infty}x_n = 0 或\lim\limits_{n\rightarrow \infty}y_n = 0}$
$$\begin{aligned}
& set\quad x_n = 1+(-1)^n , y_n = 1-(-1)^n \\
& \lim\limits_{n\rightarrow \infty}x_ny_n = 1-[(-1)^n]^2 = 0 \\
& \therefore \lim\limits_{n\rightarrow \infty}x_n或\lim\limits_{n\rightarrow \infty}y_n = no\quad  exitst[no] \\
\end{aligned}$$

$\displaystyle{D.\quad \{x_n\}收敛于a的充分必要条件是它的任意子数列都收敛于a}$

>4. $\displaystyle{set \quad  f(x) = \begin{cases} &2x-1 , x>0 \\ & 0 , x = 0 \\ &1 + x^2, x<0 \end{cases}} , \lim\limits_{x\rightarrow 0}f(x)=?$

$$\begin{aligned}
& \because x \rightarrow 0  \\
& \therefore x\rightarrow 0^+ \quad or \quad  x\rightarrow 0^- \\
& x\rightarrow 0^- => \lim\limits_{x\rightarrow 0}f(x) = 1+x^2=1 \\
& x\rightarrow 0^+ => \lim\limits_{x\rightarrow 0}f(x) = 2x-1=-1 \\
& \therefore \lim\limits_{x\rightarrow 0}f(x) = no\quad exitst \\
\end{aligned}$$

>5. $\displaystyle{set\quad f(x)和\phi (x)在(-\infty ,+\infty )有定义，f(x)为连续函数}，\\ \displaystyle{且f(x)\neq  0 , \phi (x)有间断点}$

$$\begin{aligned}
& \because f(x)为连续函数 \\
& set \quad \phi(x) = \begin{cases} & 2,x\geq 1 \\ &-2 ,x<1 \\ \end{cases},f(x) = x^2 + 2
\end{aligned}$$


$\displaystyle{A.\quad \phi [x^2+2]为连续[no]}$

$\displaystyle{B.\quad [\phi (2)]^2 = 4,没有断点[no]}$

$\displaystyle{C.\quad f[\phi (x)] 没有断点[no]}$

$\displaystyle{D. \quad set \quad f(x) \cdot \frac{\phi (x)}{f(x)} =  \phi (x),\because \phi (x)为间断函数[no] \therefore \frac{\phi (x)}{f(x)}为间断函数}[yes]$

>6. $\displaystyle{set \quad f(x) = \begin{cases}&e^x,x<0 \\&a+x+2 , x\geq 0 \\ \end{cases}}在R上连续，a=?$
$$\begin{aligned}
& \because 在R上连续 \\
& \therefore \lim\limits_{x\rightarrow 0^-}f(x) = \lim\limits_{x\rightarrow 0^+}f(x) = f(x) \\
& \therefore \lim\limits_{x\rightarrow 0^+}(a+x+2) = \lim\limits_{x\rightarrow 0^-}e^x = 1 \\
& \therefore  a = -1 \\
\end{aligned}$$

>7. $\displaystyle{exitst\quad f(x)的连续区间为[0,1),则f[\ln{(x+1)}]的连续区间为?}$ 
$$\begin{aligned}
& \therefore f(x) 连续区间[0,1) \\
& \therefore 0 \leq \ln{(x+1)} < 1 \\
& \because 0\leq  \ln{(x+1)}  => x \geq  0 \\
& \because \ln{(x+1)} < 1  => 0< x+1 < e  => -1 < x < e-1\\
& \therefore  0 \leq  x  < e-1 => [0,e-1) \\
\end{aligned}$$

>8. $\displaystyle{set \quad g(x) =  \begin{cases} &2-x,x\leq 0 \\ &x+2,x>0 \\ \end{cases}, f(x) = \begin{cases} &x^2,x<0 \\ &-x,x\geq 0 \\ \end{cases},则g[f(x)] = ?}$
$$\begin{aligned}
& if \quad x<0 => g[f(x)] = g(x^2) = x^2+2  \\
& if \quad x\geq 0 => g[f(x)] = g(-x) = 2-(-x)  = 2+x \\
\end{aligned}$$

>9. $\displaystyle{set \quad f(x) = \ln{\frac{1}{|x-2|}} ,then \quad x = 2  \quad is \quad f(x)? }$
$$\begin{aligned}
& \because x = 2  \\
& \therefore \lim\limits_{x\rightarrow 2^+}f(x) = \infty = no \quad exitst \\
& 第二类
\end{aligned}$$

>10. $\displaystyle{if \quad x\rightarrow 0时，无穷小量\sin{2x} - 2\sin{x}是x的?阶无穷小量}$
$$\begin{aligned}
& 题意得=> \lim\limits_{x\rightarrow 0}\frac{sin{2x} - 2sin{x}}{x^k} = C \\
& \sin{2x} - 2\sin{x} =  2\sin{x}\cos{x} - 2\sin{x} = 2\sin{x}(\cos{x} - 1) \\
& -2\sin{x}(1-\cos{x}) = -2\sin{x}\frac{1}{2}x^2 = -x^2 \cdot sinx = -x^3\\
& \therefore k = 3 \\
\end{aligned}$$


## 填空题
>1. $\displaystyle{exitst \quad f(x) = e^{x^2} , f[\phi (x)] = 1-x ,且\phi(x) \geq 0 ,\phi(x)=?}$
$$\begin{aligned}
& \because f(x) = e^{x^2} => f[\phi(x)] = e^{\phi(x)^2} = 1-x \\
& \therefore e^{2\phi(x)} = 1-x  => \ln{(1-x)} = \phi(x)^2\\
& \phi(x) = \pm\sqrt{\ln{(1-x)}} \\
& \because \phi(x) \geq  0  \\
& \therefore \phi(x) = \sqrt{\ln{(1-x)}}  \\
\end{aligned}$$

>2. $\displaystyle{\lim\limits_{n\rightarrow \infty}(1+\frac{2}{n}+\frac{2}{n^2})^n=?}$
$$\begin{aligned}
& set \quad a_n = \sum_{i = 1}^{n} (1+\frac{2}{n^i})^n \\
& X_n(Min) = \sum_{i=1}^{n} (1+\frac{2}{n^n})^n  \\
& Y_n(Max) = \sum_{i=1}^{n} (1+\frac{2}{n})^n \\
& \lim\limits_{n\rightarrow \infty}X_n = (1+\frac{2}{n^n})^n = e^2 \\
& \lim\limits_{n\rightarrow \infty}Y_n = (1+\frac{2}{n})^n = e^2 \\
& \therefore X_n < a_n < Y_n \\
& \therefore \lim\limits_{n\rightarrow \infty}X_n = \lim\limits_{n\rightarrow \infty}Y_n = \lim\limits_{n\rightarrow \infty}a_n = 1
\end{aligned}$$

>3. $\displaystyle{\lim\limits_{x\rightarrow n}\frac{\sin{x}-\sin{n}}{x-n}}=?$
$$\begin{aligned}
& 洛: f(x) = \sin{x}-\sin{n} = f'(x) = \cos{x}  \\
& g(x) = x- n = g'(x)  = 1 \\
& \therefore \lim\limits_{x\rightarrow n} \frac{f(x)}{g(x)} = \cos{x} \\
\end{aligned}$$

>4. $\displaystyle{\lim\limits_{x\rightarrow 0}\frac{\ln{\cos{\alpha x}}}{\ln{\cos{ \beta x}}}( \beta \neq 0)=?}$
$$\begin{aligned}
& 等阶无穷小解法：\\
& \because x\rightarrow 0 => \ln{\cos{ \alpha x}} \rightarrow 0 \\
& \therefore \lim\limits_{x\rightarrow 0}(\ln{\cos{ \alpha x}}) = \lim\limits_{x\rightarrow 0}(\ln{[(\cos{ \alpha x}-1) + 1]}) = \cos{ \alpha x} + 1  \\
& =\lim\limits_{x\rightarrow 0}(-(1-\cos{ \alpha x})) = - \frac{1}{2}( \alpha x)^2 \\
&\therefore  \lim\limits_{x\rightarrow 0}f(x) = \frac{ \alpha ^2}{ \beta ^2} \\
& \\
& 洛必达解法：\\
& 洛: f(x) = \ln{\cos{ \alpha x}} = f'(x) = \cos{ \alpha x} = -atan(ax) = -a^2sec^2(ax) = -a^2 \\
& \therefore = \frac{a^2}{b^2}\\
\end{aligned}$$
>5. $\displaystyle{set \quad f(x) 在(-\infty ,+\infty )有定义,且f(x)\neq 0 ,对于任意实数x、y都有f(xy) = f(x)f(y) ,} \\ \displaystyle{\\ 则f(2008)=?}$
$$\begin{aligned}
& set \quad y = 0  => f(0) = f(x)f(0) => f(x) = \frac{f(0)}{f(0)}\\
& \because f(x) \neq 0且f(x)存在定义 \therefore f(0) \neq  = 0 \\
& \therefore f(x) = 1 => f(2008) = 1 \\
\end{aligned}$$

<font color="red">

>6. $\displaystyle{set \quad f(x)的定义域为D=\{x|x \in R,x \neq 0,且x\neq 1\},且满足f(x) + f(\frac{x-1}{x}) = 1+x,}\\ \displaystyle{then \quad f(x)=?}$
$$\begin{aligned}
& f(x) + f(1 - \frac{1}{x}) = 1 + x\\
& set \quad x = \frac{1}{x}\\
& f(\frac{1}{x})  + f(1-x) = 1+\frac{1}{x}\\
&  \\
&  \\
\end{aligned}$$

</font>

>7. $\displaystyle{set \quad f(x) = a^x (a>0 , a\neq 1),then \quad \lim\limits_{n\rightarrow \infty}\frac{1}{n^2} \ln{[f(1)f(2)\dots f(n)]} = ?}$
$$\begin{aligned}
& set \quad a_n = \sum_{i=1}^{n} \frac{1}{n^2}\ln{a^{\frac{i(i+1)}{2}}} \\
& set \quad X_n = \sum_{i=1}^{n}\frac{1}{n^2}\ln{a} \\
& set \quad Y_n = \sum_{i=1}^{n}\frac{1}{n^2}\ln{a^{\frac{n(n+1)}{2}}} \\
& \because a>0 ,a\neq 1 \\
& \therefore \lim\limits_{n\rightarrow \infty}Y_n = \lim\limits_{n\rightarrow \infty}\frac{1}{n^2}\ln{a^{\frac{n(n+1)}{2}}}=\lim\limits_{n\rightarrow \infty}\frac{n(n+1) /2 \cdot \ln{a}}{n^2} = \lim\limits_{n\rightarrow \infty}\frac{(n+1) \cdot \ln{a} }{2n} =\frac{\ln{a}}{2} \\
& \therefore \lim\limits_{n\rightarrow \infty}a_n = \lim\limits_{n\rightarrow \infty}Y_n  = \frac{\ln{a}}{2}
\end{aligned}$$

>8. $\displaystyle{y=f(x)是最小周正期为5的偶函数，if \quad f(-1)= 1 , then \quad f(4)=?}$
$$\begin{aligned}
& \because T = 5 => f(5) = 0 ,且为偶函数\\
& \therefore f(-1 + 5) = f(-1) = 1 \\
& \because f(4) = f(-1+5) => \therefore  f(4)=1
\end{aligned}$$

>9. $\displaystyle{if\quad f(\ln{x}) = x ,then \quad f(3)=?}$
$$\begin{aligned}
& \because f(\ln{x}) = x  \\
& \therefore f(x) = e^x \\
& \therefore f(3) = e^3 \\
\end{aligned}$$

>10. $\displaystyle{exitst \quad 数列a_1 = 2 ,a_2 = 2+\frac{1}{2} ,a_3 = 2+\frac{1}{2+\frac{1}{2}},.....的极限存在，求极限}$
$$\begin{aligned}
& set \quad a_n = \sum_{i=2}^n 2+\frac{1}{a_{n-1}}\\
& if \quad n>1 \\
& \lim\limits_{n\rightarrow \infty}a_n = 2 + \lim\limits_{n\rightarrow \infty}\frac{1}{a_{n-1}} => a = 2+ \frac{1}{a} \\
&=> a^2 -1 = 2a  \\
&=>a^2 -2a -1 = 0 \\
&=> (a-1)^2 = 2 => a-1 = \pm \sqrt{2}\\
&=> a = 1-\sqrt{2} (舍) , a= 1+\sqrt{2}\\
& \therefore \lim\limits_{n\rightarrow \infty}a_n = 1+\sqrt{2} \\
& if \quad n=1\\
& \lim\limits_{n\rightarrow \infty}a_n = 1\\
& \lim\limits_{n\rightarrow \infty}a_n = \begin{cases} 1+\sqrt{2} &,x>1\\ \quad 1&,x=1 \\ \end{cases}
\end{aligned}$$


## 简答题
>1. $\displaystyle{f(x)为二次函数,f(x+1) + f(x-1) = 2x^2 - 4x ,求f(x)}$
$$\begin{aligned}
& 二次函数:ax^2 + bx + c = 0 \\
& set \quad x = x+1 \\
& a(x+1)^2 + b(x+1) + c = 0 \\
& set \quad x = x-1 \\
& a(x-1)^2 + b(x-1) + c = 0\\
& \because f(x+1) + f(x-1) = 2x^2 - 4x\\
& \therefore [a(x+1)^2 + b(x+1) + c] + [a(x-1)^2 + b(x-1) +c ] = 2x^2-4x \\
& =ax^2 + 2ax + a + bx + b + c + a^2 - 2ax + a + bx - b + c = 2x^2-4x \\
& =2ax^2 + 2a + 2bx + 2c = 2x^2 - 4x\\
& =2ax^2 + 2bx + 2(a+c) = 2x^2 - 4x\\
& \therefore 2a = 2 , 2b = -4 ,a+c = 0\\
& \therefore a = 1, b = -2 , c = -1\\
& f(x) = x^2 - 2x - 1
\end{aligned}$$

>2. $\displaystyle{set\quad a < b , f(x)对任意实数x , 有f(a-x) = f(a+x) , f(b-x) = f(b+x)}\\\displaystyle{证明:f(x)是以2b -2a 为周期的周期函数}$
$$\begin{aligned}
& 反证：set \quad f(x) = f(x + 2b -2a) \\
& \therefore f(a-x) = f(a-(x+2b-2a)) = f(a+(x+2b-2a)) = f(2b-a+x)\\
& =f(b+(b-a+x)) = f(b-(b-a+x)) = f(a+x)\\
\end{aligned}$$

>3. $\displaystyle{set\quad f(x) = \lim\limits_{n\rightarrow \infty}\frac{\ln{(e^n + x^n)}}{ n} (x > 0), 求f(x)=?}$
$$\begin{aligned}
& if \quad x > e \\
& \ln{(e^n + x^n)} = \ln{[x^n(\frac{e^n }{x^n}+ 1)]} = \ln{x^n} + \ln{(\frac{e^n}{x^n} + 1)} \\
& \therefore \lim\limits_{n\rightarrow \infty}f(x) = (\ln{x^n} + (\frac{e}{x})^n) \cdot \frac{1}{n} = \ln{x^n} \cdot \frac{1}{n} = \ln{x} \\
& if \quad x < e  \\
& \ln{(e^n + x^n)} = \ln{[e^n (1 + \frac{x^n}{e^n}) ]} = \ln{e^n} + \ln{(1 + \frac{x^n}{e^n})} = n+\ln{(1+\frac{x^n}{e^n})} \\
& \therefore \lim\limits_{n\rightarrow \infty}f(x) = \frac{n + \frac{x^n}{e^n}}{ n }  =  1 + (\frac{x}{e})^n \cdot \frac{1}{n} = 1+0 \cdot  0 = 1\\
& \therefore f(x) = \begin{cases} \ln{x} &, x>e \\ 1 &,0 < x \leq e \\ \end{cases} \\
\end{aligned}$$

>4. $\displaystyle{if \quad f(x) 在 [0,a](a>0)上连续，且f(0)=f(a) , 则方程f(x) = f(x+\frac{a}{2}) 在(0,a)内至少一个实根}$
$$\begin{aligned}
& !!!!有实根表示b^2-4ac > 0 即f(x_1) = f(x_2)!!!!\\
& \\
& set \quad g(x) = f(x) - f(x+ \frac{a}{2}) \\
& g(0) = f(0) - f(\frac{a}{2}) \\
& g(\frac{a}{2}) = f(\frac{a}{2}) - f(a) = f(\frac{a}{2}) - f(0)  = -[f(0) - f(\frac{a}{2})] \\
& \therefore g(0) = - g(\frac{a}{2}) \\
& if \quad g(0) = -g(\frac{a}{2}) = 0,则f(0) = f(\frac{a}{2}) ,则x=\frac{a}{2}为实根\\
& if \quad g(0) \neq  0 ,则g(0) \cdot g(\frac{a}{2}) < 0 \\
& 根据零点定理，函数连续且g(0) = -g(\frac{a}{2}) , 存在一点x_0 \in (0,\frac{a}{2})使得g(x_0) = 0\\
& \therefore g(x_0) = f(x_0) - f(x_0 + \frac{a}{2}) = 0 \\
& \therefore x_0 为(0,a)中f(x) = f(x+ \frac{a}{2})的实根
\end{aligned}$$

>5. $\displaystyle{set \quad x_1 = \sqrt{2}  , x_n = \sqrt{2+x_{n-1}}(n\geq 2) }求\lim\limits_{n\rightarrow \infty}x_n$
$$\begin{aligned}
& \because x_1 = \sqrt{2}  , x_2 = \sqrt{2 + \sqrt{2} }  \\
& set \quad x_n > x_{n-1} => \sqrt{2+x_n} > \sqrt{2+x_{n-1}}  \\
& \because n \geq 2 => \{x_n\}为单调递增,且x_n > 0  \\
& \because n -> \infty  \\
& \therefore \lim\limits_{n\rightarrow \infty}x_n = \lim\limits_{n\rightarrow \infty}x_{n-1} = a\\
& \therefore 可得出等式=> a = \sqrt{2+a} => a^2 = 2 + a => a^2-a-2 = 0\\
& \therefore 十字相乘=> (a+1)(a-2) = 0 => a = \begin{cases} &-1(舍) \\ &2 \\ \end{cases} \\
& \therefore  a = 2\\
\end{aligned}$$

