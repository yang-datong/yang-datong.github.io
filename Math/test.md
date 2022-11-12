> $\displaystyle{实数a,b满足 a^3 = 2 - \sqrt{5}  , b^3 = \sqrt{5}  + 2 求a+b?}$

$$\begin{aligned}
& a^3 + b^3 = 4   , (ab)^3 = (2-\sqrt{5})(2+\sqrt{5}) = -1 , ab = -1 \\
& (a+b)(a^2-ab+b^2) = 4 \\
& a+b = 4 / (a^2  -ab + b^2)  = 4 / (a + b)^2  - 3ab  \\
& a+b = \frac{4}{(a+b)^2 - 3ab} \\
& 将ab = -1代入 => a+b = \frac{4}{(a+b)^2 + 3} \\
& 3(a+b) + (a+b)^3 = 4 \\
& set \quad  x=  a+b \\
& \therefore x^3 + 3x -4 =  0 \\
& (x-1)(x^2+4 + x) = 0 \\
& x = 1 ,\quad or \quad   x^2+x+4 = 0\\
& b^2 - 4ac <0 ,\therefore x^2+x+4=0无解\\
& \therefore x = 1 ,即a+b = 1
\end{aligned}$$


> 求极限 $\displaystyle{\lim\limits_{x\rightarrow +\infty }x(1-\frac{\ln{x}}{x})^x}$
$$\begin{aligned}
& set \quad \lim\limits_{x\rightarrow +\infty }\ln{f(x)} = \lim\limits_{x\rightarrow +\infty }\ln{x} + x\ln{(1- \frac{\ln{x}}{x})}  \\
& 以下为缩写\lim{}  \\
& \ln{x} + x \ln{(1-\frac{\ln{x}}{x})} = x[\frac{\ln{x}}{x} + \ln{(1-\frac{\ln{x}}{x})}] = x[\ln{(1- \frac{\ln{x}}{x}) - (- \frac{\ln{x}}{x})}]\\
& set \quad t = - \frac{\ln{x}}{x}  \therefore t \rightarrow 0\\
& \therefore f(x) = x[\ln{(1+t) - t}] = x \cdot -\frac{1}{2}t^2 = x \cdot -\frac{1}{2} \cdot (-\frac{\ln{x}}{x})^2  =  - \frac{x}{2} \cdot \frac{(\ln{x})^2}{x^2} = - \frac{(\ln{x})^2}{2x}\\
& \therefore \lim\limits_{x\rightarrow \infty}f(x) =  \frac{1}{2} \cdot \lim\limits_{x\rightarrow +\infty } - \frac{\ln{^2x}}{x}  \\
& \because x >> \ln{^2x}\\
& \therefore \lim\limits_{x\rightarrow +\infty }-\frac{\ln{^2x}}{x} = 0 \\
& \therefore \lim\limits_{x\rightarrow +\infty }\ln{f(x)} = 0 \\ 
& \therefore \lim\limits_{x\rightarrow +\infty }{f(x)} = 1  \\
& \because \ln{(1 - \frac{\ln{x}}{x})}^x = \ln{(1 + (-\frac{\ln{x}}{x}))}^{- \frac{\ln{x}}{x} \cdot  - \frac{x}{\ln{x}} \cdot x} = e^{-\ln{x}} = \frac{1}{e^{\ln{x}}} = \frac{1}{x} \\
\end{aligned}$$
https://www.bilibili.com/video/BV1ag411X7rE?spm_id_from=333.1007.tianma.2-3-6.click&vd_source=1ec51cb8123536a0bf872aa061240412

$\displaystyle{\lim\limits_{x\rightarrow 0}[x - \ln{(1+x)}] = \frac{1}{2}x^2}$

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>












