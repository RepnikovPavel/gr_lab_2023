# Подбор параметров в системе ОДУ первого порядка


# Содержание  
1. [Выкладки](#p1)
2. [Adjoint Method](#AdjointMethod)
3. [Список литературы](#literature)


<a name="p1"></a>

# Схема с градиентным спуском:  
$$ 
\theta = e^{x}+a:\mathbb{R} \rightarrow (a;+\infty)
$$

$$ 
\theta = -e^{x}+a:\mathbb{R} \rightarrow (-\infty;a)
$$

$$ 
\theta = a+\frac{b-a}{1+e^{-x}}:\mathbb{R} \rightarrow (a;b)
$$

$x \in \mathbb{R}$ варьируется как угодно, при этом параметр $\theta$ остается в заданном диапазоне. Это необходимо для схемы с градиентным спуском, которая не учитывает дополнительные ограничения на параметры. 

# Метод оптимизации параметров нормальной системы  

$$
\frac{d}{dt}Y(t) = F(Y(t),t,\theta) \\
t \in [t_{0},T+t_{0}] \\
\theta \in  \Theta=[\theta_{1}^{1},\theta_{2}^{1}] \times \dots \times [\theta_{1}^{k},\theta_{2}^{k}]
$$
___
$$
Y(t,\theta) \\
\mathcal{L} = f(Y(t,\theta)) \\
\mathcal{\theta}^{*} = \argmin_{\theta \in \Theta}{\mathcal{L}}
$$

## Если известно аналитическое решение   


1. 
$$
\theta_{i+1} = \theta_{i} - \lambda (\nabla_{\theta} \mathcal{L})_{\theta_{i}} , \: i = \overline{1,\text{last step}}
$$
2.  
$$
    \frac{\partial}{\partial \theta_{i}}\mathcal{L} = 0, i = \overline{1,\text{last index of parameter}}
$$

## Если аналитическое решение не известно  

1.  
строится сеточное решение $Y_i,i = \overline{1,\text{last index of grid}}$
$$
L = \phi(Y_i,\theta)
$$
пробема: огромное число слагаемых в итоговом выражении.  
я уже делал ровно то же самое только для системы, состоящей из двух уравнений. результаты были, но они были такими как и ожидалось. значение $L$ уменьшалось, но не до желаемого значения.  

2.  

если это не заработает, то только метод монтекарло и его вариации  
проблема: достаточно большое число параметров $\sim 10^2$.
скорее всего прийдется использовать жадный алгоритм.  
как итог решение будет не ахти.

3.  
еще один метод, пока я его не до конца понял, но выкладки уже частично воспроизвел:


<a name="AdjointMethod"></a>

## Adjoint Method  

modern proof of adjoint method ([Pontryagin et al., 1962](#pont))
<!-- $$
\mathcal{L}(Y(t,\theta)) = \mathcal{L}(Y(t_{0})  + \int_{t_{0}}^{t_{0}+t}{F(Y)  dt}
) \\ 
t \geq t_{0}
$$ -->

$$
Y(t+\epsilon) = Y(t) + \int_{t}^{t+\epsilon}{F(Y(t),t,\theta)  dt} \\
L = \phi(Y_{t+\epsilon}) = \phi (Y_{t+\epsilon}(Y_{t})) = \\ 
\frac{\partial L}{\partial Y(t)} =\frac{\partial L}{\partial Y(t+\epsilon)} \frac{\partial Y(t+\epsilon)}{\partial Y(t)} 
$$

$$
a(t) := \frac{\partial \mathcal{L}}{\partial Y(t)} \\ 
L \in \mathbb{R}, Y^{m \times 1} \rightarrow a^{1 \times m}
$$

$$
a(t) = a(t+\epsilon) \frac{\partial Y(t+\epsilon)}{\partial Y(t)} 
$$

Taylor series of $Y(t+\epsilon)$ around $Y(t)$:  
$$
Y(t+\epsilon) = Y(t) + \frac{\partial Y(t+\epsilon)}{\partial \epsilon} \epsilon +
\mathcal{O}(\epsilon^{2})= \\ 
Y(t) + \epsilon \frac{\partial }{\partial \epsilon}(Y(t) + \int_{t}^{t+\epsilon}{F(Y(t),t,\theta)  dt}) +
\mathcal{O}(\epsilon^{2}) = \\ 
Y(t) + \epsilon \frac{\partial }{\partial \epsilon}(\epsilon F(Y(t),t,\theta)) +
\mathcal{O}(\epsilon^{2})= \\ 
Y(t) + \epsilon F(Y(t),t,\theta) +
\mathcal{O}(\epsilon^{2})
$$
вообще говоря, судя по тому, что  $\frac{\partial Y(t+\epsilon)}{\partial \epsilon} = F(Y(t),t,\theta)$, получается $\mathcal{O(\epsilon^{2})}=0$. при этом слагаемое $\mathcal{O(\epsilon^{2})}$ в [статье](https://arxiv.org/pdf/1806.07366.pdf) тащат до конца.
$$
\frac{\partial Y(t+\epsilon)}{\partial Y(t)} = \frac{\partial }{\partial Y(t)}(Y(t) + \epsilon F(Y(t),t,\theta) +
\mathcal{O}(\epsilon^{2})) = I+ \epsilon \frac{\partial F(Y(t),t,\theta)}{\partial Y(t)} \\ 
I^{m \times m}
$$


$$
\frac{d a(t)}{d t} = \lim_{\epsilon \rightarrow +0}{\frac{a(t+\epsilon)-a(t)}{\epsilon}} = \\ 
\lim_{\epsilon \rightarrow +0}{\frac{a(t+\epsilon)-
a(t+\epsilon)(I+ \epsilon \frac{\partial F(Y(t),t,\theta)}{\partial Y(t)})
}{\epsilon}} = \\ 
\lim_{\epsilon \rightarrow +0}{-a(t+\epsilon) \frac{\partial F(Y(t),t,\theta)}{\partial Y(t)}}
$$

итог
$$
\frac{d a(t)}{d t} = {-a(t) \frac{\partial F(Y(t),t,\theta)}{\partial Y(t)}} \\ 
a(t)= \frac{\partial L}{\partial Y(t)}
$$
дальнейших выкладок в статье нет. и не понятно как их делать. однако в статье есть формула без вывода:  
$$
\frac{\partial L}{\partial \theta} = - \int_{t_{0}+T}^{t_{0}} a(t)^{T} \frac{\partial F(Y(t),t,\theta)}{\partial \theta}dt
$$
причем в статье в одном месте транспонирование не стоит (при выводе основного тождества), в другом - нет. формально по размерам матрицы и вектора транспонирование не должно быть. в докладах по этой теме кто-то пишет транспонирование кто-то нет. это не принципиальный момент. главное договорится, как считать производную вектора по вектору. 

[доклад](https://www.youtube.com/watch?v=76rN-dFBwr0), в котором показано, эта формула выводится из метода множителей лагранжа.

## Некоторое свойство  

[Понтрягин Математическая теория оптимальных процессов стр.92](#pont)
$$
H= \sum_{i=1}^{m}a_{i}F_{i}(Y(t),t,\theta) \\ 
\frac{d}{dt}a_{j}=-\frac{\partial H}{\partial y_{j}} \\ 
\frac{d}{dt}y_{j}=\frac{\partial H}{\partial a_{j}}
$$


<a name="literature"></a>  

# Cписок литературы

1. [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf)
2. Lev Semenovich Pontryagin, EF Mishchenko, VG Boltyanskii, and RV Gamkrelidze. The mathematical theory of optimal processes. 1962 <a name="pont"></a>
  






<style>
* {
    /* font-family: Arial, Helvetica, sans-serif;
    color: rgb(65, 65, 65); */
    -webkit-print-color-adjust: exact !important;
    color-adjust: exact !important;
    print-color-adjust: exact !important;
}

@media print {
   @page {
     margin-left: 0.8in;
     margin-right: 0.8in;
     margin-top: 0;
     margin-bottom: 0;
   }
}




</style>



