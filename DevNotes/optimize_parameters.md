# Оптимизация параметров

$$
\frac{d}{dt}Y(t) = F(Y(t),t,\theta)
$$

$$
J(t) = 
        \begin{pmatrix}
        |\frac{\partial y_{1}}{\partial \theta_{1}}|_{t}| & |\frac{\partial y_{1}}{\partial \theta_{2}}|_{t}| & \dots &  |\frac{\partial y_{1}}{\partial \theta_{k}}|_{t}|\\
        |\frac{\partial y_{2}}{\partial \theta_{2}}|_{t}| & |\frac{\partial y_{2}}{\partial \theta_{2}}|_{t}| & \dots &  |\frac{\partial y_{2}}{\partial \theta_{k}}|_{t}|\\
        \vdots \\
        |\frac{\partial y_{n}}{\partial \theta_{2}}|_{t}| & |\frac{\partial y_{n}}{\partial \theta_{2}}|_{t}| & \dots &  |\frac{\partial y_{n}}{\partial \theta_{k}}|_{t}|\\
        \end{pmatrix}
$$

$$
    p = (\argmax_{}{J^{1}(t)},\dots,\argmax_{}{J^{n}(t)}) \\
    \mathcal{L} = \frac{1}{n(k-1)} \sum_{i}^{n} \sum_{j \neq p_{i}}^{k} \rho (J_{j}^{i},J_{p_{i}}^{i}) \\ 
    \mathcal{L} \sim \min_{\vec{\theta}}{}
$$

Смысл данной функции $\mathcal{L}$ потерь - степень малости вклада (модуля производной по параметру) параметов в изменение концентраций в момент времени $t$.  
С точки зрения ДУ - чем меньше эта функция потерь, тем сильнее система связана, параметры "проникают" в большее число уравнений и дают вклад в эти уравнения.  
Мерика 
$$
\mathcal{M} = \frac{100}{nk}\sum_{i=1}^{n} \sum_{j=1}^{k} [J_{j}^{i}(t)=0]
$$
показывает долю в процентах неработающих параметров в системе в момент времени $t$

На графиках изображён процесс случайного поиска минимума функции $\mathcal{L}$ методом Монте-Карло.

![Alt text](image.png)

# Список неработающих параметров.
![Alt text](image-1.png)
# Примеры отранжированных по важности параметров после оптимизации.  
Слева - уравнение. Справа - отранжирвоанные по убыванию важности параметры, участвующие в этом уравнении.

![Alt text](image-2.png)

![Alt text](image-3.png)