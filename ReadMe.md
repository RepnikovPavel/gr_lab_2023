

# Первое действие  
```console 
git clone https://<access__token>@github.com/RepnikovPavel/gr_lab_2023.git
```

[//]: # (# Tasks )

[//]: # (> Объединить две системы уравнений)

[//]: # ()
[//]: # (>Найти диапазоны параметров $\theta$ в системе $\dot{Y}&#40;t&#41;= F&#40;Y&#40;t&#41;,t,\theta&#41;$ таким образом, )

[//]: # (>чтобы $Y&#40;t&#41;  \geq 0$, $\forall  t \in [t_{0},T]$. )

[//]: # (>>Логично предположить, что диапазоны $[\theta_{i}^{1},\theta_{i}^{2}]$ для параметров $\theta_{i},i=\overline{1,n}$  )

[//]: # (>>зависят от  стартовой точки $Y&#40;t_{0}&#41;$, например, от степени чисел.)

[//]: # ()
[//]: # (> Найти закон или законы сохранения для этой системы.)

[//]: # (> )

[//]: # (>> К примеру,  для систем, описывающих динамику механической системы)

[//]: # (>> типичны законы сохранения энергии, импульса, момента импульса и пр.)

[//]: # (>> Для систем, описывающих перенос тепла должен выполнять закон )

[//]: # (>> сохранения количества теплоты и т.д. и т.п.)

[//]: # (>)

[//]: # (>> Зачем нужно найти закон сохранения в этой системе?)

[//]: # (>> Закон сохранения позволяет использовать метод для интегрироания системы)

[//]: # (>> на большие помежутки времени. Для этого есть стандартные подходы. Однако, )

[//]: # (>> есть и экзотичные, если нужно интегрировать максимально далеко [К.Э. Плохотников, Об устойчивости гравитационной системы многих тел]&#40;https://www.mathnet.ru/links/d099d7ed2f2d8b9341571f70a7d09cf5/crm898.pdf&#41;.)

# После git clone  
```console
cd gr_lab_2023  
touch SystemV1/local_contributor_config.py
```  
записать в этот файл абсолютный путь до проекта и до папки, куда будут сохраняться промежуточные файлы:  
```python 
project_path = '/home/user/gr_lab_2023'
problem_folder = '/home/user/gr_lab_2023_data'
```

# How to setup python 
```python 
python -m venv ./path/to/new/virtual/environment
cd ./path/to/new/virtual/environment/Scripts
activate.bat
pip install -r requirements.txt
```
# как запустить код

1. создаем файл кофиг
2. создаем папку problem_folder, указанную в конфиге
3. копируем туда файл diet_Mikhail.xlsx
4. запустить SystemV1/FPC.ipynb
5. запустить SystemV1/test_system.py

# Адипоцит, Миоцит pdf

[link to pdf](https://drive.google.com/drive/u/0/folders/1h03NgDYrl5OfgVzO8GumeO7z0DfZlk1M)


# Модель Холла  
[link to info](https://drive.google.com/drive/folders/17gs2YveZCemDOVpd598RsL5xiCdNPUPI) 
