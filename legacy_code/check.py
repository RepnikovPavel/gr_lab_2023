import os

import pandas as pd
import numpy as np
import math

import MyocyteSystem.config as config
from Plotting import myplt

def pik(t,t_0,t_end,A):
    if t<=t_0 or t>t_end:
        f = 0
    elif t>t_0 and t<=(t_0+t_end)/2:
        f = 2*A*(t-t_0)/(t_end-t_0)
    elif t>(t_0+t_end)/2 and t<=t_end:
            f = 2*A*(t_end-t)/(t_end-t_0)
    return f

def piki(t,t_0,t_end,A):
    J_in = 0
    for i in range(len(A)):
        J_in = J_in + pik(t,t_0[i],t_end[i],A[i])
    return J_in


if __name__ == '__main__':
    diet = pd.read_excel(os.path.join(config.problem_folder, 'diet_Mikhail.xlsx'))
    # diet = pd.read_excel(r"C:\Users\User\PycharmProjects\gr_lab_2023\legacy_code\diet_Mikhail.xlsx")

    # Начало приёма пиши
    meal_beginning = np.array(diet.MealTime.dropna())

    # Выбор периода диеты в днях. Атоматическое определение того, сколько недель (даже не целых) занимает диета.
    diet_period = 14  # days
    weeks = math.ceil(diet_period / 7)

    awake_time = 7.5  # Час пробуждения
    assert awake_time < np.min(meal_beginning), 'Error: awake_time should be smaller than the first meal time'

    sleep_time = 23.5  # Час отбоя
    assert sleep_time > np.max(meal_beginning), 'Error: sleep_time should be bigger than the last meal time'

    # Регулирование калорийности диеты в процентах от уже установленной
    calorage = np.array([100, 100])  # ,110,120,130,140])
    assert len(calorage) == weeks, 'Error: lenth of calorage should be equal number of weeks'

    time_for_main_meals = 30  # продолжительность основных приёмов пищи в минутах
    time_for_minor_meals = 10  # продолжительность неосновных приёмов пищи в минутах
    meals_per_day = len(meal_beginning)  # кол-во приёмов пищи в день
    meal_starts = meal_beginning * 60  # время приёма пищи в минутах
    # временная метка подъема [min]. отсчет от начала главного периода.
    t_awake = [awake_time * 60 + i * 24 * 60 for i in range(diet_period)]  # время пробуждения в минутах на каждый день
    # временная метка отбоя [min]. отсчет от начала главного периода.
    t_sleep = [sleep_time * 60 + i * 24 * 60 for i in range(diet_period)]  # время отбоя в минутах на каждый день
    # время старта приема пищи в течении дня. номер стороки - номер дня. номер столбца - номер приема пищи
    t_0 = [[meal_starts[j] + i * 24 * 60 for j in range(meals_per_day)] for i in range(diet_period)]
    # время окончания приема пищи в течении дня. номер стороки - номер дня. номер столбца - номер приема пищи
    t_end = [
        [t_0[i][j] + time_for_main_meals * ((j + 1) % 2) + time_for_minor_meals * (j % 2) for j in range(meals_per_day)]
        for i in range(diet_period)]
    t_0 = np.reshape(t_0, np.size(t_0))  # время начала приёма пищи на каждый день
    t_end = np.reshape(t_end, np.size(
        t_end))  # время конца приёма пищи на каждый день ( 30 минут на обычный приём и 10 на перекус)

    # Чтение данных из таблицы
    Carbs0_, Prots0_, Fats0_, Calories0_, GL_, IL_, GI_, II_ = np.zeros((8, int(np.max(diet.N))))
    for i in range(int(np.max(diet.N))):
        # i - номер дня. в столбце "N" он может дублирвоаться. т.к. за день несколько приемов пищи. а каждая запись-строка
        # - это очередной примем пищи

        # суммарное число углеводов за iй день
        Carbs0_[i] = np.sum([np.array(diet.Carbs[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число белков за iй день
        Prots0_[i] = np.sum([np.array(diet.Protein[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число жиров за iй день
        Fats0_[i] = np.sum([np.array(diet.Fats[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # суммарное число калорий за iй день
        Calories0_[i] = np.sum([np.array(diet.Calories[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        GL_[i] = np.sum([np.array(diet.Carbs[j]) * np.array(diet.GI[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        IL_[i] = np.sum(
            [np.array(diet.Calories[j]) * np.array(diet.II[j]) for j in np.where(np.array(diet.N) == (i + 1))])
        # доп фильтр, защищающий от ошибок в данных
        if Carbs0_[i] == 0:
            GI_[i] = 0
        else:
            GI_[i] = GL_[i] / Carbs0_[i]
        if Calories0_[i] == 0:
            II_[i] = 0
        else:
            II_[i] = IL_[i] / Calories0_[i]
    # скалирование прочитанных данных в прцоентах(для игры со значениями?)
    # заполнение начальных условий на весь период диеты
    Carbs0, Prots0, Fats0, Calories0, GL, IL, GI, II = np.zeros((8, diet_period * meals_per_day))
    for i in range(diet_period * meals_per_day):
        Carbs0[i] = Carbs0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Prots0[i] = Prots0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Fats0[i] = Fats0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        Calories0[i] = Calories0_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        GL[i] = GL_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        IL[i] = IL_[i % len(Carbs0_)] * calorage[math.ceil((i + 1) / len(Carbs0_)) - 1] / 100
        GI[i] = GI_[i % len(Carbs0_)]
        II[i] = II_[i % len(Carbs0_)]
    myplt.plot_vec(x=np.arange(start=0, stop=len(Carbs0)), y=Carbs0,block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(Prots0)), y=Prots0, block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(Fats0)), y=Fats0, block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(Calories0)), y=Calories0, block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(GL)), y=GL, block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(IL)), y=IL, block=False)
    myplt.plot_vec(x=np.arange(start=0, stop=len(GI)), y=GI, block=False)
    print('last plot')
    myplt.plot_vec(x=np.arange(start=0, stop=len(II)), y=II, block=True)

