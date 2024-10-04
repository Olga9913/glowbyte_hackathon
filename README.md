# О ЧЕМ ЭТОТ КЕЙС?
Для создания кейса использовались реальные данные. В ходе решения кейса могут использоваться как предоставленные данные, так и данные из открытых источников, найденные в сети Интернет. 

Постоянный рост потребления электроэнергии один из наиболее заметных и сложных вызовов, стоящих перед современным обществом. В течение последних десятилетий спрос на электричество стремительно возрастает, и это явление имеет глобальный характер. Повышение потребления электроэнергии обусловлено несколькими факторами, которые оказывают влияние на образ жизни и технологические тенденции. 

Факторы влияющие на рост потребления:
1.Первым и наиболее значимым фактором является постоянный рост населения во всем мире.
2.Вторым фактором, содействующим росту потребления электроэнергии, является технологический прогресс и цифровизация общества.
3.Третий фактор связан с расширением применения электроэнергии в промышленности.

# ОПИСАНИЕ ПРОБЛЕМЫ

Прогнозирование электропотребления представляет собой сложную задачу из-за нескольких ключевых проблем:

1. Неопределенность: Будущее поведение потребителей и их потребности в электроэнергии не всегда можно предсказать точно. Различные факторы, такие как экономические изменения, социокультурные тенденции, технологические прорывы и политические решения, могут существенно повлиять на потребление электроэнергии, и невозможно учесть все возможные варианты в прогнозах.
2. Сезонные и временные изменения: Электропотребление часто подвержено сезонным и временным колебаниям. Например, летом потребление может возрасти из-за использования кондиционеров, а в праздничные дни или выходные потребление может снизиться. Прогнозирование этих изменений требует учета множества факторов и данных.
3. Развивающиеся технологии: Быстрое развитие новых технологий, таких как электромобили, возобновляемые источники энергии и энергоэффективные устройства, меняет пейзаж потребления электроэнергии. Прогнозирование, как и когда эти технологии будут внедрены, представляет вызов для экспертов.
4. Энергетическая политика и регулирование: Политические решения и изменения в законодательстве в области энергетики могут существенно повлиять на спрос на электроэнергию.
5. Глобальные факторы: Электроэнергия является частью мировой экономики, и множество глобальных факторов, таких как изменения климата, геополитические конфликты и экономические кризисы, могут оказать влияние на ее потребление.

# ЗАДАЧА «ОБЩЕЕ ПОТРЕБЛЕНИЕ ЗА СУТКИ

## Задача

Разработка модели прогнозирования общего энергопотребления региона на сутки, в МВт*ч

## Цель

Разработать надежную и точную модель прогнозирования объема энергопотребления на сутки для Калининградской области с использованием доступных исторических данных и соответствующих переменных

## Описание задачи

В данной задаче необходимо разработать предиктивную модель, которая позволит прогнозировать энергопотребление региона на основе имеющихся данных о потреблении электроэнергии в прошлом и соответствующих факторах, влияющих на потребление энергии. Модель должна быть способна учесть сезонные, временные и другие зависимости для более точного прогноза. Модель должна предсказывать общее потребление региона на 1 сутки.

## Замечание (!)

Если мы делаем прогноз на сегодня, то у нас есть все данные за вчера и более ранние, но нет данных из будущего. Учесть это при конструировании признаков для модели.

# ДАТАСЕТ

date – дата
time – время, время представлено в диапазоне 0 – 23, что означает 24
часа в сутках
target – Фактическое потребление на указанную дату
temp – фактическая температура на указанную дату
temp_pred – прогноз температуры на указанную дату
weather_fact – фактическая погода на указанную дату
weather_pred – прогноз погоды на указанную дату

## Обучающий датасет
Датасет для обучения. Содержит данные за
период
### 2019-01-01 – 2023-03-31

## Публичный тестовый датасет
Представляет
продолжение
датасета. Содержит данные за период
### 2023-04-01 – 2023-07-31

## Приватный тестовый датасет
Содержит данные за период
### 2023-08-01 – 2023-09-30

# МЕТРИКИ «ОБЩЕЕ ПОТРЕБЛЕНИЕ»

## MAE (Главная метрика)
## MAPE (Вспомогательная метрика)
## R2-score (Вспомогательная метрика)
