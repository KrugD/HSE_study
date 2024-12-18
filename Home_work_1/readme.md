I.Сделано:
  1) EDA:
     а) С помошью ydata-profiling построены дашборды для train и тест.
     б) Преобразован столбец torque в 2 столбца типа float: torque_Nm, torque_rpm.
     в) Преобразованы столбцы max_power, engine, mileage в числовые.
     г) Обработаны пропуски в данных (заполнены медианой).
     д) Удалены дубликаты.
     е) Построены графики, корреляции, ящики с усами.
  2) Числовые признаки стандартизированы, категориальные признаки закодированы.
  3) Построены модели (LinearRegression, Lasso, ElasticNet, Ridge)
  4) Подобраны гипермапарметры моделей с помощью GridSearchCV.
  5) Выбрана лучшая модель (по MSE, r2_score и business_metric)
  6) Построен pipeline обработки данных (обучен на трейне).
  7) Реализован сервис с помощью FastAPI.
     
II.Результаты:
  1) Лучшая модель имеет метрики на тесте:
      а) r2_score: 0.78347
      б) MSE: 124467545072.89255
      в) business_metric: Доля нужных прогнозов (отличающиеся от реальных цен на авто не более чем на 10%): 30%
     
III.Наибольший буст в качестве:
  1) Наибольший буст в качестве дало добавление закодированных категориальных переменных (r2_score вырос с 0.59138 до 0.78348)
     
IV.Что сделать не вышло:
  1) При кодировании seats, как категориальной переменной не удалось в pipeline сначала заполнить ее пропуски медианой (пока она float), а потом уже закодировать. Поэтому я сразу поменял тип seats на object и закодировал с помощью OneHotEncoder. Не удалось это, т.к. preprocessor удут последовательно и нельзя перепрыгнуть на определенном этапе в один, потом обратно в другой (по крайней мере я пока не знаю как).
  2) Не удалось в POSTMAN загрузить csv файл, чтобы проверить работу сервиса (подробнее я рассказал в видео проверки работы сервиса), поэтому я протестировал сервис в терминале с помощью: curl -X POST "http://localhost:8000/predict_csv" -F.....

Ссылка на демонстрацию работы сервиса: https://drive.google.com/file/d/17ifTmomEfPUzsfpfjkEz-1yqKaH6H3RC/view?usp=sharing

P.s. предполагаю, что на фото - кот Кантонистовой Е.О.
