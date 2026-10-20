================================================================================
ИНСТРУКЦИЯ И ГОТОВЫЕ ПРОМПТЫ ДЛЯ GOOGLE FLOW (VEO / GENERATIVE VIDEO)
Проект: MedAI Longevity Platform
Папка ассетов: C:\Users\Administrator\Desktop\MedAI_Flow_Assets
================================================================================

В этой папке подготовлены все визуальные материалы для создания рекламного и обзорного видео:
- medai_demo_walkthrough.webm — реальная Full HD (1920x1080) видеозапись работы платформы: выбор профиля, расчёт рисков, симуляция шагов, графики XAI и траектории, узбекский язык, страница методологии.
- 01_screening_dashboard.png — главный экран скрининга с 4 метриками.
- 02_interventions_simulation.png — блок персонализированных шагов и выигрыша лет.
- 03_xai_attributions.png — вклад факторов риска (Integrated Gradients).
- 04_trajectory_20years.png — прогноз траектории риска на 20 лет.
- 05_export_fhir.png — экспорт в HL7 FHIR R4 и печатный протокол.
- 06_uzbek_localization.png — интерфейс на узбекском языке (латиница).
- 07_methodology_page.png — страница методологии и валидации.

--------------------------------------------------------------------------------
КАК ЗАГРУЖАТЬ В GOOGLE FLOW:
--------------------------------------------------------------------------------
1. В Google Flow создайте новый проект ("New Project").
2. Откройте панель "Ingredients" (или "Assets / Media") и перетащите все файлы из этой папки.
3. Используйте скриншоты как референсы первого кадра (Start Frame / Keyframe Reference) для соответствующих сцен, чтобы нейросеть не искажала надписи интерфейса.
4. Скопируйте соответствующие промпты ниже.

================================================================================
ВАРИАНТ 1: РЕКЛАМНЫЙ ТИЗЕР (30 СЕКУНД)
================================================================================

[СЦЕНА 1 | 0:00 - 0:07] Проблема
- Референс: сгенерировать кадр человека в Ташкенте.
- Промпт для Google Flow:
Cinematic close-up of a 50-year-old Central Asian man walking in a sunny modern city street in Tashkent, looking fatigued, subtle dramatic lighting. Slow-motion push-in shot, 35mm lens, f/1.8. Seamless match-cut transition into a photorealistic, glowing holographic 3D model of human heart arteries with subtle data streams flowing through them, dark modern medical-tech backdrop, 8k resolution.
- Голос за кадром (RU):
«Более половины смертей в Узбекистане связаны с болезнями сердца. Они развиваются годами без единого симптома».

[СЦЕНА 2 | 0:07 - 0:15] Знакомство с платформой
- Референс: файл 01_screening_dashboard.png (как Start Image)
- Промпт для Google Flow:
Slow cinematic push-in on a silver laptop screen resting on a clean white doctor's desk. The screen displays the MedAI Longevity Platform dashboard with 4 metric cards (Cardio Risk 18.5%, Vascular Age, Life Expectancy). Soft natural daylight from clinic windows, subtle camera glide, photorealistic commercial quality.
- Голос за кадром (RU):
«MedAI Longevity Platform — национальный ИИ-скрининг, который оценивает угрозу инфаркта и диабета ещё до того, как они наступят».

[СЦЕНА 3 | 0:15 - 0:23] Симуляция шагов и выигрыш лет
- Референс: файл 02_interventions_simulation.png
- Промпт для Google Flow:
Macro camera pan over an interactive clinical dashboard showing lifestyle intervention cards: smoking cessation and blood pressure control. Digital counter and green badges highlighting lifespan extension: minus 4.2% risk and plus 5.1 years vascular rejuvenation. Smooth high-tech UI transitions, sleek aesthetics.
- Голос за кадром (RU):
«Нейросеть не просто ставит оценку — она рассчитывает, сколько лет жизни вернёт каждый шаг: отказ от курения или контроль давления».

[СЦЕНА 4 | 0:23 - 0:30] Финал и статус
- Референс: файл 06_uzbek_localization.png или сцена с доктором
- Промпт для Google Flow:
Warm cinematic medium shot of a grateful patient shaking hands with an Uzbek doctor in a bright modern clinic. Camera gently dollies back, bokeh lighting. Final transition to clean floating logo "MedAI Longevity Platform" with Uzbek subtitle "Milliy AI Skrining Tizimi", deep navy background, elegant typography.
- Голос за кадром (RU):
«MedAI: доказательная медицина и искусственный интеллект на страже долголетия Узбекистана».

================================================================================
ВАРИАНТ 2: ОБЗОРНЫЙ ПРОДУКТОВЫЙ ЭКСПЛЕЙНЕР (90 СЕКУНД)
================================================================================

[БЛОК 1 | 0:00 - 0:20] Экспресс-скрининг
- Референс: 01_screening_dashboard.png
- Промпт:
First-person POV seated at a modern clinic desk. Fast typing of blood test values into a clean web interface: cholesterol in mmol/L, systolic blood pressure 142 mmHg. Camera rack focuses to reveal real-time computation badge. Crisp clinical UI design.
- Диктор (RU):
«В поликлиниках у семейных врачей нет времени на сложные ручные формулы. В MedAI достаточно ввести базовые анализы — система автоматически производит точный расчёт за считанные секунды».

[БЛОК 2 | 0:20 - 0:45] 4 показателя здоровья
- Референс: 01_screening_dashboard.png + 04_trajectory_20years.png
- Промпт:
Smooth sequence highlighting 4 diagnostic metric cards: 10-year CHD risk percentage with uncertainty interval, vascular age gauge showing +16 years wear, actuarial life expectancy based on WHO Uzbekistan mortality tables, and diabetes screening probability. Sophisticated medical data visualization.
- Диктор (RU):
«Система рассчитывает 4 клинических ориентира: риск инфаркта с честной погрешностью нейросети, сосудистый возраст по шкале D'Agostino, вероятность диабета и ожидаемую продолжительность жизни по официальным данным ВОЗ именно для Узбекистана».

[БЛОК 3 | 0:45 - 1:10] Объяснимый ИИ и шаги
- Референс: 02_interventions_simulation.png + 03_xai_attributions.png
- Промпт:
Macro tilt shot revealing the Integrated Gradients feature attribution waterfall chart explaining systolic pressure and smoking impact. Transitioning into the personalized intervention ranking where counterfactual simulation shows instant risk reduction metrics.
- Диктор (RU):
«Никакого «чёрного ящика». Метод Integrated Gradients показывает вклад каждого фактора. А нейросеть симулирует шаги: отказ от курения омолодит сосуды на 5 лет и вернёт годы активной жизни».

[БЛОК 4 | 1:10 - 1:30] Национальная локализация и стандарт HL7 FHIR
- Референс: 05_export_fhir.png + 06_uzbek_localization.png
- Промпт:
Split view demonstration: seamless toggle into Latin Uzbek language interface with medical terms, and an interactive preview of HL7 FHIR R4 JSON standard code alongside an A4 medical report ready for clinical integration.
- Диктор (RU):
«Платформа полностью говорит на узбекском языке, формирует печатный протокол и готова к интеграции в систему здравоохранения через международный стандарт HL7 FHIR R4».
