# ImageAugmentation
_Аугментация изображений с помощью моделей и трансформаций_

Необходимо с помощью Qwen-Image-Edit (или аналогичной) из нескольких случайных изображений получить изображения с любым промптом, добавить аугментации с помощью средств Albumentations. Полученные результаты сохранить в формате json, в котором будет указатель на исходное изображение, промпт редактирования, указатель на финальное изображение.

## Работа с данными
Рассматривал данную задачу как задачу аугментации существующего датасета.

Первым этапом необходимо написать инструмент для получения доступа к этому датасету. В конкретном случае реализовал класс кастомного датасета `CustomDataset`, который получает изображения, загружая их из локальной папки ([исходные данные](https://github.com/laskovatr/ImageAugmentation/tree/main/images)). Для легкой масштабируемости проекта необходимо учесть различные способы получения датасета (например, может понадобиться скачивать данные из интернета). Для этого были реализованы инструменты, позволяющие получать экземпляр датасета с помощью десереализации конфигурационного файла. Таким образом, в проект можно легко интегрировать новый тип загрузчика данных.

[Десереализация](https://github.com/laskovatr/ImageAugmentation/blob/main/dataset_factory.py#L88-L96)

[Кастомный класс датасета](https://github.com/laskovatr/ImageAugmentation/blob/main/dataset_factory.py#L60-L85)

[Пример конфигурационного файла](https://github.com/laskovatr/ImageAugmentation/blob/main/configs/ds_cfg.json)

[Классы, реализующие десереализацию](https://github.com/laskovatr/ImageAugmentation/blob/main/dataset_factory.py#L11-L57)

## Работа с моделью 
Необходимо было использовать `Qwen-Image-Edit` или квантованные версии. Работая на видеокарте `NVIDIA GeForce RTX 3070 Laptop GPU`, было доступно _8GB_ VRAM. Поэтому использовал модель [Qwen_Image_Edit-Q2_K](https://huggingface.co/QuantStack/Qwen-Image-Edit-GGUF?show_file_info=Qwen_Image_Edit-Q2_K.gguf). Однако и этого было много для видеокарты, поэтому генерация для одного изображения длилась несколько часов. Поэтому и было обработано всего 3 изображения.

Так как могут понадобиться разные способы генерации (например, с помощью API, либо локально, но с использованием других фреймворков или моделей (например, FLUX, StableDiffusion и другие)), то для этого были реализованы инструменты, позволяющие получать экземпляр генератора с помощью десереализации конфигурационного файла. Таким образом, в проект можно легко интегрировать новый тип генератора.

[Десереализация](https://github.com/laskovatr/ImageAugmentation/blob/main/generator_factory.py#L107-L115)

[Кастомный класс генератора](https://github.com/laskovatr/ImageAugmentation/blob/main/generator_factory.py#L60-L104)

[Пример конфигурационного файла](https://github.com/laskovatr/ImageAugmentation/blob/main/configs/generator_cfg.json)

[Классы, реализующие десереализацию](https://github.com/laskovatr/ImageAugmentation/blob/main/generator_factory.py#L11-L57)

## Генерация
Основная логика находится в файле `images_generator.py`.

После того, как модель сгенерирует по промпту изображение, оно, при наличии конфигурационного файла аугментаций, будет дополнено еще несколькими копиями с трансформациями (трансформации и количество копий указывается в файле. [Пример конфигурационного файла аугментаций](https://github.com/laskovatr/ImageAugmentation/blob/main/configs/augmentation_cfg.json)). Для трансформаций использовалась библиотека `Albumentations`. В файл трансформаций можно записать любую из [существующих в этой библиотеке трансформаций и их сочетаний](https://albumentations.ai/explore/). 

Каждое созданное изображение сохраняется в указанную папку (например, в `results`). Также вместе с созданным изображением туда же сохраняется и JSON-файл, содержащий в себе указатель на исходное изображение, промпт редактирования, указатель на финальное изображение ([пример JSON-файла](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_1.json)).

Пример вызова скрипта из командной строки:
```
python images_generator.py --inference-config configs/generator_cfg.json \
                           --ds-config configs/ds_cfg.json \
                           --prompts prompts/prompts.yaml \
                           --output-path result \
                           --augmentation-config configs/augmentation_cfg.json
```

## Результат
В качестве датасета были использованы изображение из папки `images`. [Использовался данный промпт](https://github.com/laskovatr/ImageAugmentation/blob/main/prompts/prompts.yaml).

| Исходное изображение | Сгенерированное изображение | Трансформированное изображение №1 | Трансформированное изображение №2 | Трансформированное изображение №3 | Трансформированное изображение №4 | Трансформированное изображение №5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![Исходное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/images/cat.jpg) | ![Сгенерированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_0.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_1.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_2.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_3.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/cat_4.jpg) |

| Исходное изображение | Сгенерированное изображение | Трансформированное изображение №1 | Трансформированное изображение №2 | Трансформированное изображение №3 | Трансформированное изображение №4 | Трансформированное изображение №5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![Исходное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/images/dog.jpg) | ![Сгенерированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog_0.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog_1.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog_2.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog_3.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/dog_4.jpg) |

| Исходное изображение | Сгенерированное изображение | Трансформированное изображение №1 | Трансформированное изображение №2 | Трансформированное изображение №3 | Трансформированное изображение №4 | Трансформированное изображение №5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![Исходное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/images/jackal.jpeg) | ![Сгенерированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal_0.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal_1.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal_2.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal_3.jpg) | ![Трансформированное изображение](https://github.com/laskovatr/ImageAugmentation/blob/main/result/jackal_4.jpg) |

Сформированные JSON-файлы так же лежат в папке `result`.
