"""
Dataset factory module
"""
import json
import os
from typing import Dict, Type, Any
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class DatasetRegistryMeta(type):
    """
    Метакласс, который позволяет вызывать базовый класс как функцию 
    для десериализации (например, BaseDataset(json_data)).
    """
    def __call__(cls, *args, **kwargs):
        if cls.__name__ != "BaseDataset":
            return super().__call__(*args, **kwargs)

        if not args and not kwargs:
            return super().__call__(*args, **kwargs)

        return cls.deserialize(*args, **kwargs)


class BaseDataset(metaclass=DatasetRegistryMeta):
    """
    Базовый класс. Он автоматически регистрирует всех наследников 
    и десериализует их при вызове BaseDataset(данные).
    """
    _registry: Dict[str, Type['BaseDataset']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def deserialize(cls, raw_data: Any) -> 'BaseDataset':
        """
        Основной метод десериализации. Принимает строку JSON или словарь.
        """
        if isinstance(raw_data, str):
            data_dict = json.loads(raw_data)
        elif isinstance(raw_data, dict):
            data_dict = raw_data.copy()
        else:
            raise TypeError("Данные должны быть строкой JSON или словарем (dict)")

        class_name = data_dict.pop("__class__", None)
        if not class_name:
            raise KeyError("В данных отсутствует обязательное поле '__class__'")

        target_class = cls._registry.get(class_name)
        if not target_class:
            raise ValueError(f"Класс '{class_name}' не зарегистрирован в системе.")

        return target_class(**data_dict)


class CustomDataset(BaseDataset, Dataset):
    """
    Custom Dataset class
    """
    def __init__(self, img_path, img_size=None):
        assert os.path.isdir(img_path), 'Image folder does not exist'

        self.img_path = img_path
        self.images = sorted(os.listdir(img_path))
        self.img_size = img_size

    def __getitem__(self, index):
        full_img_path = os.path.join(self.img_path, self.images[index])
        try:
            img = Image.open(full_img_path).convert("RGB")
            cur_img_size, _ = img.size
            if self.img_size  and self.img_size !=cur_img_size:
                img = img.resize(self.img_size, resample=Image.Resampling.LANCZOS)
        except UnidentifiedImageError:
            print(f'Не удалось загрузить изображение {full_img_path}')
            img = None

        return {"img": img, "img_name": self.images[index]}

    def __len__(self):
        return len(self.images)


def create_ds_from_cfg_file(ds_config):
    """
    Create ds from cfg file
    """
    with open(ds_config, 'r', encoding='utf-8') as f:
        data = json.load(f)
    ds = BaseDataset(data)

    return ds
