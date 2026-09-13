"""
Generator factory module
"""
import json
from typing import Dict, Type, Any

import torch
from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig, QwenImageEditPipeline


class GeneratorRegistryMeta(type):
    """
    Метакласс, который позволяет вызывать базовый класс как функцию
    для десериализации (например, BaseGenerator(json_data)).
    """
    def __call__(cls, *args, **kwargs):
        if cls.__name__ != "BaseGenerator":
            return super().__call__(*args, **kwargs)

        if not args and not kwargs:
            return super().__call__(*args, **kwargs)

        return cls.deserialize(*args, **kwargs)


class BaseGenerator(metaclass=GeneratorRegistryMeta):
    """
    Базовый класс. Он автоматически регистрирует всех наследников
    и десериализует их при вызове BaseGenerator(данные).
    """
    _registry: Dict[str, Type['BaseGenerator']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def deserialize(cls, raw_data: Any) -> 'BaseGenerator':
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


class QwenImageEditQ2Generator(BaseGenerator):
    """
    QwenImageEditQ2Generator class
    """
    def __init__(self, gguf_model_url,
                 num_inference_steps=1,
                 true_cfg_scale=3.5,
                 seed=0):
        base_model_id = "Qwen/Qwen-Image-Edit"

        print("Загрузка трансформера из GGUF (Q2_K квантование)...")
        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_model_url,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_model_id,
            subfolder="transformer"
        )

        print("Инициализация пайплайна...")
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )

        self.pipeline.enable_model_cpu_offload()

        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.seed = seed

    def generate(self, img, prompt, negative_prompt):
        print("Генерация изображения...")
        with torch.inference_mode():
            result = self.pipeline(
                image=img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.num_inference_steps,
                true_cfg_scale=self.true_cfg_scale,
                generator=torch.manual_seed(self.seed)
            )

        return result.images[0]


def create_generator_from_cfg_file(inference_config):
    """
    Create generator from cfg file
    """
    with open(inference_config, 'r', encoding='utf-8') as f:
        data = json.load(f)
    generator = BaseGenerator(data)

    return generator
