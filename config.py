import yaml
import os

CONFIG_PATH= "config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Kritik hataÇ {CONFIG_PATH}bulunamadı")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

settings= load_config()
