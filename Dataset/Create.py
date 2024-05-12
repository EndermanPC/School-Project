from collections import deque
import os
from Dataset.Images import Images
from Information import Wikipedia_Information

def load_to_deque():
    with open("animals.list", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return deque(lines)

def download_images(animals):
    animals = list(animals)
    for animal, animal_name in zip(animals[::2], animals[1::2]):
        animal = animal.replace('\n', '')
        os.makedirs("Dataset/Train/" + animal, exist_ok=True)
        print(animal, ": ", animal_name)
        Wikipedia_Information(animal)
        Images(animal)

animals_list = load_to_deque()

download_images(animals_list)
