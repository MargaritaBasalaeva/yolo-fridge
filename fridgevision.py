from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

def load_model(model_path="fridge_model.pt"):
    return YOLO(model_path)

def predict_image(model, image_path, conf=0.2):
    results = model.predict(image_path, save=True, conf=conf)
    return results[0]

def display_prediction(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Products")
    plt.show()

def count_products(results):
    boxes = results.boxes
    food = {}
    for cls_id in boxes.cls:
        name = results.names[int(cls_id)]
        food[name] = food.get(name, 0) + 1
    return food

def generate_shopping_list(detected, base_set):
    missing = {}
    for item, qty in base_set.items():
        detected_qty = detected.get(item, 0)
        if detected_qty < qty:
            missing[item] = qty - detected_qty
    return missing

if __name__ == "__main__":
    image_path = "example_fridge.jpg"

    base_products = {
        'apple': 3, 'banana': 1, 'eggs': 10, 'chicken': 1,
        'milk': 1, 'onion': 1, 'pepper': 1, 'cheese': 1
    }

    model = load_model()
    results = predict_image(model, image_path)

    result_image_path = results.save_dir + "/" + os.path.basename(results.path)
    display_prediction(result_image_path)

    detected_products = count_products(results)

    print("Твой базовый набор:")
    print(base_products)
    print("Что у тебя в холодильнике:")
    print(detected_products)

    shopping_list = generate_shopping_list(detected_products, base_products)
    print("Список покупок:")
    print(shopping_list)