from PIL import Image


def resize_image(input_path, output_path, width, height):
    try:
        image = Image.open(input_path)
        resize_image = image.resize((width, height))
        resize_image.save(output_path)
        print(f"Image resized and saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")




if __name__ == "__main__":
    input_path = "./dog/Unknow_1.jpg"
    output_path = "./dog_resize/Unknow_1.jpg"
    width = 224
    height = 244

    resize_image(input_path, output_path, width, height)