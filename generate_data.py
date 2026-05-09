# The object of this file is to generate dirty captchas for our model to train on, the length of captchas should vary, as well as fonts and noise, making high-variability data

# For now(until this comment is removed on github), this file can be ignored since 3rd part library is used for generating Math captcha images

from PIL import Image, ImageDraw, ImageFont
import random


def get_random_length(minimum:int=4,maximum:int=8):
    #Return the random number for length of the captcha, can specify minimum and maximum number of character

    return random.randint(minimum,maximum)


def get_random_font():
    pass
#First step: Testine pillow capabilities, generating one image at a time that will satisfy captcha constraints until moving further
def generate_image(img_name:str)->bool:


    random.seed(1) #Set seed for reproducibility

    #Create a random number to draw
    num = random.randint(1000,9999)
    print(num)


    #Create an image
    out = Image.new("RGB", (150, 100), (255, 255, 255))

    d = ImageDraw.Draw(out)

    # Draw the number
    d.text((10, 10), text=str(num), fill=(0, 0, 0))

    #Show the image
    out.show()


if __name__=="__main__":
    generate_image("images//captcha.png")
