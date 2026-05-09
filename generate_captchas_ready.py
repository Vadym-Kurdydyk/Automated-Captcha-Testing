import os
import random
from multicolorcaptcha import CaptchaGenerator

# Configuration
TOTAL_IMAGES = 60000
OUTPUT_DIR = "diverse_math_dataset"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def generate_diverse_dataset():
    # Use size 2 (640x360) for better training resolution
    generator = CaptchaGenerator(2)

    for i in range(TOTAL_IMAGES):
        # Manually vary difficulty and operators if the default is stuck
        diff = random.randint(1, 5)

        # This call handles the image creation
        math_captcha = generator.gen_math_captcha_image(difficult_level=diff)

        result = math_captcha.equation_result
        equation = math_captcha.equation_str.strip()

        # Safety check: if you're still getting too many zeros,
        # you can skip or re-roll here
        if result == 0 and random.random() > 0.1:  # Allow only 10% zeros
            # Re-generate once to find a non-zero
            math_captcha = generator.gen_math_captcha_image(difficult_level=diff + 1)
            result = math_captcha.equation_result
            equation = math_captcha.equation_str.strip()

        # Sanitize filename (remove spaces/special chars)
        clean_eq = equation.replace(" ", "").replace("*", "x")
        filename = f"res_{result}_eq_{clean_eq}_{i}.png"

        math_captcha.image.save(os.path.join(OUTPUT_DIR, filename))

        if i % 1000 == 0:
            print(f"Generated {i} images...")


if __name__ == "__main__":
    generate_diverse_dataset()