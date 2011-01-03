import Image


def merge_images(images):
    """Return the images in `images` joined one next another"""
    images_pil = [Image.open(img) for img in images]

    width = sum(img.size[0] for img in images_pil)
    heigth = max(img.size[1] for img in images_pil)

    result = Image.new('RGBA', (width, heigth))

    pos = 0
    for img in images_pil:
        result.paste(img, (pos, 0))
        pos += img.size[0]
    return result
