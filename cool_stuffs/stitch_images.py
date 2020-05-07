import sys
from PIL import Image
from glob import glob
from pathlib import Path


def stitch_images_horizontally(inp_files, out_file, spacing=20):
    images = [Image.open(x) for x in inp_files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)+spacing*len(inp_files)
    max_height = min(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]+20
    new_im.save(out_file)


def stitch_images_vertically(inp_files, out_file, spacing=20):
    images = [Image.open(x) for x in inp_files]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)+spacing*len(inp_files)
    max_width = min(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (0, x_offset))
        x_offset += im.size[1]+20
    new_im.save(out_file)


if __name__ == "__main__":
    base_path = Path.home()/"Downloads"
    inp_files = [base_path/"gelu.png", base_path/"gelu-1.png"]
    out_file = base_path/"gelu.png"
    stitch_images_horizontally(inp_files, out_file, 30)

