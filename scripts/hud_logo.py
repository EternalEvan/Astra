from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("wasd_ui", exist_ok=True)

# UI sizes (small)
key_size = (48, 48)
corner = 10
bg_padding = 6
font = ImageFont.truetype("arial.ttf", 28)  # 替换成本地支持的字体

def rounded_rect(im, bbox, radius, fill):
    draw = ImageDraw.Draw(im, "RGBA")
    draw.rounded_rectangle(bbox, radius=radius, fill=fill)

# background plate
bg_width = key_size[0] * 3 + bg_padding * 4
bg_height = key_size[1] * 2 + bg_padding * 4
ui_bg = Image.new("RGBA", (bg_width, bg_height), (0,0,0,0))
rounded_rect(ui_bg, (0,0,bg_width,bg_height), corner, (0,0,0,140))
ui_bg.save("wasd_ui/ui_background.png")

keys = ["W","A","S","D"]

def draw_key(char, active):
    im = Image.new("RGBA", key_size, (0,0,0,0))
    rounded_rect(im, (0,0,key_size[0],key_size[1]), corner,
                 (255,255,255,230) if active else (200,200,200,180))
    draw = ImageDraw.Draw(im)
    color = (0,0,0) if active else (50,50,50)
    w,h = draw.textsize(char, font=font)
    draw.text(((key_size[0]-w)//2,(key_size[1]-h)//2),
              char, font=font, fill=color)
    return im

for k in keys:
    draw_key(k, False).save(f"wasd_ui/key_{k}_idle.png")
    draw_key(k, True).save(f"wasd_ui/key_{k}_active.png")

print("✅ WASD UI assets generated in ./wasd_ui/")