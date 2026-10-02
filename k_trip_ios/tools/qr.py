"""Print a QR for the share link in the console and save it as a PNG with a caption.

Usage: python tools/qr.py <url> <png path> <label>
"""
import sys
import qrcode
from PIL import Image, ImageDraw, ImageFont

url, png, label = sys.argv[1], sys.argv[2], sys.argv[3]

qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(url)
qr.make(fit=True)
qr.print_ascii(invert=True)

code = qr.make_image(fill_color="black", back_color="white").get_image().convert("RGB")
code = code.resize((720, 720), Image.NEAREST)
big = small = None
for bold, regular in (("malgunbd.ttf", "malgun.ttf"), ("arialbd.ttf", "arial.ttf")):  # Malgun Gothic covers Korean PC names
    try:
        big, small = ImageFont.truetype(bold, 40), ImageFont.truetype(regular, 22)
        break
    except OSError:
        pass
if big is None:
    big = small = ImageFont.load_default()
card = Image.new("RGB", (720, 860), "white")
card.paste(code, (0, 0))
draw = ImageDraw.Draw(card)
draw.text((360, 760), label, fill="black", font=big, anchor="mm")
draw.text((360, 820), url, fill="#444444", font=small, anchor="mm")
card.save(png)
