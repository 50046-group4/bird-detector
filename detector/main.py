import asyncio
from io import BytesIO

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from typing import Dict
import tflite_runtime.interpreter as tflite
import redis
from multiprocessing import Process, Manager


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

def sub(rds: redis.client.Redis, latest_timestamps: Dict[str, str]):
    pubsub = rds.pubsub()
    pubsub.psubscribe("sentry.*.updates")
    for message in pubsub.listen():
        if message.get("type") == "message":
            # get the sentry name by inspecting the channel
            sentry_name = message.get("channel")[6]
            latest_timestamps[sentry_name] = str(message.get("data"))

def detection_worker(rds: redis.client.Redis, latest_timestamps: Dict[str, str]):
    whose_turn = "A"
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="tflite/model.tflite")
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    while True:
        # rotate turn
        whose_turn = "B" if whose_turn == "A" else "A"
        # get image from A using redis
        latest_timestamp = latest_timestamps[whose_turn]
        image_bytes = rds.get(f"sentry.{whose_turn}.raw_images.{latest_timestamp}")
        if image_bytes is None:
            continue
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((width, height), Image.ANTIALIAS)

        output_details = interpreter.get_output_details()
        tensor_index = interpreter.get_input_details()[0]["index"]
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])

        try:
          font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                    25)
        except IOError:
          print("Font not found, using default font.")
          font = ImageFont.load_default()

        draw_bounding_box_on_image(image,*boxes[0], list(ImageColor.colormap.values())[0], font, display_str_list=[str(round(scores[0]*10))+"%"])

        await rds.set(f"sentry.{whose_turn}.processed_image", image.tobytes())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hello world!")
    manager = Manager()
    latest_timestamps = manager.dict()
    latest_timestamps["A"] = "kek"
    rds = redis.Redis(host="redis")
    p1 = Process(target=sub, args=(rds, latest_timestamps))
    p2 = Process(target=detection_worker, args=(rds, latest_timestamps))
    p1.start()
    p2.start()
    p2.join()
